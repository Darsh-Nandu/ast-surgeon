"""
Orchestrator — executes a TaskPlan DAG, dispatching SubAgents in parallel.

KEY IMPROVEMENTS:
  1. HEALTH AGGREGATION
     After each subtask completes, the Orchestrator reads the subtask's
     PipelineHealth. If any subtask entered sleep mode, the Orchestrator
     logs it clearly and marks the final AgentResult.sleep_mode=True.

  2. SUBAGENT SIGNAL AWARENESS
     The Orchestrator checks subtask.signal after completion. A SLEEPING
     signal causes the task to be marked FAILED and dependents to be skipped,
     exactly as if the task had crashed. A repair hook is called (stub for now,
     will be wired to RepairAgent later).

  3. PARALLEL EXECUTION WITH HEALTH GATE
     If a sleeping subtask is detected mid-parallel run, remaining futures
     still complete (no brutal cancel) but the final result reports sleep mode.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import Optional

from .models import (
    AgentMode, AgentResult, PipelineHealth, SleepReason, SubAgentSignal,
    SubTask, SubTaskStatus, TaskPlan, StepTrace,
)
from .router import ModelRouter
from .subagent import SubAgent
from .synthesiser import Synthesiser

logger = logging.getLogger(__name__)

MAX_WORKERS = 4
SUBTASK_TIMEOUT = 300


class Orchestrator:
    """Executes a TaskPlan, managing SubAgent lifecycle and parallelism."""

    def __init__(
        self,
        router: ModelRouter,
        project_root: str | Path,
        vector_store=None,
        embed_pipeline=None,
        indexer=None,
    ):
        self._router = router
        self._root = Path(project_root)
        self._store = vector_store
        self._pipeline = embed_pipeline
        self._indexer = indexer          # optional live-reindex after writes
        self._synthesiser = Synthesiser(router)
        self._memory_coordinator = None  # set in execute()

    # ─── Public ──────────────────────────────────────────────────────────────

    def execute(
        self,
        plan: TaskPlan,
        conversation_history: Optional[list[dict]] = None,
        memory_coordinator=None,
    ) -> AgentResult:
        t0 = time.monotonic()
        logger.info("Orchestrator: %s", plan.summary())
        self._memory_coordinator = memory_coordinator  # passed to SubAgents

        if plan.mode == AgentMode.DIRECT:
            all_traces = self._run_direct(plan, conversation_history)
        else:
            all_traces = self._run_parallel(plan, conversation_history)

        # ── Aggregate health across all subtasks ──────────────────────────
        aggregate_health = self._aggregate_health(plan)

        # ── Collect side effects ──────────────────────────────────────────
        files_modified = self._collect_files_modified(plan)
        commands_run = self._collect_commands_run(plan)

        # ── Synthesise ────────────────────────────────────────────────────
        result = self._synthesiser.synthesise(
            plan=plan,
            all_traces=all_traces,
            conversation_history=conversation_history or [],
            aggregate_health=aggregate_health,
        )

        result.files_modified = files_modified
        result.commands_run = commands_run
        result.total_steps = len(all_traces)
        result.total_latency_ms = (time.monotonic() - t0) * 1000
        result.sleep_mode = aggregate_health.sleep_mode
        result.health_report = aggregate_health

        if aggregate_health.sleep_mode:
            logger.warning(
                "Orchestrator: pipeline entered SLEEP MODE — reason=%s",
                aggregate_health.sleep_reason.value if aggregate_health.sleep_reason else "?",
            )

        logger.info(
            "Orchestrator done: %d steps, %.0fms, %d files modified, sleep=%s",
            result.total_steps, result.total_latency_ms,
            len(files_modified), aggregate_health.sleep_mode,
        )
        return result

    # ─── Execution modes ─────────────────────────────────────────────────────

    def _run_direct(
        self,
        plan: TaskPlan,
        conversation_history: Optional[list[dict]],
    ) -> list[StepTrace]:
        all_traces: list[StepTrace] = []
        completed: set[str] = set()

        while not plan.is_complete():
            ready = plan.ready_tasks(completed)
            if not ready:
                logger.warning("No ready tasks but plan not complete — possible cycle")
                break

            for subtask in ready:
                traces = self._run_one_task(subtask, conversation_history)
                all_traces.extend(traces)
                self._handle_post_task(plan, subtask, completed)

        return all_traces

    def _run_parallel(
        self,
        plan: TaskPlan,
        conversation_history: Optional[list[dict]],
    ) -> list[StepTrace]:
        all_traces: list[StepTrace] = []
        completed: set[str] = set()
        pipeline_sleeping = False  # set True if any subtask enters sleep

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while not plan.is_complete():
                ready = plan.ready_tasks(completed)
                if not ready:
                    break

                # If a previous wave entered sleep, skip remaining work
                if pipeline_sleeping:
                    logger.warning(
                        "Orchestrator: skipping %d ready tasks — pipeline in sleep mode",
                        len(ready),
                    )
                    for subtask in ready:
                        subtask.status = SubTaskStatus.SKIPPED
                        completed.add(subtask.id)
                    continue

                futures: dict[Future, SubTask] = {}
                for subtask in ready:
                    if subtask.spawn_subagent:
                        future = executor.submit(
                            self._run_one_task, subtask, conversation_history
                        )
                        futures[future] = subtask
                        logger.info(
                            "Orchestrator: submitted subagent for subtask=%s", subtask.id
                        )
                    else:
                        # Simple tasks run inline in the orchestrator thread
                        traces = self._run_one_task(subtask, conversation_history)
                        all_traces.extend(traces)
                        self._handle_post_task(plan, subtask, completed)
                        if subtask.signal == SubAgentSignal.SLEEPING:
                            pipeline_sleeping = True

                for future in as_completed(futures, timeout=SUBTASK_TIMEOUT):
                    subtask = futures[future]
                    try:
                        traces = future.result(timeout=1)
                        all_traces.extend(traces)
                    except Exception as exc:
                        logger.error("SubAgent %s raised: %s", subtask.id, exc)
                        subtask.status = SubTaskStatus.FAILED
                        subtask.error = str(exc)

                    self._handle_post_task(plan, subtask, completed)

                    # Check if this completion triggered sleep
                    if subtask.signal == SubAgentSignal.SLEEPING:
                        pipeline_sleeping = True
                        logger.warning(
                            "Orchestrator: subtask %s is SLEEPING — "
                            "remaining futures will complete but no new waves start.",
                            subtask.id,
                        )

        return all_traces

    # ─── Task execution ──────────────────────────────────────────────────────

    def _run_one_task(
        self,
        subtask: SubTask,
        conversation_history: Optional[list[dict]],
    ) -> list[StepTrace]:
        from ..tools.file_tools import ToolRegistry

        # Wire live-reindex: after every write_file/edit_file the file is
        # immediately re-embedded into the session Qdrant collection so
        # subsequent SubAgents can find it via _observe().
        on_file_write = None
        if self._indexer is not None:
            def on_file_write(path: str, _idx=self._indexer) -> None:
                try:
                    result = _idx.reindex_file(path)
                    import logging
                    logging.getLogger(__name__).debug(
                        "Auto-reindexed %s: +%d/-%d chunks",
                        path, result.chunks_added, result.chunks_deleted,
                    )
                except Exception as exc:
                    import logging
                    logging.getLogger(__name__).warning(
                        "Auto-reindex failed for %s: %s", path, exc
                    )

        tool_registry = ToolRegistry(self._root, on_file_write=on_file_write)
        agent = SubAgent(
            subtask=subtask,
            router=self._router,
            tool_registry=tool_registry,
            vector_store=self._store,
            embed_pipeline=self._pipeline,
            conversation_history=conversation_history,
            memory_coordinator=self._memory_coordinator,
        )

        try:
            agent.run()
        except Exception as exc:
            logger.error("SubAgent for %s crashed: %s", subtask.id, exc)
            subtask.status = SubTaskStatus.FAILED
            subtask.error = str(exc)

        return agent.traces

    def _handle_post_task(
        self,
        plan: TaskPlan,
        subtask: SubTask,
        completed: set[str],
    ) -> None:
        """Post-task bookkeeping: signal checks, dependency propagation."""
        completed.add(subtask.id)

        # Check if subtask entered sleep mode
        if subtask.signal == SubAgentSignal.SLEEPING:
            logger.warning(
                "SubTask %s entered SLEEP MODE — skipping dependents. "
                "Future: RepairAgent will be invoked here.",
                subtask.id,
            )
            subtask.status = SubTaskStatus.FAILED
            if not subtask.error:
                subtask.error = f"sleep_mode: {subtask.health.sleep_reason}"
            # --- REPAIR HOOK (stub) ---
            self._repair_hook(subtask, plan)

        if subtask.status != SubTaskStatus.DONE:
            self._skip_dependents(plan, subtask.id)

    def _repair_hook(self, sleeping_subtask: SubTask, plan: TaskPlan) -> None:
        """
        Stub called when a subtask enters sleep mode.

        In a future iteration, this will:
          1. Instantiate a RepairAgent with the sleep reason and full health signals
          2. Let the RepairAgent diagnose and attempt to fix the pipeline
          3. If repair succeeds, re-queue the sleeping subtask

        For now: log clearly and mark as failed so the session reports it.
        """
        logger.warning(
            "[REPAIR_HOOK] SubTask %s slept: reason=%s, signals=%d. "
            "RepairAgent not yet implemented — marking failed.",
            sleeping_subtask.id,
            sleeping_subtask.health.sleep_reason,
            len(sleeping_subtask.health.signals),
        )

    # ─── Health aggregation ───────────────────────────────────────────────────

    def _aggregate_health(self, plan: TaskPlan) -> PipelineHealth:
        """Merge health from all subtasks into one aggregate view."""
        agg = PipelineHealth()
        for subtask in plan.subtasks:
            h = subtask.health
            agg.parse_failures += h.parse_failures
            agg.total_llm_errors += h.total_llm_errors
            agg.consecutive_tool_errors = max(
                agg.consecutive_tool_errors, h.consecutive_tool_errors
            )
            for signal in h.signals:
                agg.signals.append(signal)
            if h.sleep_mode:
                agg.sleep_mode = True
                agg.sleep_reason = h.sleep_reason  # last one wins (fine for now)
        return agg

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _skip_dependents(self, plan: TaskPlan, failed_id: str) -> None:
        for task in plan.subtasks:
            if failed_id in task.dependencies and task.status == SubTaskStatus.PENDING:
                task.status = SubTaskStatus.SKIPPED
                logger.info("Skipped %s (dependency %s failed)", task.id, failed_id)
                self._skip_dependents(plan, task.id)

    def _collect_files_modified(self, plan: TaskPlan) -> list[str]:
        seen = set()
        result = []
        for subtask in plan.subtasks:
            for trace in subtask.traces:
                if trace.tool_name in ("write_file", "edit_file"):
                    path = trace.tool_args.get("path", "")
                    if path and path not in seen:
                        seen.add(path)
                        result.append(path)
        return result

    def _collect_commands_run(self, plan: TaskPlan) -> list[str]:
        result = []
        for subtask in plan.subtasks:
            for trace in subtask.traces:
                if trace.tool_name == "run_command":
                    cmd = trace.tool_args.get("command", "")
                    if cmd:
                        result.append(cmd)
        return result