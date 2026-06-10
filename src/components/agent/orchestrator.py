"""
Orchestrator — executes a TaskPlan DAG, dispatching SubAgents in parallel.

WHY the Orchestrator is separate from the SubAgent:
  The Orchestrator reasons about the GRAPH of tasks — which can run now,
  which are blocked, which failed. The SubAgent reasons about ONE task in
  depth. Mixing these two concerns would produce an unmaintainable mess.

Execution model:
  1. Get the TaskPlan from the Planner
  2. Find all tasks with no unsatisfied dependencies (ready_tasks)
  3. If mode=PARALLEL: submit ready tasks to ThreadPoolExecutor simultaneously
     If mode=DIRECT: run tasks one at a time (respecting dependencies)
  4. As tasks complete, mark them done and check if new tasks are now unblocked
  5. If a task fails and other tasks depend on it, mark those as SKIPPED
  6. When all tasks are resolved, pass results to Synthesiser

DESIGN NOTE on ThreadPoolExecutor vs asyncio:
  We use threads, not asyncio. Reasons:
  1. Our LLM calls (httpx) are synchronous — wrapping them in asyncio adds
     complexity with no real benefit here.
  2. SubAgents are CPU-light (mostly waiting on HTTP) and IO-light (small file ops).
     Python's GIL doesn't hurt us — threads release it during IO.
  3. ThreadPoolExecutor gives us clean futures, timeouts, and exception isolation
     out of the box.

DESIGN NOTE on per-task ToolRegistry:
  Each SubAgent gets its OWN ToolRegistry instance pointing at the same project
  root. This means their WriteFileTool audit logs are independent. But they all
  write to the same filesystem — a subtask that creates a file IS visible to
  other subtasks. This is intentional: subtask A can create a file that subtask
  B then reads. The Synthesiser handles conflict detection.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, Future, as_completed
from pathlib import Path
from typing import Optional

from .models import (
    AgentMode, SubTask, SubTaskStatus, TaskPlan, StepTrace, AgentResult
)
from .router import ModelRouter
from .subagent import SubAgent
from .synthesiser import Synthesiser

logger = logging.getLogger(__name__)

# Max parallel subagents - keeps API rate limits manageable
MAX_WORKERS = 4
# Per-subtask wall-clock timeout in seconds
SUBTASK_TIMEOUT = 300


class Orchestrator:
    """Executes a TaskPlan, managing SubAgent lifecycle and parallelism.

    Usage:
        orch = Orchestrator(router, project_root, store, pipeline)
        result = orch.execute(plan, conversation_history)
    """

    def __init__(
        self,
        router: ModelRouter,
        project_root: str | Path,
        vector_store=None,
        embed_pipeline=None,
    ):
        self._router = router
        self._root = Path(project_root)
        self._store = vector_store
        self._pipeline = embed_pipeline
        self._synthesiser = Synthesiser(router)


    # Public

    def execute(
        self,
        plan: TaskPlan,
        conversation_history: Optional[list[dict]] = None,
    ) -> AgentResult:
        """Execute the full TaskPlan and return a synthesised AgentResult.

        Args:
            plan:                   TaskPlan from the Planner.
            conversation_history:   Prior session turns for multi-turn context.

        Returns:
            AgentResult with final response, all traces, and side-effect lists.
        """
        t0 = time.monotonic()
        logger.info("Orchestrator: %s", plan.summary())

        if plan.mode == AgentMode.DIRECT:
            all_traces = self._run_direct(plan, conversation_history)
        else:
            all_traces = self._run_parallel(plan, conversation_history)

        # Collect side effects from all subtasks
        files_modified = self._collect_files_modified(plan)
        commands_run = self._collect_commands_run(plan)

        # Synthesise results into one coherent response
        result = self._synthesiser.synthesise(
            plan=plan,
            all_traces=all_traces,
            conversation_history=conversation_history or [],
        )

        result.files_modified = files_modified
        result.commands_run = commands_run
        result.total_steps = len(all_traces)
        result.total_latency_ms = (time.monotonic() - t0) * 1000

        logger.info(
            "Orchestrator done: %d steps, %.0fms, %d files modified",
            result.total_steps, result.total_latency_ms, len(files_modified)
        )
        return result


    # Execution modes

    def _run_direct(
        self,
        plan: TaskPlan,
        conversation_history: Optional[list[dict]],
    ) -> list[StepTrace]:
        """Execute tasks sequentially, respecting dependency order."""
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

                if subtask.status == SubTaskStatus.DONE:
                    completed.add(subtask.id)
                else:
                    # Mark dependents as skipped
                    self._skip_dependents(plan, subtask.id)
                    completed.add(subtask.id)

        return all_traces

    def _run_parallel(
        self,
        plan: TaskPlan,
        conversation_history: Optional[list[dict]],
    ) -> list[StepTrace]:
        """Execute independent tasks in parallel, blocked tasks sequentially after."""
        all_traces: list[StepTrace] = []
        completed: set[str] = set()

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            while not plan.is_complete():
                ready = plan.ready_tasks(completed)
                if not ready:
                    break

                # Submit all ready tasks in parallel
                futures: dict[Future, SubTask] = {}
                for subtask in ready:
                    if subtask.spawn_subagent:
                        future = executor.submit(
                            self._run_one_task, subtask, conversation_history
                        )
                        futures[future] = subtask
                        logger.info("Submitted subagent for: %s", subtask.id)
                    else:
                        # Simple tasks run inline (no thread overhead)
                        traces = self._run_one_task(subtask, conversation_history)
                        all_traces.extend(traces)
                        completed.add(subtask.id)
                        if subtask.status != SubTaskStatus.DONE:
                            self._skip_dependents(plan, subtask.id)

                # Collect futures
                for future in as_completed(futures, timeout=SUBTASK_TIMEOUT):
                    subtask = futures[future]
                    try:
                        traces = future.result(timeout=1)
                        all_traces.extend(traces)
                    except Exception as exc:
                        logger.error("SubAgent %s raised: %s", subtask.id, exc)
                        subtask.status = SubTaskStatus.FAILED
                        subtask.error = str(exc)

                    completed.add(subtask.id)
                    if subtask.status != SubTaskStatus.DONE:
                        self._skip_dependents(plan, subtask.id)

        return all_traces


    # Task execution

    def _run_one_task(
        self,
        subtask: SubTask,
        conversation_history: Optional[list[dict]],
    ) -> list[StepTrace]:
        """Run a single subtask — either via SubAgent loop or inline for simple tasks."""
        from ..tools.file_tools import ToolRegistry

        # Each SubAgent gets its own ToolRegistry (isolated audit log)
        tool_registry = ToolRegistry(self._root)

        agent = SubAgent(
            subtask=subtask,
            router=self._router,
            tool_registry=tool_registry,
            vector_store=self._store,
            embed_pipeline=self._pipeline,
            conversation_history=conversation_history,
        )

        try:
            agent.run()
        except Exception as exc:
            logger.error("SubAgent for %s crashed: %s", subtask.id, exc)
            subtask.status = SubTaskStatus.FAILED
            subtask.error = str(exc)

        return agent.traces


    # Helpers

    def _skip_dependents(self, plan: TaskPlan, failed_id: str) -> None:
        """Mark all tasks that depend on a failed task as SKIPPED."""
        for task in plan.subtasks:
            if failed_id in task.dependencies and task.status == SubTaskStatus.PENDING:
                task.status = SubTaskStatus.SKIPPED
                logger.info("Skipped %s (dependency %s failed)", task.id, failed_id)
                # Cascade: skip tasks that depend on this skipped task
                self._skip_dependents(plan, task.id)

    def _collect_files_modified(self, plan: TaskPlan) -> list[str]:
        """Collect unique file paths modified across all subtasks."""
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
        """Collect commands run across all subtasks."""
        result = []
        for subtask in plan.subtasks:
            for trace in subtask.traces:
                if trace.tool_name == "run_command":
                    cmd = trace.tool_args.get("command", "")
                    if cmd:
                        result.append(cmd)
        return result