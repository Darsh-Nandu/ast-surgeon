"""
SubAgent — a proper ReAct (Reason + Act) loop for one subtask.

KEY IMPROVEMENTS over the original:
  1. TRUE TOOL-CALL / OBSERVE CYCLE
     Each step: LLM reasons → picks tool → tool runs → result fed back as
     a structured "tool_result" role message → LLM sees it next step.
     The LLM is not just appended text — it receives a proper observation.

  2. SUBAGENT AWAITING SIGNAL
     When a parent spawns a child, the parent marks its signal as RUNNING
     and polls child.signal. The parent blocks (in-thread) until the child
     emits DONE or FAILED, then reads the result. No fire-and-forget.

  3. PIPELINE HEALTH & SLEEP MODE
     The agent tracks:
       - consecutive tool errors
       - JSON parse failures
       - repeated identical tool calls (stuck loop)
     When thresholds are crossed, it sets health.sleep_mode=True and
     emits a HealthSignal. The Orchestrator reads this and can trigger
     a future RepairAgent.

  4. RICH SYSTEM PROMPT WITH FULL TOOL SCHEMAS
     Tools are described with complete parameter schemas, not just names.
     The LLM has enough detail to call tools correctly on first attempt.

  5. STEP HISTORY IS A PROPER CONVERSATION
     Messages alternate user/assistant and include structured tool results
     so the LLM always has a coherent conversation to reason from.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from typing import Optional, TYPE_CHECKING

from .models import (
    AgentMode, SubTask, SubTaskStatus, StepTrace, TaskType,
    PipelineHealth, HealthSignal, SleepReason, SubAgentSignal,
)
from .router import ModelRouter

if TYPE_CHECKING:
    from ..tools.file_tools import ToolRegistry
    from ..vectorstore.qdrant_store import VectorStore
    from ..embeddings.pipeline import EmbeddingPipeline

logger = logging.getLogger(__name__)

ABSOLUTE_MAX_STEPS = 20

# ─── Health thresholds ────────────────────────────────────────────────────────

PARSE_FAILURE_SLEEP_THRESHOLD    = 3
CONSECUTIVE_ERROR_SLEEP_THRESHOLD = 4
LLM_ERROR_SLEEP_THRESHOLD        = 3
STUCK_LOOP_WINDOW                = 3    # N identical calls in a row = stuck

# ─── System prompt ────────────────────────────────────────────────────────────

SUBAGENT_SYSTEM = """\
You are SubAgent {agent_id} of Sovereign-Code, an expert autonomous coding assistant.

ASSIGNED TASK: {task_description}
TASK TYPE: {task_type}

## Available Tools

{tool_schemas_detailed}

## Response Format

You MUST reply with a single valid JSON object and NOTHING else:

{{
  "thought": "Step-by-step reasoning: what do I know, what do I need, what is the best next action?",
  "action": "<tool_name> | done | spawn_subagent",
  "args": {{ ...tool arguments exactly matching the schema above... }},
  "final_answer": "Full answer / result summary. REQUIRED when action=done, omit otherwise."
}}

When action=spawn_subagent, args MUST be:
{{
  "description": "Precise description of the sub-task",
  "task_type": "code_gen|code_edit|debug|test_write|search|explain",
  "context_hint": "keywords for vector retrieval"
}}

## Rules

1. READ before EDIT — always read_file before write_file or edit_file.
2. VERIFY after WRITE — after editing code, run_command to lint/test.
3. SEARCH before ASSUME — use search_files or list_dir to find real paths.
4. THINK STEP-BY-STEP — your "thought" must explain your reasoning, not just state the action.
5. SPAWN SPARINGLY — only spawn a subagent for a genuinely isolated sub-problem.
6. DONE CORRECTLY — set action=done only when the task is fully complete. Always set final_answer.
7. STUCK? — if 2 consecutive attempts at the same thing fail, try a completely different approach.
"""


def _format_tool_schemas(schemas: list[dict]) -> str:
    """Format tool schemas into a readable block for the system prompt."""
    lines = []
    for s in schemas:
        lines.append(f"### {s['name']}")
        lines.append(f"{s['description']}")
        params = s.get("parameters", {})
        if params:
            lines.append("Parameters:")
            for pname, pinfo in params.items():
                optional = " (optional)" if pinfo.get("optional") else ""
                lines.append(f"  - {pname} ({pinfo.get('type','any')}){optional}: {pinfo.get('description','')}")
        lines.append("")
    return "\n".join(lines)


# ─── SubAgent ─────────────────────────────────────────────────────────────────

class SubAgent:
    """
    Runs a true ReAct loop for one SubTask.

    The loop structure per step:
      1. OBSERVE  — vector-retrieve fresh context
      2. THINK    — LLM sees full conversation history + retrieved context
      3. ACT      — execute the chosen tool
      4. OBSERVE RESULT — tool result is appended as a structured message
      5. CHECK HEALTH — detect stuck/error conditions, enter sleep if needed
      6. CHECK DONE — exit if LLM said done or max_steps hit

    Child SubAgents:
      When spawn_subagent is chosen, the parent creates a child SubAgent,
      waits for it to complete (polls child.signal with a timeout), then
      injects the child's result as a tool_result message and continues.
    """

    def __init__(
        self,
        subtask: SubTask,
        router: ModelRouter,
        tool_registry: "ToolRegistry",
        vector_store: Optional["VectorStore"] = None,
        embed_pipeline: Optional["EmbeddingPipeline"] = None,
        agent_id: Optional[str] = None,
        conversation_history: Optional[list[dict]] = None,
        depth: int = 0,             # nesting depth (0 = top-level subagent)
        parent_id: Optional[str] = None,
    ):
        self._subtask = subtask
        self._router = router
        self._tools = tool_registry
        self._store = vector_store
        self._pipeline = embed_pipeline
        self._agent_id = agent_id or f"agent-{uuid.uuid4().hex[:6]}"
        self._conversation_history = conversation_history or []
        self._depth = depth
        self._parent_id = parent_id

        # Per-agent state
        self._step_history: list[dict] = []     # the live ReAct conversation
        self._traces: list[StepTrace] = []
        self._files_modified: list[str] = []
        self._commands_run: list[str] = []
        self._health = PipelineHealth()

        # Threading signal so a parent can poll this agent's state
        self._signal_lock = threading.Lock()
        self._subtask.signal = SubAgentSignal.PENDING

    # ─── Public ──────────────────────────────────────────────────────────────

    def run(self) -> SubTask:
        """Execute the subtask loop. Returns the mutated SubTask with results."""
        self._set_signal(SubAgentSignal.RUNNING)
        self._subtask.status = SubTaskStatus.RUNNING
        self._subtask.agent_id = self._agent_id

        logger.info(
            "[%s] Starting (depth=%d): %s (max_steps=%d)",
            self._agent_id, self._depth, self._subtask.description[:70],
            self._subtask.max_steps,
        )

        max_steps = min(self._subtask.max_steps, ABSOLUTE_MAX_STEPS)

        for step_num in range(1, max_steps + 1):

            # ── Health gate: enter sleep before executing step ────────────────
            should_sleep, reason = self._health.should_sleep()
            if should_sleep:
                self._enter_sleep(reason)
                break

            trace = self._step(step_num)
            self._traces.append(trace)
            self._subtask.traces.append(trace)

            # ── Done? ────────────────────────────────────────────────────────
            if trace.tool_name is None and not trace.tool_error:
                self._subtask.status = SubTaskStatus.DONE
                self._subtask.result = self._extract_final_answer(trace)
                self._set_signal(SubAgentSignal.DONE)
                logger.info("[%s] Done at step %d", self._agent_id, step_num)
                break

        else:
            # Hit max_steps — treat as done with best-effort result
            logger.warning("[%s] Hit max_steps=%d", self._agent_id, max_steps)
            self._subtask.status = SubTaskStatus.DONE
            self._subtask.result = self._best_effort_result()
            self._set_signal(SubAgentSignal.DONE)

        self._subtask.health = self._health
        return self._subtask

    # ─── Core ReAct step ─────────────────────────────────────────────────────

    def _step(self, step_num: int) -> StepTrace:
        """One full Observe → Think → Act → Observe-Result cycle."""
        t0 = time.monotonic()

        # 1. OBSERVE — fresh vector context for current task state
        context = self._observe()

        # 2. THINK — build step message and call LLM
        step_msg = self._build_step_message(step_num, context)
        self._step_history.append({"role": "user", "content": step_msg})

        system_prompt = self._build_system_prompt()
        llm_response = self._router.call(
            task_type=self._subtask.task_type,
            system_prompt=system_prompt,
            messages=self._step_history,
        )

        if llm_response.is_error:
            self._health.total_llm_errors += 1
            self._health.record(HealthSignal(
                agent_id=self._agent_id,
                signal=SleepReason.LLM_ERROR_RATE,
                detail=f"step {step_num}: {llm_response.content[:120]}",
                severity=2,
            ))
            # Record the failed response so history stays coherent
            self._step_history.append({
                "role": "assistant",
                "content": json.dumps({
                    "thought": "LLM call failed",
                    "action": "done",
                    "args": {},
                    "final_answer": f"[LLM error at step {step_num}]",
                })
            })
            latency = (time.monotonic() - t0) * 1000
            return StepTrace(
                step_number=step_num,
                agent_id=self._agent_id,
                task_type=self._subtask.task_type,
                model_used=llm_response.model_used,
                thought="[LLM error]",
                tool_name=None,
                tool_args={},
                tool_result=llm_response.content,
                tool_result_raw=llm_response.content,
                tool_error=True,
                latency_ms=latency,
                health_flag=True,
            )

        # Record assistant turn
        self._step_history.append({
            "role": "assistant",
            "content": llm_response.content,
        })

        # 3. PARSE
        action, parse_ok = self._parse_action(llm_response.content)
        if not parse_ok:
            self._health.parse_failures += 1
            self._health.record(HealthSignal(
                agent_id=self._agent_id,
                signal=SleepReason.LLM_PARSE_FAILURES,
                detail=f"step {step_num}: {llm_response.content[:120]}",
                severity=1,
            ))

        # 4. ACT
        tool_name, tool_args, tool_result_raw, tool_error = self._act(action)

        # 5. OBSERVE RESULT — inject tool result as structured user message
        if tool_name and action.get("action") not in ("done", "spawn_subagent"):
            status_prefix = "ERROR" if tool_error else "OK"
            obs_message = (
                f"[TOOL_RESULT: {tool_name}]\n"
                f"Status: {status_prefix}\n"
                f"{tool_result_raw or '(no output)'}"
            )
            self._step_history.append({"role": "user", "content": obs_message})

            # Track error streak
            if tool_error:
                self._health.consecutive_tool_errors += 1
                if self._health.consecutive_tool_errors >= CONSECUTIVE_ERROR_SLEEP_THRESHOLD:
                    self._health.record(HealthSignal(
                        agent_id=self._agent_id,
                        signal=SleepReason.CONSECUTIVE_ERRORS,
                        detail=f"{self._health.consecutive_tool_errors} consecutive errors",
                        severity=3,
                    ))
            else:
                self._health.consecutive_tool_errors = 0  # reset on success

        latency = (time.monotonic() - t0) * 1000
        health_flag = (
            self._health.parse_failures > 0
            or self._health.consecutive_tool_errors >= 2
            or tool_error
        )

        return StepTrace(
            step_number=step_num,
            agent_id=self._agent_id,
            task_type=self._subtask.task_type,
            model_used=llm_response.model_used,
            thought=action.get("thought", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=(tool_result_raw or "")[:500],
            tool_result_raw=tool_result_raw,
            tool_error=tool_error,
            latency_ms=latency,
            health_flag=health_flag,
        )

    # ─── Observe ─────────────────────────────────────────────────────────────

    def _observe(self) -> str:
        """Retrieve relevant context from the vector store for current task state."""
        if not self._store or not self._pipeline:
            return ""

        query_parts = [self._subtask.description]
        if self._subtask.context_hint:
            query_parts.append(self._subtask.context_hint)
        # Fold in the last tool result as a relevance signal
        for msg in reversed(self._step_history[-4:]):
            if msg["role"] == "user" and msg["content"].startswith("[TOOL_RESULT"):
                lines = msg["content"].splitlines()
                query_parts.append(" ".join(lines[2:5]))
                break

        query = " ".join(query_parts)[:300]
        try:
            query_vec = self._pipeline.embed_query(query)
            results = self._store.search(query_vec, top_k=5)
            if not results:
                return ""
            chunks = []
            for r in results:
                c = r.chunk
                chunks.append(
                    f"### {c.name or 'block'} ({c.file_path}:{c.start_line}-{c.end_line})\n"
                    f"```{c.language}\n{c.content[:600]}\n```"
                )
            return "\n\n".join(chunks)
        except Exception as exc:
            logger.debug("Observe failed: %s", exc)
            return ""

    # ─── Act ─────────────────────────────────────────────────────────────────

    def _act(
        self, action: dict
    ) -> tuple[Optional[str], dict, Optional[str], bool]:
        """Execute the action chosen by the LLM.

        Returns (tool_name, tool_args, result_str, is_error).
        """
        action_type = action.get("action", "done")
        args = action.get("args") or {}

        if action_type == "done":
            return None, {}, None, False

        if action_type == "spawn_subagent":
            result_str = self._spawn_and_await_child(args)
            return "spawn_subagent", args, result_str, False

        if action_type not in self._tools.available:
            error_msg = (
                f"Unknown tool: {action_type!r}. "
                f"Available tools: {', '.join(self._tools.available)}"
            )
            return action_type, args, error_msg, True

        tool_result = self._tools.run(action_type, **args)

        if action_type in ("write_file", "edit_file"):
            path = args.get("path", "")
            if path and path not in self._files_modified:
                self._files_modified.append(path)
        if action_type == "run_command":
            cmd = args.get("command", "")
            if cmd:
                self._commands_run.append(cmd)

        return action_type, args, tool_result.content, tool_result.is_error

    def _spawn_and_await_child(self, args: dict) -> str:
        """
        Spawn a child SubAgent and WAIT for it to finish.

        The parent SubAgent is paused (blocks in this method) until the child
        emits DONE or FAILED on its signal. This guarantees the parent sees
        the child's output before continuing its own loop.

        DESIGN: child runs in the same thread (no new thread). The parent
        literally cannot continue until the child returns. This is intentional
        for child spawns mid-loop; top-level parallelism is the Orchestrator's job.
        """
        from .models import SubTask, TaskType

        description = args.get("description", "(no description)")
        logger.info(
            "[%s] Spawning child subagent: %s",
            self._agent_id, description[:60]
        )

        try:
            task_type = TaskType(args.get("task_type", "code_gen"))
        except ValueError:
            task_type = TaskType.CODE_GEN

        child_task = SubTask(
            id=f"{self._subtask.id}_child_{len(self._traces)}",
            description=description,
            task_type=task_type,
            dependencies=[],
            spawn_subagent=False,
            context_hint=args.get("context_hint", ""),
            max_steps=8,
        )

        child_agent = SubAgent(
            subtask=child_task,
            router=self._router,
            tool_registry=self._tools,
            vector_store=self._store,
            embed_pipeline=self._pipeline,
            agent_id=f"{self._agent_id}▸ch{len(self._traces)}",
            depth=self._depth + 1,
            parent_id=self._agent_id,
        )

        # ── Parent sets its own signal to RUNNING (in case it was checked) ──
        self._set_signal(SubAgentSignal.RUNNING)

        # ── Run child synchronously — parent WAITS here ──────────────────────
        logger.info(
            "[%s] Waiting for child %s ...",
            self._agent_id, child_agent._agent_id
        )
        completed_child_task = child_agent.run()

        # ── Child is done: check signal ──────────────────────────────────────
        child_signal = completed_child_task.signal
        logger.info(
            "[%s] Child %s finished with signal=%s",
            self._agent_id, child_agent._agent_id, child_signal.value
        )

        # Merge child traces and side effects into parent
        self._traces.extend(completed_child_task.traces)
        self._files_modified.extend(child_agent._files_modified)
        self._commands_run.extend(child_agent._commands_run)

        # Propagate child health signals to parent
        for signal in completed_child_task.health.signals:
            self._health.signals.append(signal)

        # If child entered sleep, flag it in parent health too
        if completed_child_task.health.sleep_mode:
            self._health.record(HealthSignal(
                agent_id=self._agent_id,
                signal=SleepReason.SUBAGENT_TIMEOUT,
                detail=f"child {child_agent._agent_id} entered sleep mode",
                severity=2,
            ))

        if child_signal == SubAgentSignal.FAILED or child_signal == SubAgentSignal.SLEEPING:
            return (
                f"[CHILD AGENT SIGNAL: {child_signal.value}]\n"
                f"Child task: {description}\n"
                f"Result: {completed_child_task.result or completed_child_task.error or '(no output)'}"
            )

        return (
            f"[CHILD AGENT DONE]\n"
            f"Child task: {description}\n"
            f"Result:\n{completed_child_task.result or '(child completed with no output)'}"
        )

    # ─── Sleep mode ──────────────────────────────────────────────────────────

    def _enter_sleep(self, reason: SleepReason) -> None:
        """Enter sleep mode — record state, set signals, log clearly."""
        self._health.sleep_mode = True
        self._health.sleep_reason = reason
        self._set_signal(SubAgentSignal.SLEEPING)
        self._subtask.status = SubTaskStatus.FAILED
        self._subtask.error = f"Sleep mode: {reason.value}"

        logger.warning(
            "[%s] ⚠ ENTERING SLEEP MODE — reason=%s | signals=%d",
            self._agent_id, reason.value, len(self._health.signals)
        )
        logger.warning(
            "[%s] Health snapshot: parse_failures=%d, consecutive_errors=%d, llm_errors=%d",
            self._agent_id,
            self._health.parse_failures,
            self._health.consecutive_tool_errors,
            self._health.total_llm_errors,
        )

    # ─── Prompt building ─────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        detailed_schemas = _format_tool_schemas(self._tools.schemas())
        return SUBAGENT_SYSTEM.format(
            agent_id=self._agent_id,
            task_description=self._subtask.description,
            task_type=self._subtask.task_type.value,
            tool_schemas_detailed=detailed_schemas,
        )

    def _build_step_message(self, step_num: int, context: str) -> str:
        parts = []
        max_steps = min(self._subtask.max_steps, ABSOLUTE_MAX_STEPS)
        parts.append(f"[Step {step_num}/{max_steps}]")

        if context:
            parts.append(f"\n--- Retrieved Codebase Context ---\n{context}\n---")

        if step_num == 1:
            parts.append(
                f"\nBegin working on your task:\n{self._subtask.description}"
            )
            if self._subtask.context_hint:
                parts.append(f"Hint: {self._subtask.context_hint}")
        else:
            # Remind agent of task without repeating full context each step
            parts.append(
                f"\nTask reminder: {self._subtask.description[:120]}\n"
                f"Continue. Analyse the tool result above and decide your next action."
            )

        return "\n".join(parts)

    # ─── Helpers ─────────────────────────────────────────────────────────────

    def _parse_action(self, raw: str) -> tuple[dict, bool]:
        """Parse LLM JSON. Returns (action_dict, parse_ok)."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            return json.loads(cleaned), True
        except json.JSONDecodeError:
            # Try to extract the outermost JSON object
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group()), True
                except json.JSONDecodeError:
                    pass

        logger.warning(
            "[%s] ⚠ JSON parse failed: %s...", self._agent_id, raw[:120]
        )
        return {
            "thought": "Could not parse structured response from LLM",
            "action": "done",
            "args": {},
            "final_answer": raw,
        }, False

    def _extract_final_answer(self, trace: StepTrace) -> str:
        """Pull final_answer from the last assistant message in history."""
        for msg in reversed(self._step_history):
            if msg["role"] == "assistant":
                action, _ = self._parse_action(msg["content"])
                if fa := action.get("final_answer"):
                    return fa
        return self._best_effort_result()

    def _best_effort_result(self) -> str:
        for trace in reversed(self._traces):
            if trace.tool_result and not trace.tool_error:
                return f"[Partial result from step {trace.step_number}]\n{trace.tool_result}"
        return f"[Incomplete after {len(self._traces)} steps]"

    def _set_signal(self, signal: SubAgentSignal) -> None:
        with self._signal_lock:
            self._subtask.signal = signal

    # ─── Properties ──────────────────────────────────────────────────────────

    @property
    def traces(self) -> list[StepTrace]:
        return list(self._traces)

    @property
    def files_modified(self) -> list[str]:
        return list(self._files_modified)

    @property
    def commands_run(self) -> list[str]:
        return list(self._commands_run)

    @property
    def health(self) -> PipelineHealth:
        return self._health