"""
SubAgent — a proper ReAct (Reason + Act) loop for one subtask.

DESIGN PRINCIPLES:
  1. STRICT MESSAGE ALTERNATION
     Every LLM call sees a history that ends with a user message.
     After each tool execution the result is injected as a user message
     so the next LLM call always follows the pattern:
       user → assistant → user → assistant → ...
     This is the invariant that makes Groq and Gemini both happy.

  2. TRUE TOOL-CALL / OBSERVE CYCLE
     Think → Act → Observe is one atomic unit. The observation (tool result
     or child agent result) is immediately appended to history before the
     next step so the LLM always reasons from up-to-date state.

  3. SPAWN-AND-AWAIT WITH PROPER SIGNALLING
     When spawn_subagent is chosen, the parent signals AWAITING (not just
     RUNNING), blocks until the child emits DONE/FAILED/SLEEPING, then
     injects the child's full output as a user observation so the parent
     can act on it in the very next step.

  4. STUCK LOOP DETECTION
     A rolling window of (tool_name, args_fingerprint) tuples detects
     when the agent is calling the same thing repeatedly. On detection
     a health signal is emitted and the agent is nudged to try a
     different approach before sleep mode is triggered.

  5. PIPELINE HEALTH & SLEEP MODE
     Tracks parse failures, consecutive tool errors, LLM errors, and
     stuck loops. When thresholds are crossed, health.sleep_mode=True
     and the Orchestrator is notified to invoke a future RepairAgent.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
import uuid
from collections import deque
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
    from ..memory.coordinator import MemoryCoordinator

logger = logging.getLogger(__name__)

ABSOLUTE_MAX_STEPS = 20

# ─── Health thresholds ────────────────────────────────────────────────────────

PARSE_FAILURE_SLEEP_THRESHOLD     = 3
CONSECUTIVE_ERROR_SLEEP_THRESHOLD = 4
LLM_ERROR_SLEEP_THRESHOLD         = 3
STUCK_LOOP_WINDOW                 = 3    # N identical calls in a row = stuck loop

# ─── System prompt ────────────────────────────────────────────────────────────

SUBAGENT_SYSTEM = """\
You are SubAgent {agent_id} of Sovereign-Code, an expert autonomous coding assistant.

ASSIGNED TASK: {task_description}
TASK TYPE: {task_type}

## Available Tools

{tool_schemas_detailed}

## ReAct Protocol

You operate in a strict Think → Act → Observe loop:
  1. THINK  — reason step-by-step about what you know and what to do next.
  2. ACT    — choose exactly one tool to call (or declare done / spawn a child).
  3. OBSERVE — you will receive the tool result as the next message.

Repeat until the task is fully complete, then set action=done.

## Response Format

You MUST reply with a single valid JSON object and NOTHING else (no markdown fences):

{{
  "thought": "Step-by-step reasoning: what do I know, what's missing, what is the best next action?",
  "action": "<tool_name> | done | spawn_subagent",
  "args": {{ ...tool arguments exactly matching the schema above... }},
  "final_answer": "Full answer / result summary. REQUIRED when action=done, omit otherwise."
}}

When action=spawn_subagent, args MUST be:
{{
  "description": "Precise self-contained description of the sub-task",
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
7. STUCK? — if 2+ consecutive attempts at the same thing fail, try a completely different approach.
8. OBSERVE — after every tool call you will see a [TOOL_RESULT] message. Read it carefully before acting.
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


def _args_fingerprint(tool_name: str, args: dict) -> str:
    """Stable fingerprint of a (tool_name, args) pair for stuck-loop detection."""
    args_str = json.dumps(args, sort_keys=True, default=str)
    digest = hashlib.md5(args_str.encode()).hexdigest()[:8]
    return f"{tool_name}:{digest}"


# ─── SubAgent ─────────────────────────────────────────────────────────────────

class SubAgent:
    """
    Runs a true ReAct loop for one SubTask.

    Conversation history invariant:
      Before every LLM call, the last message in _step_history is ALWAYS
      a user message. This guarantees strict user/assistant alternation.

    History structure:
      [user: initial task + context]
      [assistant: action_1]
      [user: TOOL_RESULT from action_1]
      [assistant: action_2]
      [user: TOOL_RESULT from action_2]
      ...
      [assistant: done + final_answer]
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
        depth: int = 0,
        parent_id: Optional[str] = None,
        memory_coordinator=None,
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
        self._memory_coordinator = memory_coordinator

        # Per-agent state
        self._step_history: list[dict] = []     # live ReAct conversation
        self._traces: list[StepTrace] = []
        self._files_modified: list[str] = []
        self._commands_run: list[str] = []
        self._health = PipelineHealth()

        # Layer 1: Working Memory — created now, discarded after run()
        if memory_coordinator is not None:
            self._wm = memory_coordinator.create_working_memory(
                agent_id=self._agent_id,
                task_description=subtask.description,
            )
        else:
            from ..memory.working_memory import WorkingMemory
            self._wm = WorkingMemory(
                task_description=subtask.description,
                agent_id=self._agent_id,
            )

        # Stuck-loop detection: rolling window of call fingerprints
        self._recent_calls: deque[str] = deque(maxlen=STUCK_LOOP_WINDOW)

        # Threading signal so a parent can poll this agent's state
        self._signal_lock = threading.Lock()
        self._subtask.signal = SubAgentSignal.PENDING

    # ─── Public ──────────────────────────────────────────────────────────────

    def run(self) -> SubTask:
        """
        Execute the ReAct loop.

        Conversation invariant is maintained here in run():
          1. We inject the initial task message (user) once.
          2. Each iteration: LLM call (always sees user-last history) →
             assistant response appended → tool executed → observation
             injected as user message → repeat.
          3. On done: assistant response is already appended; we just break.

        Returns the mutated SubTask with results.
        """
        self._set_signal(SubAgentSignal.RUNNING)
        self._subtask.status = SubTaskStatus.RUNNING
        self._subtask.agent_id = self._agent_id

        max_steps = min(self._subtask.max_steps, ABSOLUTE_MAX_STEPS)

        logger.info(
            "[%s] Starting (depth=%d): %s (max_steps=%d)",
            self._agent_id, self._depth,
            self._subtask.description[:70], max_steps,
        )

        # ── Inject initial task message (user) ───────────────────────────────
        # This is the only place we build a "user prompt" besides observations.
        # Subsequent steps consume the [TOOL_RESULT] observations as user msgs.
        context = self._observe()
        initial_msg = self._build_initial_message(context)
        self._step_history.append({"role": "user", "content": initial_msg})

        # ── ReAct loop ───────────────────────────────────────────────────────
        for step_num in range(1, max_steps + 1):

            # Health gate — enter sleep before any new LLM call
            should_sleep, reason = self._health.should_sleep()
            if should_sleep:
                self._enter_sleep(reason)
                break

            # ── THINK: call LLM (history ends with user message) ─────────────
            t0 = time.monotonic()
            system_prompt = self._build_system_prompt()
            llm_response = self._router.call(
                task_type=self._subtask.task_type,
                system_prompt=system_prompt,
                messages=self._step_history,
            )
            latency = (time.monotonic() - t0) * 1000

            # Handle LLM errors
            if llm_response.is_error:
                trace = self._handle_llm_error(step_num, llm_response, latency)
                self._traces.append(trace)
                self._subtask.traces.append(trace)
                continue

            # Append assistant response — history now ends with assistant msg
            self._step_history.append({
                "role": "assistant",
                "content": llm_response.content,
            })

            # ── Parse action ─────────────────────────────────────────────────
            action, parse_ok = self._parse_action(llm_response.content)
            if not parse_ok:
                self._health.parse_failures += 1
                self._health.record(HealthSignal(
                    agent_id=self._agent_id,
                    signal=SleepReason.LLM_PARSE_FAILURES,
                    detail=f"step {step_num}: {llm_response.content[:120]}",
                    severity=1,
                ))

            action_type = action.get("action", "done")

            # ── DONE? ─────────────────────────────────────────────────────────
            if action_type == "done":
                # History ends with assistant msg here — correct terminal state
                trace = StepTrace(
                    step_number=step_num,
                    agent_id=self._agent_id,
                    task_type=self._subtask.task_type,
                    model_used=llm_response.model_used,
                    thought=action.get("thought", ""),
                    tool_name=None,
                    tool_args={},
                    tool_result=action.get("final_answer", ""),
                    tool_result_raw=action.get("final_answer", ""),
                    tool_error=False,
                    latency_ms=latency,
                    health_flag=False,
                )
                self._traces.append(trace)
                self._subtask.traces.append(trace)

                self._subtask.status = SubTaskStatus.DONE
                self._subtask.result = action.get("final_answer") or self._best_effort_result()
                self._set_signal(SubAgentSignal.DONE)
                logger.info("[%s] Done at step %d", self._agent_id, step_num)
                break

            # ── ACT: execute tool ─────────────────────────────────────────────
            tool_name, tool_args, result_raw, tool_error = self._act(action)

            # Stuck-loop detection (only for real tools, not done/spawn)
            if action_type not in ("done", "spawn_subagent"):
                fp = _args_fingerprint(tool_name or action_type, tool_args)
                self._recent_calls.append(fp)
                if (
                    len(self._recent_calls) == STUCK_LOOP_WINDOW
                    and len(set(self._recent_calls)) == 1
                ):
                    logger.warning(
                        "[%s] ⚠ Stuck loop detected — same call %d times: %s",
                        self._agent_id, STUCK_LOOP_WINDOW, fp,
                    )
                    self._health.record(HealthSignal(
                        agent_id=self._agent_id,
                        signal=SleepReason.TOOL_LOOP_DETECTED,
                        detail=f"Repeated {STUCK_LOOP_WINDOW}x: {fp}",
                        severity=2,
                    ))
                    # Nudge the LLM to try something different
                    result_raw = (
                        f"{result_raw or ''}\n\n"
                        f"⚠ SYSTEM NOTICE: You have called '{tool_name}' with identical "
                        f"arguments {STUCK_LOOP_WINDOW} times in a row. "
                        f"This is a stuck loop. You MUST try a completely different approach."
                    )
                    self._recent_calls.clear()

            # Track consecutive errors / resets
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
                self._health.consecutive_tool_errors = 0

            # Side-effect tracking + Layer 1 Working Memory recording
            if tool_name in ("write_file", "edit_file"):
                path = tool_args.get("path", "")
                if path and path not in self._files_modified:
                    self._files_modified.append(path)
                if path and not tool_error:
                    content_written = tool_args.get("content", "")
                    self._wm.record_file_write(path, content_written, operation=tool_name)
            elif tool_name == "read_file":
                path = tool_args.get("path", "")
                if path and not tool_error:
                    self._wm.record_file_read(path, result_raw or "")
            if tool_name == "run_command":
                cmd = tool_args.get("command", "")
                if cmd:
                    self._commands_run.append(cmd)
                    self._wm.record_command(cmd, result_raw or "", ok=not tool_error)
            if tool_error:
                self._wm.record_error(step=step_num, tool=tool_name or "unknown", message=result_raw or "")
            elif step_num > 1 and self._wm.unresolved_errors:
                # If last step had an error and this one succeeded, mark it resolved
                self._wm.mark_error_resolved(step=step_num - 1, resolution=f"{tool_name} succeeded")

            # Build trace
            health_flag = (
                self._health.parse_failures > 0
                or self._health.consecutive_tool_errors >= 2
                or tool_error
            )
            trace = StepTrace(
                step_number=step_num,
                agent_id=self._agent_id,
                task_type=self._subtask.task_type,
                model_used=llm_response.model_used,
                thought=action.get("thought", ""),
                tool_name=tool_name,
                tool_args=tool_args,
                tool_result=(result_raw or "")[:500],
                tool_result_raw=result_raw,
                tool_error=tool_error,
                latency_ms=latency,
                health_flag=health_flag,
            )
            self._traces.append(trace)
            self._subtask.traces.append(trace)

            # ── OBSERVE: inject result as user message ────────────────────────
            # This restores the invariant: history ends with user message.
            # The LLM will see this as the "observation" for its next think step.
            observation = self._build_observation(
                step_num=step_num,
                max_steps=max_steps,
                action_type=action_type,
                tool_name=tool_name,
                result_raw=result_raw,
                tool_error=tool_error,
            )
            self._step_history.append({"role": "user", "content": observation})

        else:
            # Exhausted max_steps
            logger.warning("[%s] Hit max_steps=%d", self._agent_id, max_steps)
            self._subtask.status = SubTaskStatus.DONE
            self._subtask.result = self._best_effort_result()
            self._set_signal(SubAgentSignal.DONE)

        self._subtask.health = self._health

        # Layer 1 → Layer 2 harvest: attach files_written and summary to subtask
        self._subtask.files_written = self._wm.files_written  # type: ignore[attr-defined]
        self._subtask.working_memory_summary = self._wm.summary()  # type: ignore[attr-defined]
        if self._memory_coordinator is not None:
            self._memory_coordinator.on_task_complete(self._subtask, self._wm)
            self._memory_coordinator.release_working_memory(self._agent_id)

        return self._subtask

    # ─── Observe (vector context) ─────────────────────────────────────────────

    def _observe(self) -> str:
        """Retrieve relevant context from the vector store."""
        if not self._store or not self._pipeline:
            return ""

        query_parts = [self._subtask.description]
        if self._subtask.context_hint:
            query_parts.append(self._subtask.context_hint)
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

    # ─── Act ──────────────────────────────────────────────────────────────────

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
        return action_type, args, tool_result.content, tool_result.is_error

    def _spawn_and_await_child(self, args: dict) -> str:
        """
        Spawn a child SubAgent and WAIT for it to finish.

        The parent signals AWAITING so the Orchestrator knows it is not
        stalled but deliberately blocked on child output. The parent
        blocks synchronously (same thread) — no new thread — until the
        child returns. This guarantees the parent's next ReAct step
        always sees the child's complete output.
        """
        from .models import SubTask, TaskType

        description = args.get("description", "(no description)")
        logger.info(
            "[%s] Spawning child subagent: %s",
            self._agent_id, description[:60],
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

        # Signal: parent is now AWAITING a child — not stalled, not done
        self._set_signal(SubAgentSignal.AWAITING)
        logger.info(
            "[%s] → AWAITING child %s …",
            self._agent_id, child_agent._agent_id,
        )

        # ── Run child synchronously — parent blocks here ──────────────────
        completed_child_task = child_agent.run()

        child_signal = completed_child_task.signal
        logger.info(
            "[%s] ← Child %s finished: signal=%s",
            self._agent_id, child_agent._agent_id, child_signal.value,
        )

        # Resume running
        self._set_signal(SubAgentSignal.RUNNING)

        # Merge child traces and side effects
        self._traces.extend(completed_child_task.traces)
        self._files_modified.extend(child_agent._files_modified)
        self._commands_run.extend(child_agent._commands_run)

        # Propagate child health signals to parent
        for signal in completed_child_task.health.signals:
            self._health.signals.append(signal)

        if completed_child_task.health.sleep_mode:
            self._health.record(HealthSignal(
                agent_id=self._agent_id,
                signal=SleepReason.SUBAGENT_TIMEOUT,
                detail=f"child {child_agent._agent_id} entered sleep mode",
                severity=2,
            ))

        if child_signal in (SubAgentSignal.FAILED, SubAgentSignal.SLEEPING):
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

    # ─── Observation builder ──────────────────────────────────────────────────

    def _build_observation(
        self,
        step_num: int,
        max_steps: int,
        action_type: str,
        tool_name: Optional[str],
        result_raw: Optional[str],
        tool_error: bool,
    ) -> str:
        """
        Build the user-role observation message that follows a tool call.

        This message restores the conversation invariant (ends with user msg)
        and gives the LLM everything it needs for the next think step.
        """
        steps_left = max_steps - step_num

        if action_type == "spawn_subagent":
            header = f"[CHILD_AGENT_RESULT | step {step_num}/{max_steps} | {steps_left} steps left]"
            body = result_raw or "(no output from child agent)"
        else:
            status = "❌ ERROR" if tool_error else "✅ OK"
            header = (
                f"[TOOL_RESULT: {tool_name} | {status} | "
                f"step {step_num}/{max_steps} | {steps_left} steps left]"
            )
            body = result_raw or "(no output)"

        # Optionally add a fresh context snippet for long tasks
        # (only on even steps to avoid noise)
        context_note = ""
        if step_num % 4 == 0 and steps_left > 1:
            fresh_ctx = self._observe()
            if fresh_ctx:
                context_note = f"\n\n--- Refreshed Codebase Context ---\n{fresh_ctx}\n---"

        footer = "\nContinue: analyse the result above and decide your next action."
        if steps_left <= 2:
            footer = f"\n⚠ Only {steps_left} step(s) remaining — work towards action=done soon."

        return f"{header}\n\n{body}{context_note}{footer}"

    # ─── Sleep mode ───────────────────────────────────────────────────────────

    def _enter_sleep(self, reason: SleepReason) -> None:
        """Enter sleep mode — record state, set signals, log clearly."""
        self._health.sleep_mode = True
        self._health.sleep_reason = reason
        self._set_signal(SubAgentSignal.SLEEPING)
        self._subtask.status = SubTaskStatus.FAILED
        self._subtask.error = f"Sleep mode: {reason.value}"

        logger.warning(
            "[%s] ⚠ ENTERING SLEEP MODE — reason=%s | signals=%d",
            self._agent_id, reason.value, len(self._health.signals),
        )
        logger.warning(
            "[%s] Health snapshot: parse_failures=%d, consecutive_errors=%d, llm_errors=%d",
            self._agent_id,
            self._health.parse_failures,
            self._health.consecutive_tool_errors,
            self._health.total_llm_errors,
        )

    # ─── LLM error handling ───────────────────────────────────────────────────

    def _handle_llm_error(
        self,
        step_num: int,
        llm_response,
        latency: float,
    ) -> StepTrace:
        """Handle a failed LLM call: update health, inject recovery observation."""
        self._health.total_llm_errors += 1
        self._health.record(HealthSignal(
            agent_id=self._agent_id,
            signal=SleepReason.LLM_ERROR_RATE,
            detail=f"step {step_num}: {llm_response.content[:120]}",
            severity=2,
        ))

        # Inject recovery message so history stays coherent (ends with user msg)
        # The existing last message was already user (initial or previous obs),
        # but if we somehow got an assistant turn in, add a nudge.
        if self._step_history and self._step_history[-1]["role"] == "assistant":
            self._step_history.append({
                "role": "user",
                "content": (
                    f"[SYSTEM: LLM error at step {step_num}. "
                    f"Error: {llm_response.content[:200]}. "
                    f"Please continue with your task.]"
                ),
            })

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

    # ─── Prompt building ──────────────────────────────────────────────────────

    def _build_system_prompt(self) -> str:
        detailed_schemas = _format_tool_schemas(self._tools.schemas())
        return SUBAGENT_SYSTEM.format(
            agent_id=self._agent_id,
            task_description=self._subtask.description,
            task_type=self._subtask.task_type.value,
            tool_schemas_detailed=detailed_schemas,
        )

    def _build_initial_message(self, context: str) -> str:
        """
        The ONLY user-authored message in the conversation.
        All subsequent user messages are [TOOL_RESULT] observations.

        Layer 1 injection: working memory context block is prepended here
        so the LLM always knows what it has already done in this task.
        """
        parts = [f"Task: {self._subtask.description}"]

        if self._subtask.context_hint:
            parts.append(f"Context hint: {self._subtask.context_hint}")

        # Layer 1: inject working memory (files already read/written, prior errors)
        wm_block = self._wm.to_context_block()
        if wm_block:
            parts.append(wm_block)

        if context:
            parts.append(f"\n--- Relevant Codebase Context (Layer 3: Semantic) ---\n{context}\n---")

        parts.append(
            "\nBegin. Think step-by-step, then choose your first tool call."
        )
        return "\n".join(parts)

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _parse_action(self, raw: str) -> tuple[dict, bool]:
        """Parse LLM JSON. Returns (action_dict, parse_ok)."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            return json.loads(cleaned), True
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group()), True
                except json.JSONDecodeError:
                    pass

        logger.warning(
            "[%s] ⚠ JSON parse failed: %s…", self._agent_id, raw[:120]
        )
        return {
            "thought": "Could not parse structured response from LLM",
            "action": "done",
            "args": {},
            "final_answer": raw,
        }, False

    def _best_effort_result(self) -> str:
        for trace in reversed(self._traces):
            if trace.tool_result and not trace.tool_error:
                return f"[Partial result from step {trace.step_number}]\n{trace.tool_result}"
        return f"[Incomplete after {len(self._traces)} steps]"

    def _set_signal(self, signal: SubAgentSignal) -> None:
        with self._signal_lock:
            self._subtask.signal = signal

    # ─── Properties ───────────────────────────────────────────────────────────

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