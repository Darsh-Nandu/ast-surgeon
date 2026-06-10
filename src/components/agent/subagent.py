"""
SubAgent — an isolated Observe → Think → Act loop for one subtask.

WHY each SubAgent is fully isolated:
  Each SubAgent gets its own ToolRegistry, its own step trace, and its own
  context window. This means:
  - Two SubAgents editing different files never interfere
  - A failing SubAgent doesn't crash the Orchestrator
  - The full trace of every SubAgent is available for replay and evals

The loop per step:
  1. OBSERVE  — retrieve relevant chunks from Qdrant for the current task state
  2. THINK    — call the routed LLM with full context: system prompt + history
                + retrieved chunks + available tools. LLM returns a JSON action.
  3. ACT      — execute the chosen tool, record result in step trace
  4. CHECK    — if LLM said "done" or max_steps hit, exit loop

DESIGN NOTE on action format:
  We ask the LLM to respond in strict JSON:
  {
    "thought": "reasoning about what to do next",
    "action": "tool_name" | "done" | "spawn_subagent",
    "args": { ...tool args... },
    "final_answer": "only set when action=done"
  }

  "spawn_subagent" is a special action — the SubAgent can request a child
  SubAgent for a sub-problem it discovers mid-task. The child runs inline
  (not in a new thread) and its result comes back as a tool result.

DESIGN NOTE on context budget:
  We retrieve top-5 chunks per step (not top-20). Each step has fresh context
  based on the CURRENT state of the task — earlier steps may have changed files
  that are now more relevant. Fresh retrieval per step beats a stale large context.
"""

from __future__ import annotations

import json
import logging
import re
import time
import uuid
from typing import Optional, TYPE_CHECKING

from .models import (
    AgentMode, SubTask, SubTaskStatus, StepTrace, TaskType, AgentResult
)
from .router import ModelRouter

if TYPE_CHECKING:
    from ..tools.file_tools import ToolRegistry
    from ..vectorstore.qdrant_store import VectorStore
    from ..embeddings.pipeline import EmbeddingPipeline

logger = logging.getLogger(__name__)

# Hard ceiling — no matter what, a SubAgent exits after this many steps
ABSOLUTE_MAX_STEPS = 15


# System prompt template

SUBAGENT_SYSTEM = """You are a SubAgent of Sovereign-Code, an expert coding assistant.

Your assigned task: {task_description}
Task type: {task_type}
Agent ID: {agent_id}

You have access to these tools:
{tool_schemas}

You must respond with ONLY a valid JSON object — no markdown, no explanation:

{{
  "thought": "your reasoning about what to do next (be specific)",
  "action": "<tool_name> | done | spawn_subagent",
  "args": {{ ...tool arguments matching the tool schema... }},
  "final_answer": "your complete answer/result (only when action=done)"
}}

When action=spawn_subagent, args must be:
{{
  "description": "what the sub-task should do",
  "task_type": "code_gen|code_edit|debug|test_write|search|explain",
  "context_hint": "keywords for retrieval"
}}

Rules:
- Think step by step. Read files before editing them.
- Use search_files to find relevant code before assuming structure.
- Use edit_file for targeted changes, write_file only for new files.
- Use run_command to verify changes (run tests, lint, etc).
- Spawn a subagent only for a genuinely isolated sub-problem you discover.
- Set action=done when the task is complete. Always set final_answer when done.
- If stuck after 3 steps on the same problem, try a different approach.
"""


# SubAgent

class SubAgent:
    """Runs an isolated Observe→Think→Act loop for one SubTask.

    Can be run directly by the Orchestrator or spawned as a child agent
    from within another SubAgent's loop.
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
    ):
        self._subtask = subtask
        self._router = router
        self._tools = tool_registry
        self._store = vector_store
        self._pipeline = embed_pipeline
        self._agent_id = agent_id or f"agent-{uuid.uuid4().hex[:6]}"
        self._conversation_history = conversation_history or []

        # Per-agent state
        self._step_history: list[dict] = []   # LLM message history for this agent
        self._traces: list[StepTrace] = []
        self._files_modified: list[str] = []
        self._commands_run: list[str] = []


    # Public

    def run(self) -> SubTask:
        """Execute the subtask loop. Returns the mutated SubTask with results."""
        self._subtask.status = SubTaskStatus.RUNNING
        self._subtask.agent_id = self._agent_id

        logger.info(
            "[%s] Starting: %s (max_steps=%d)",
            self._agent_id, self._subtask.description[:60], self._subtask.max_steps
        )

        max_steps = min(self._subtask.max_steps, ABSOLUTE_MAX_STEPS)

        for step_num in range(1, max_steps + 1):
            trace = self._step(step_num)
            self._traces.append(trace)
            self._subtask.traces.append(trace)

            # Check if agent said done
            if self._is_done(trace):
                self._subtask.status = SubTaskStatus.DONE
                self._subtask.result = self._extract_final_answer(trace)
                logger.info("[%s] Done at step %d", self._agent_id, step_num)
                break

            # Check for unrecoverable error
            if self._is_stuck(step_num):
                logger.warning("[%s] Stuck at step %d, forcing completion", self._agent_id, step_num)
                self._subtask.status = SubTaskStatus.DONE
                self._subtask.result = self._best_effort_result()
                break

        else:
            # Hit max_steps
            logger.warning("[%s] Hit max_steps=%d", self._agent_id, max_steps)
            self._subtask.status = SubTaskStatus.DONE
            self._subtask.result = self._best_effort_result()

        return self._subtask


    # Core loop step

    def _step(self, step_num: int) -> StepTrace:
        """One full Observe → Think → Act cycle."""
        t0 = time.monotonic()

        # 1. OBSERVE — retrieve fresh context for current task state
        context = self._observe()

        # 2. THINK — build prompt and call LLM
        system_prompt = self._build_system_prompt()
        user_content = self._build_step_prompt(step_num, context)
        self._step_history.append({"role": "user", "content": user_content})

        llm_response = self._router.call(
            task_type=self._subtask.task_type,
            system_prompt=system_prompt,
            messages=self._step_history,
        )

        # Add assistant response to history
        self._step_history.append({
            "role": "assistant",
            "content": llm_response.content
        })

        # 3. ACT - parse action and execute
        action = self._parse_action(llm_response.content)
        tool_name, tool_args, tool_result_str, tool_error = self._act(action)

        # Inject tool result back into history so LLM sees it next step
        if tool_result_str and action.get("action") not in ("done", "spawn_subagent"):
            self._step_history.append({
                "role": "user",
                "content": f"Tool result:\n{tool_result_str}"
            })

        latency = (time.monotonic() - t0) * 1000

        return StepTrace(
            step_number=step_num,
            agent_id=self._agent_id,
            task_type=self._subtask.task_type,
            model_used=llm_response.model_used,
            thought=action.get("thought", ""),
            tool_name=tool_name,
            tool_args=tool_args,
            tool_result=tool_result_str[:500] if tool_result_str else None,
            tool_error=tool_error,
            latency_ms=latency,
        )


    # Observe

    def _observe(self) -> str:
        """Retrieve relevant context from the vector store for the current task."""
        if not self._store or not self._pipeline:
            return ""

        # Build a query from task description + recent step history
        query_parts = [self._subtask.description]
        if self._subtask.context_hint:
            query_parts.append(self._subtask.context_hint)
        # Add last tool result as context hint (what we just found/changed)
        if self._step_history:
            last = self._step_history[-1].get("content", "")[:100]
            query_parts.append(last)

        query = " ".join(query_parts)

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
                    f"```{c.language}\n{c.content[:800]}\n```"
                )
            return "\n\n".join(chunks)
        except Exception as exc:
            logger.debug("Observe failed: %s", exc)
            return ""


    # Act

    def _act(self, action: dict) -> tuple[Optional[str], dict, Optional[str], bool]:
        """Execute the action chosen by the LLM.

        Returns (tool_name, tool_args, result_str, is_error).
        """
        action_type = action.get("action", "done")
        args = action.get("args", {})

        if action_type == "done":
            return None, {}, None, False

        if action_type == "spawn_subagent":
            result_str = self._spawn_child_subagent(args)
            return "spawn_subagent", args, result_str, False

        # Regular tool call
        if action_type not in self._tools.available:
            error_msg = f"Unknown tool: {action_type!r}. Available: {self._tools.available}"
            return action_type, args, error_msg, True

        tool_result = self._tools.run(action_type, **args)

        # Track side effects
        if action_type in ("write_file", "edit_file"):
            path = args.get("path", "")
            if path and path not in self._files_modified:
                self._files_modified.append(path)
        if action_type == "run_command":
            cmd = args.get("command", "")
            if cmd:
                self._commands_run.append(cmd)

        return action_type, args, tool_result.content, tool_result.is_error

    def _spawn_child_subagent(self, args: dict) -> str:
        """Spawn a child SubAgent inline for a discovered sub-problem.

        DESIGN NOTE: child runs synchronously in the same thread. This is
        intentional — the parent needs the result before continuing. True
        parallel spawning is the Orchestrator's job for top-level tasks.
        """
        from .models import SubTask, TaskType

        logger.info("[%s] Spawning child subagent: %s", self._agent_id, args.get("description", "")[:50])

        try:
            task_type = TaskType(args.get("task_type", "code_gen"))
        except ValueError:
            task_type = TaskType.CODE_GEN

        child_task = SubTask(
            id=f"{self._subtask.id}_child_{len(self._traces)}",
            description=args.get("description", ""),
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
            agent_id=f"{self._agent_id}-child{len(self._traces)}",
        )

        child_task = child_agent.run()

        # Merge child traces into our trace (for full audit trail)
        self._traces.extend(child_task.traces)
        self._files_modified.extend(child_agent._files_modified)
        self._commands_run.extend(child_agent._commands_run)

        return child_task.result or "(child subagent completed with no output)"


    # Prompt building

    def _build_system_prompt(self) -> str:
        tool_schemas = "\n".join(
            f"- {s['name']}: {s['description']}"
            for s in self._tools.schemas()
        )
        return SUBAGENT_SYSTEM.format(
            task_description=self._subtask.description,
            task_type=self._subtask.task_type.value,
            agent_id=self._agent_id,
            tool_schemas=tool_schemas,
        )

    def _build_step_prompt(self, step_num: int, context: str) -> str:
        parts = [f"Step {step_num}/{self._subtask.max_steps}"]
        if context:
            parts.append(f"\nRelevant codebase context:\n{context}")
        if step_num == 1:
            parts.append(f"\nBegin working on: {self._subtask.description}")
        else:
            parts.append("\nContinue. What is your next action?")
        return "\n".join(parts)


    # Helpers

    def _parse_action(self, raw: str) -> dict:
        """Parse LLM JSON response into an action dict. Robust to markdown wrapping."""
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to extract JSON object from anywhere in the string
            match = re.search(r'\{.*\}', cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            logger.warning("[%s] Failed to parse action JSON: %s...", self._agent_id, raw[:100])
            # Treat entire response as a final answer
            return {
                "thought": "Could not parse structured response",
                "action": "done",
                "args": {},
                "final_answer": raw,
            }

    def _is_done(self, trace: StepTrace) -> bool:
        return trace.tool_name is None and not trace.tool_error

    def _is_stuck(self, step_num: int) -> bool:
        """Detect if we've been calling the same tool with the same args repeatedly."""
        if step_num < 4:
            return False
        recent = self._traces[-3:]
        if all(t.tool_name == recent[0].tool_name for t in recent):
            if all(t.tool_args == recent[0].tool_args for t in recent):
                return True
        return False

    def _extract_final_answer(self, trace: StepTrace) -> str:
        """Extract the final_answer from the last assistant message."""
        for msg in reversed(self._step_history):
            if msg["role"] == "assistant":
                action = self._parse_action(msg["content"])
                if fa := action.get("final_answer"):
                    return fa
        return self._best_effort_result()

    def _best_effort_result(self) -> str:
        """Return the most useful result we have, even if incomplete."""
        # Look for last non-error tool result in traces
        for trace in reversed(self._traces):
            if trace.tool_result and not trace.tool_error:
                return f"[Partial result]\n{trace.tool_result}"
        return f"[Task incomplete after {len(self._traces)} steps]"

    @property
    def traces(self) -> list[StepTrace]:
        return list(self._traces)

    @property
    def files_modified(self) -> list[str]:
        return list(self._files_modified)

    @property
    def commands_run(self) -> list[str]:
        return list(self._commands_run)