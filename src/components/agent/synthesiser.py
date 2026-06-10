"""
Synthesiser — merges all SubAgent results into one coherent response.

WHY a dedicated synthesis step:
  When N SubAgents work in parallel, each produces a partial result scoped to
  its subtask. The user asked ONE question. The Synthesiser's job is to:
  1. Detect conflicts (two agents edited the same file differently)
  2. Merge results into a single coherent narrative
  3. Surface what actually changed (files written, tests run, etc.)
  4. For DIRECT mode, just clean up and return the single agent's result

DESIGN NOTE on conflict detection:
  We detect conflicts by scanning traces for multiple write_file/edit_file
  calls on the same path from different agents. When found, we note the
  conflict in the response — we do NOT silently pick one. The user should
  see that two agents touched the same file.

DESIGN NOTE on synthesis LLM call:
  For PARALLEL mode with 3+ subtasks, we call the LLM to produce a coherent
  merged response. For DIRECT mode or simple 1-2 task results, we return
  the result directly without an extra LLM call — the overhead isn't worth it.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .models import (
    AgentMode, AgentResult, SubTaskStatus, TaskPlan, StepTrace, TaskType
)
from .router import ModelRouter

logger = logging.getLogger(__name__)

SYNTHESISER_SYSTEM = """You are the Synthesiser for Sovereign-Code.

Multiple specialised subagents have completed work on different parts of a task.
Your job is to merge their results into ONE clear, coherent response for the user.

Rules:
- Write in first person as the agent ("I analysed...", "I updated...")
- Mention specific files changed and what changed in them
- If tests were run, include their outcome
- If there were errors or incomplete parts, be honest about them
- Keep it concise — no unnecessary repetition of each subagent's output
- Format code blocks properly with language tags
- End with a clear summary of what was accomplished
"""


class Synthesiser:
    """Merges TaskPlan results into a final AgentResult."""

    def __init__(self, router: ModelRouter):
        self._router = router

    def synthesise(
        self,
        plan: TaskPlan,
        all_traces: list[StepTrace],
        conversation_history: list[dict],
    ) -> AgentResult:
        """Build the final AgentResult from completed TaskPlan.

        Args:
            plan:                   Completed TaskPlan (subtasks have results set).
            all_traces:             Every StepTrace from every agent.
            conversation_history:   Original session context.

        Returns:
            AgentResult ready for the CLI to render.
        """
        completed = [t for t in plan.subtasks if t.status == SubTaskStatus.DONE]
        failed = [t for t in plan.subtasks if t.status == SubTaskStatus.FAILED]
        skipped = [t for t in plan.subtasks if t.status == SubTaskStatus.SKIPPED]

        # Detect file conflicts
        conflicts = self._detect_conflicts(all_traces)

        # Choose synthesis strategy
        if plan.mode == AgentMode.DIRECT or len(completed) <= 1:
            response = self._direct_response(plan, completed, failed, conflicts)
        else:
            response = self._llm_merge(plan, completed, failed, skipped, conflicts, conversation_history)

        # Append conflict warnings if any
        if conflicts:
            response += self._format_conflicts(conflicts)

        # Append failure summary if any
        if failed:
            response += self._format_failures(failed)

        return AgentResult(
            response=response,
            mode=plan.mode,
            plan=plan,
            all_traces=all_traces,
            files_modified=[],   # filled by Orchestrator
            commands_run=[],     # filled by Orchestrator
            success=len(failed) == 0,
            error=failed[0].error if failed and not completed else None,
        )


    # Direct (single task) response

    def _direct_response(
        self,
        plan: TaskPlan,
        completed: list,
        failed: list,
        conflicts: dict,
    ) -> str:
        if completed:
            result = completed[0].result or "(task completed)"
            return result
        if failed:
            return f"Task failed: {failed[0].error or 'unknown error'}"
        return "No tasks were completed."


    # LLM merge (parallel mode, 2+ subtasks)

    def _llm_merge(
        self,
        plan: TaskPlan,
        completed: list,
        failed: list,
        skipped: list,
        conflicts: dict,
        conversation_history: list[dict],
    ) -> str:
        # Build a summary of each subtask's result
        results_text = f"Original task: {plan.original_query}\n\n"
        results_text += "## Subtask Results\n\n"

        for task in plan.subtasks:
            status_icon = {"done": "✓", "failed": "✗", "skipped": "⊘"}.get(
                task.status.value, "?"
            )
            results_text += f"### {status_icon} {task.id}: {task.description}\n"
            if task.result:
                results_text += f"{task.result}\n\n"
            elif task.error:
                results_text += f"Error: {task.error}\n\n"
            else:
                results_text += "(no output)\n\n"

        messages = conversation_history + [
            {"role": "user", "content": results_text}
        ]

        response = self._router.call(
            task_type=TaskType.SYNTHESISE,
            system_prompt=SYNTHESISER_SYSTEM,
            messages=messages,
        )

        if response.is_error:
            # Fall back to concatenated results
            logger.warning("Synthesiser LLM call failed, concatenating results")
            return self._concat_results(plan)

        return response.content

    def _concat_results(self, plan: TaskPlan) -> str:
        """Fallback: just concatenate all subtask results."""
        parts = []
        for task in plan.subtasks:
            if task.result:
                parts.append(f"**{task.description}**\n{task.result}")
        return "\n\n---\n\n".join(parts) if parts else "Tasks completed."


    # Conflict detection

    def _detect_conflicts(self, traces: list[StepTrace]) -> dict[str, list[str]]:
        """Find files written by more than one agent.

        Returns {file_path: [agent_id, agent_id, ...]}
        """
        file_writers: dict[str, list[str]] = defaultdict(list)
        for trace in traces:
            if trace.tool_name in ("write_file", "edit_file"):
                path = trace.tool_args.get("path", "")
                if path and trace.agent_id not in file_writers[path]:
                    file_writers[path].append(trace.agent_id)

        return {
            path: agents
            for path, agents in file_writers.items()
            if len(agents) > 1
        }

    def _format_conflicts(self, conflicts: dict[str, list[str]]) -> str:
        lines = ["\n\n⚠️  **File conflicts detected:**"]
        for path, agents in conflicts.items():
            lines.append(f"- `{path}` was modified by: {', '.join(agents)}")
        lines.append("The last write wins. Review these files manually.")
        return "\n".join(lines)

    def _format_failures(self, failed: list) -> str:
        lines = ["\n\n❌ **Failed subtasks:**"]
        for task in failed:
            lines.append(f"- {task.description}: {task.error or 'unknown error'}")
        return "\n".join(lines)