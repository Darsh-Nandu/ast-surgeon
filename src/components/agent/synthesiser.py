"""
Synthesiser — merges all SubAgent results into one coherent response.

CHANGES:
  - Accepts aggregate_health from Orchestrator
  - Appends sleep-mode notice if pipeline slept
  - No LLM call for direct/single-task mode (unchanged, still correct)
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Optional

from .models import (
    AgentMode, AgentResult, PipelineHealth, SubTaskStatus,
    TaskPlan, StepTrace, TaskType,
)
from .router import ModelRouter

logger = logging.getLogger(__name__)

SYNTHESISER_SYSTEM = """\
You are the Synthesiser for Sovereign-Code.

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
    def __init__(self, router: ModelRouter):
        self._router = router

    def synthesise(
        self,
        plan: TaskPlan,
        all_traces: list[StepTrace],
        conversation_history: list[dict],
        aggregate_health: Optional[PipelineHealth] = None,
    ) -> AgentResult:
        completed = [t for t in plan.subtasks if t.status == SubTaskStatus.DONE]
        failed    = [t for t in plan.subtasks if t.status == SubTaskStatus.FAILED]
        skipped   = [t for t in plan.subtasks if t.status == SubTaskStatus.SKIPPED]

        conflicts = self._detect_conflicts(all_traces)

        if plan.mode == AgentMode.DIRECT or len(completed) <= 1:
            response = self._direct_response(plan, completed, failed, conflicts)
        else:
            response = self._llm_merge(
                plan, completed, failed, skipped, conflicts, conversation_history
            )

        if conflicts:
            response += self._format_conflicts(conflicts)
        if failed:
            response += self._format_failures(failed)
        if aggregate_health and aggregate_health.sleep_mode:
            response += self._format_sleep_notice(aggregate_health)

        return AgentResult(
            response=response,
            mode=plan.mode,
            plan=plan,
            all_traces=all_traces,
            files_modified=[],
            commands_run=[],
            success=len(failed) == 0,
            error=failed[0].error if failed and not completed else None,
        )

    # ─── Direct ──────────────────────────────────────────────────────────────

    def _direct_response(self, plan, completed, failed, conflicts) -> str:
        if completed:
            return completed[0].result or "(task completed)"
        if failed:
            return f"Task failed: {failed[0].error or 'unknown error'}"
        return "No tasks were completed."

    # ─── LLM merge ───────────────────────────────────────────────────────────

    def _llm_merge(self, plan, completed, failed, skipped, conflicts, history) -> str:
        results_text = f"Original task: {plan.original_query}\n\n## Subtask Results\n\n"
        for task in plan.subtasks:
            icon = {"done": "✓", "failed": "✗", "skipped": "⊘"}.get(task.status.value, "?")
            results_text += f"### {icon} {task.id}: {task.description}\n"
            if task.result:
                results_text += f"{task.result}\n\n"
            elif task.error:
                results_text += f"Error: {task.error}\n\n"
            else:
                results_text += "(no output)\n\n"

        messages = history + [{"role": "user", "content": results_text}]
        response = self._router.call(
            task_type=TaskType.SYNTHESISE,
            system_prompt=SYNTHESISER_SYSTEM,
            messages=messages,
        )
        if response.is_error:
            logger.warning("Synthesiser LLM call failed, concatenating results")
            return self._concat_results(plan)
        return response.content

    def _concat_results(self, plan) -> str:
        parts = []
        for task in plan.subtasks:
            if task.result:
                parts.append(f"**{task.description}**\n{task.result}")
        return "\n\n---\n\n".join(parts) if parts else "Tasks completed."

    # ─── Formatting helpers ───────────────────────────────────────────────────

    def _detect_conflicts(self, traces: list[StepTrace]) -> dict[str, list[str]]:
        fw: dict[str, list[str]] = defaultdict(list)
        for trace in traces:
            if trace.tool_name in ("write_file", "edit_file"):
                path = trace.tool_args.get("path", "")
                if path and trace.agent_id not in fw[path]:
                    fw[path].append(trace.agent_id)
        return {p: a for p, a in fw.items() if len(a) > 1}

    def _format_conflicts(self, conflicts: dict[str, list[str]]) -> str:
        lines = ["\n\n⚠️  **File conflicts detected:**"]
        for path, agents in conflicts.items():
            lines.append(f"- `{path}` modified by: {', '.join(agents)}")
        lines.append("Last write wins. Review these files manually.")
        return "\n".join(lines)

    def _format_failures(self, failed: list) -> str:
        lines = ["\n\n❌ **Failed subtasks:**"]
        for task in failed:
            lines.append(f"- {task.description}: {task.error or 'unknown error'}")
        return "\n".join(lines)

    def _format_sleep_notice(self, health: PipelineHealth) -> str:
        lines = [
            "\n\n⚠️  **Pipeline entered SLEEP MODE**",
            f"Reason: `{health.sleep_reason.value if health.sleep_reason else 'unknown'}`",
            f"Health signals: {len(health.signals)}",
        ]
        if health.signals:
            for sig in health.signals[-3:]:  # show last 3
                lines.append(f"  - [{sig.severity}] {sig.signal.value}: {sig.detail[:80]}")
        lines.append(
            "\nA RepairAgent can be invoked to diagnose and fix the pipeline. "
            "Run with `--repair` flag when available."
        )
        return "\n".join(lines)