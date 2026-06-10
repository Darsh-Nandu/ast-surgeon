"""
Planner — converts a user query into a TaskPlan DAG.

WHY the Planner is a separate LLM call (not baked into the agent loop):
  The Planner has one job: think about the STRUCTURE of the work, not do the work.
  It uses Gemini Flash (best at structured reasoning) and is prompted to return
  strict JSON. The rest of the system never sees the planning prompt — clean separation.

WHY the Planner decides mode (DIRECT vs PARALLEL):
  Only the Planner has the full context of the query AND the codebase summary
  to decide if parallelism is worth the overhead. Spawning subagents has a cost
  (setup time, context duplication, synthesis overhead). The Planner should only
  choose PARALLEL when tasks are genuinely independent and each complex enough
  to warrant its own loop.

PARALLEL when:
  - 2+ clearly independent subtasks (different files/modules/concerns)
  - Each subtask would take 3+ steps
  - Subtasks don't need each other's intermediate output

DIRECT when:
  - Simple Q&A / explanation
  - Single-file edit
  - Short command run
  - Anything sequential by nature

DESIGN NOTE on JSON parsing robustness:
  LLMs sometimes wrap JSON in markdown fences (```json ... ```).
  We strip those before parsing. We also validate the schema and fall back
  to a single DIRECT task if the JSON is malformed — the agent always makes
  progress rather than crashing on a bad plan.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from .models import AgentMode, SubTask, TaskPlan, TaskType
from .router import ModelRouter

logger = logging.getLogger(__name__)



# System prompt for the Planner

PLANNER_SYSTEM = """You are the Planner for Sovereign-Code, a production coding agent.
Your job is to analyse a user's coding task and produce a structured execution plan.
You must respond with ONLY a valid JSON object - no markdown, no explanation, just JSON.

Schema:
{
  "mode": "direct" | "parallel",
  "reasoning": "one sentence explaining your choice",
  "subtasks": [
    {
      "id": "task_0",
      "description": "what this subtask should accomplish",
      "task_type": "planning|code_gen|code_edit|code_review|debug|test_write|search|run|explain|synthesise",
      "dependencies": [],
      "spawn_subagent": false,
      "context_hint": "keywords to use for codebase retrieval",
      "max_steps": 5
    }
  ]
}

Rules:
- Use mode=direct for simple tasks (Q&A, single file edit, short explanation)
- Use mode=parallel ONLY when there are 2+ truly independent subtasks each needing 3+ steps
- spawn_subagent=true means this subtask gets its own isolated agent loop with tools
- spawn_subagent=false means the orchestrator handles it inline (for simple/quick subtasks)
- dependencies must reference valid subtask IDs that appear earlier in the list
- max_steps: 3-5 for simple subtasks, 8-10 for complex ones
- context_hint: specific function names, class names, or file paths relevant to this subtask
"""


# Planner

class Planner:
    """Converts a user query into a TaskPlan DAG.

    Usage:
        planner = Planner(router)
        plan = planner.plan(
            query="Refactor AuthService and add unit tests",
            codebase_summary="AuthService in src/auth.py, tests in tests/",
            conversation_history=[...],
        )
    """

    def __init__(self, router: ModelRouter):
        self._router = router

    def plan(
        self,
        query: str,
        codebase_summary: str = "",
        conversation_history: Optional[list[dict]] = None,
    ) -> TaskPlan:
        """Produce a TaskPlan for the given query.

        Args:
            query:                User's task/question.
            codebase_summary:     Brief summary of relevant codebase context
                                  (from vector search, injected by Orchestrator).
            conversation_history: Prior turns for multi-turn context.

        Returns:
            TaskPlan — always returns something valid, even on LLM failure.
        """
        user_content = self._build_user_prompt(query, codebase_summary)
        messages = (conversation_history or []) + [
            {"role": "user", "content": user_content}
        ]

        response = self._router.call(
            task_type=TaskType.PLANNING,
            system_prompt=PLANNER_SYSTEM,
            messages=messages,
        )

        if response.is_error:
            logger.warning("Planner LLM call failed, falling back to direct plan")
            return self._fallback_plan(query)

        plan = self._parse_plan(response.content, query)
        logger.info(
            "Plan: %s — %d subtasks (model: %s, %.0fms)",
            plan.mode.value,
            len(plan.subtasks),
            response.model_used,
            response.latency_ms,
        )
        return plan


    # Internal

    def _build_user_prompt(self, query: str, codebase_summary: str) -> str:
        prompt = f"User task: {query}"
        if codebase_summary:
            prompt += f"\n\nRelevant codebase context:\n{codebase_summary}"
        return prompt

    def _parse_plan(self, raw: str, original_query: str) -> TaskPlan:
        """Parse LLM output into a TaskPlan. Falls back gracefully on bad JSON."""
        # Strip markdown fences if present
        cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", raw).strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.warning("Planner JSON parse failed: %s — using fallback", exc)
            return self._fallback_plan(original_query)

        try:
            mode = AgentMode(data.get("mode", "direct"))
            reasoning = data.get("reasoning", "")
            subtasks_raw = data.get("subtasks", [])

            if not subtasks_raw:
                return self._fallback_plan(original_query)

            subtasks = []
            seen_ids = set()
            for raw_task in subtasks_raw:
                task_id = raw_task.get("id", f"task_{len(subtasks)}")

                # Validate dependencies reference known IDs
                deps = [
                    d for d in raw_task.get("dependencies", [])
                    if d in seen_ids
                ]

                try:
                    task_type = TaskType(raw_task.get("task_type", "code_gen"))
                except ValueError:
                    task_type = TaskType.CODE_GEN

                subtasks.append(SubTask(
                    id=task_id,
                    description=raw_task.get("description", ""),
                    task_type=task_type,
                    dependencies=deps,
                    spawn_subagent=bool(raw_task.get("spawn_subagent", False)),
                    context_hint=raw_task.get("context_hint", ""),
                    max_steps=int(raw_task.get("max_steps", 8)),
                ))
                seen_ids.add(task_id)

            # Sanity check: only allow PARALLEL if there are actually independent tasks
            independent = [t for t in subtasks if not t.dependencies]
            if mode == AgentMode.PARALLEL and len(independent) < 2:
                mode = AgentMode.DIRECT
                reasoning += " (downgraded to direct: insufficient parallelism)"

            return TaskPlan(
                mode=mode,
                subtasks=subtasks,
                original_query=original_query,
                reasoning=reasoning,
            )

        except Exception as exc:
            logger.warning("Planner plan construction failed: %s", exc)
            return self._fallback_plan(original_query)

    def _fallback_plan(self, query: str) -> TaskPlan:
        """Single direct task — always valid, never crashes."""
        return TaskPlan(
            mode=AgentMode.DIRECT,
            subtasks=[SubTask(
                id="task_0",
                description=query,
                task_type=TaskType.CODE_GEN,
                dependencies=[],
                spawn_subagent=False,
                context_hint=query[:100],
                max_steps=10,
            )],
            original_query=query,
            reasoning="fallback: single direct task due to planning error",
        )