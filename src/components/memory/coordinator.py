"""
MemoryCoordinator — wires all three memory layers into one API.

This is the single import every other module needs. It provides:
  - get_working_memory(agent_id)   → Layer 1 (creates one per agent)
  - episodic                       → Layer 2 (one per session)
  - (Layer 3 = Qdrant, already wired in AgentLoop)

USAGE PATTERN:

    # In AgentLoop.__init__:
    self._memory = MemoryCoordinator(session_dir=session.session_dir,
                                     project_root=project_root)

    # Before planning each turn:
    ep_context = self._memory.episodic.to_planner_context()

    # In SubAgent.run():
    wm = self._memory.working_memory   # pre-created, injected into SubAgent
    initial_msg = wm.to_context_block() + task_prompt

    # After SubAgent finishes:
    self._memory.on_task_complete(subtask, agent_result)

    # After each full turn:
    self._memory.on_turn_complete(turn_number, query, result)
    self._memory.save()

CHECKER INTEGRATION (future):
    When CheckerAgent runs the code and finds failures, call:
        self._memory.record_check_failure(
            turn_number=N,
            description="pytest found 3 failing tests",
            reason="Missing import in auth.py",
            files_involved=["src/auth.py"],
        )
    The Planner will read this on the next turn and route to a repair, not
    spawn fresh SubAgents for what's already been built.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory

if TYPE_CHECKING:
    from ..agent.models import SubTask, AgentResult

logger = logging.getLogger(__name__)


class MemoryCoordinator:
    """
    Central memory bus for a session. Owns one EpisodicMemory (Layer 2)
    and creates WorkingMemory instances (Layer 1) on demand per SubAgent.

    Layer 3 (Qdrant) is NOT owned here — it's already managed by AgentLoop
    via VectorStore + EmbeddingPipeline. This coordinator wraps layers 1 & 2.
    """

    def __init__(self, session_dir: Path, project_root: Path):
        self._session_dir = session_dir
        self._project_root = project_root

        # Layer 2 — load or create
        self.episodic = EpisodicMemory.load_or_create(session_dir)

        # Layer 1 — keyed by agent_id; each SubAgent gets its own instance
        self._working_memories: dict[str, WorkingMemory] = {}

        logger.info(
            "MemoryCoordinator: session_dir=%s, episodic=%s turns",
            session_dir,
            len(self.episodic.turn_summaries),
        )

    # ─── Session bootstrap ────────────────────────────────────────────────────

    def bootstrap(self) -> None:
        """
        Run once at session start:
          - Detect project facts from disk (language, framework, test dir, etc.)
          - Persist to episodic so Planner has it from turn 1.
        """
        self.episodic.detect_and_set_project_facts(self._project_root)
        self.episodic.save()
        logger.info("MemoryCoordinator: bootstrap complete, %d facts detected",
                    len(self.episodic.project_facts))

    # ─── Layer 1: Working memory ──────────────────────────────────────────────

    def create_working_memory(self, agent_id: str, task_description: str) -> WorkingMemory:
        """Create a fresh WorkingMemory for a SubAgent."""
        wm = WorkingMemory(task_description=task_description, agent_id=agent_id)
        self._working_memories[agent_id] = wm
        return wm

    def get_working_memory(self, agent_id: str) -> Optional[WorkingMemory]:
        """Retrieve an existing WorkingMemory by agent_id."""
        return self._working_memories.get(agent_id)

    def release_working_memory(self, agent_id: str) -> Optional[WorkingMemory]:
        """Discard a SubAgent's working memory (end of task lifetime)."""
        return self._working_memories.pop(agent_id, None)

    # ─── Layer 1 → Layer 2 harvesting ────────────────────────────────────────

    def on_task_complete(self, subtask: "SubTask", wm: WorkingMemory) -> None:
        """
        Called when a SubAgent finishes a task.

        Harvests Layer 1 data into Layer 2:
          - Files written → file registry
          - Errors → failed approaches (if unresolved)
          - files_written dict → attached to subtask for CheckerAgent
        """
        # Attach files_written to subtask for CheckerAgent
        subtask.files_written = wm.files_written  # type: ignore[attr-defined]
        subtask.working_memory_summary = wm.summary()  # type: ignore[attr-defined]

        # Harvest into episodic
        for path in wm.files_written:
            self.episodic.update_file_record(path, written=True)

        for path in wm.files_read_paths:
            self.episodic.update_file_record(path, read=True)

        # Unresolved errors → potential failed approaches
        for err in wm.unresolved_errors:
            self.episodic.record_failed_approach(
                turn_number=len(self.episodic.turn_summaries),
                description=f"[{subtask.id}] {err.tool}: {err.message[:80]}",
                reason=err.message[:200],
                files_involved=list(wm.files_written.keys()),
                do_not_repeat=False,  # errors in working memory aren't necessarily plan-level failures
            )

        logger.debug(
            "Memory harvest: subtask=%s, files_written=%d, unresolved_errors=%d",
            subtask.id, len(wm.files_written), len(wm.unresolved_errors),
        )

    def on_turn_complete(
        self,
        turn_number: int,
        query: str,
        result: "AgentResult",
    ) -> None:
        """
        Called after a full agent turn (Planner → Orchestrator → Synthesiser).

        Extracts the key finding from the result and stores a turn summary.
        """
        outcome = "failed" if result.sleep_mode else ("success" if result.success else "partial")
        key_finding = _extract_key_finding(result)

        self.episodic.record_turn(
            turn_number=turn_number,
            query=query,
            outcome=outcome,
            files_modified=result.files_modified,
            commands_run=result.commands_run,
            key_finding=key_finding,
        )

        # If sleep mode, auto-add a planner hint
        if result.sleep_mode and result.health_report:
            reason = result.health_report.sleep_reason
            self.episodic.add_planner_hint(
                f"Turn {turn_number} entered sleep mode ({reason}). "
                f"Consider simpler, single-step subtasks for similar queries."
            )

        logger.info("MemoryCoordinator: turn %d recorded [%s]", turn_number, outcome)

    # ─── Checker integration point (future) ──────────────────────────────────

    def record_check_failure(
        self,
        turn_number: int,
        description: str,
        reason: str,
        files_involved: list[str],
    ) -> None:
        """
        Called by CheckerAgent when code execution fails.

        This is the primary entry point for the checker → planner feedback loop:
          1. CheckerAgent runs the code, finds an error
          2. Calls this method with a detailed failure description
          3. Planner reads episodic.failed_approaches on next turn
          4. Planner knows to route to repair (single big LLM call) instead
             of spawning fresh SubAgents
        """
        self.episodic.record_failed_approach(
            turn_number=turn_number,
            description=description,
            reason=reason,
            files_involved=files_involved,
            do_not_repeat=True,  # checker failures are plan-level failures
        )
        # Hint the Planner to use repair mode
        self.episodic.add_planner_hint(
            f"Turn {turn_number}: code execution failed. "
            f"Prefer REPAIR mode (single LLM call) over spawning new SubAgents. "
            f"Files: {', '.join(files_involved[:3])}"
        )
        self.episodic.save()
        logger.info(
            "MemoryCoordinator: check failure recorded [turn=%d] %s",
            turn_number, description[:60],
        )

    # ─── Persistence ─────────────────────────────────────────────────────────

    def save(self) -> None:
        """Persist Layer 2 to disk. Call after every turn."""
        self.episodic.save()
        logger.debug("MemoryCoordinator: episodic saved")

    # ─── Planner injection helper ─────────────────────────────────────────────

    def planner_context(self) -> str:
        """Return the Layer 2 context block for injection into the Planner."""
        return self.episodic.to_planner_context()


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _extract_key_finding(result: "AgentResult") -> str:
    """
    Pull the most useful single-sentence finding from an AgentResult.
    Used for turn summary compression.
    """
    if not result.success and result.error:
        return f"Failed: {result.error[:150]}"

    # Use the first sentence of the response, or truncate
    response = result.response or ""
    first_sentence = response.split(".")[0].strip()
    if first_sentence:
        return first_sentence[:150]

    if result.files_modified:
        return f"Modified {len(result.files_modified)} file(s): {', '.join(result.files_modified[:3])}"

    return response[:150] or "(no output)"