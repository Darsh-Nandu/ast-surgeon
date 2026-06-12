"""
MainOrchestrator — top-level routing layer that sits between AgentLoop and
the rest of the pipeline.

Architecture (from the project diagram):
  INPUT → MAIN SMART ORCHESTRATOR → one of three flows:
    GENERATE      Normal flow: Planner → Orchestrator(SubAgents) → checks → …
    RUN_TESTS     Skip planning; run check_pipeline directly on tracked files.
    REPAIR_FILES  Skip planning; diagnose + repair named file(s) directly.

Phase 0B delivers:
  - MainRoute classification (heuristic fast-paths + LLM fallback)
  - MainOrchestrator.run() that dispatches to the right flow
  - Single re-entry guard for the "IF PHASE ONE CHECKS FAIL" loop-back arrow
  - AgentLoop integration point (Phase 7 wires the real agent instances)

Later phases add:
  Phase 1 — SystemInfoAgent.gather() called at RUN_TESTS entry
  Phase 2 — RepairAgent.repair() + permission_callback gate wired in
  Phase 3 — KGAgent / ResearchAgent post-GENERATE hooks
  Phase 5 — SecurityAgent toggle
  Phase 6 — GitManagerAgent post-commit hook
  Phase 7 — Shared single instances passed from AgentLoop to both
             MainOrchestrator and Orchestrator
"""

from __future__ import annotations

import logging
import re
from typing import Callable, Optional

from .models import (
    AgentMode,
    AgentResult,
    MainRoute,
    TaskType,
)
from .router import ModelRouter

logger = logging.getLogger(__name__)


# ─── Heuristic patterns ───────────────────────────────────────────────────────

_RUN_TESTS_RE = re.compile(
    r"\b(run|execute|trigger|start)\s+(the\s+)?(tests?|test\s+suite|pytest|unittest)\b",
    re.IGNORECASE,
)
_REPAIR_FILES_RE = re.compile(
    r"\b(fix|repair|correct|resolve|debug)\b.{0,60}\S+\.(py|js|ts|tsx|jsx|go|rb|java)\b",
    re.IGNORECASE,
)


def _heuristic_route(query: str) -> Optional[MainRoute]:
    """Return a MainRoute from fast regex heuristics, or None to fall through."""
    if _RUN_TESTS_RE.search(query):
        return MainRoute.RUN_TESTS
    if _REPAIR_FILES_RE.search(query):
        return MainRoute.REPAIR_FILES
    return None


# ─── LLM classification prompt ────────────────────────────────────────────────

_CLASSIFICATION_SYSTEM = """\
You are a routing classifier for an AI coding agent.
Given a user query (and optionally recent conversation history), classify it as
exactly one of three intents:

GENERATE     — the user wants to create, extend, refactor, or explain code/docs.
RUN_TESTS    — the user explicitly asks to run, execute, or trigger tests.
REPAIR_FILES — the user has identified specific files with errors and wants them fixed.

Reply with ONLY one of the three words above, nothing else.
"""


def _llm_route(
    query: str,
    conversation_history: list[dict],
    router: ModelRouter,
) -> MainRoute:
    """Call the LLM for route classification. Defaults to GENERATE on failure."""
    recent = conversation_history[-4:] if len(conversation_history) > 4 else conversation_history
    messages = recent + [{"role": "user", "content": query}]
    try:
        resp = router.call(
            task_type=TaskType.CHECK,      # lightweight 8b model
            system_prompt=_CLASSIFICATION_SYSTEM,
            messages=messages,
        )
        text = resp.content.strip().upper()
        if "RUN_TESTS" in text or text == "RUN_TESTS":
            return MainRoute.RUN_TESTS
        if "REPAIR_FILES" in text or text == "REPAIR_FILES":
            return MainRoute.REPAIR_FILES
        return MainRoute.GENERATE
    except Exception as exc:
        logger.warning("MainOrchestrator: LLM classification failed (%s) → GENERATE", exc)
        return MainRoute.GENERATE


# ─── File-path extraction for REPAIR_FILES route ─────────────────────────────

_FILE_PATH_RE = re.compile(r"\S+\.(py|js|ts|tsx|jsx|go|rb|java)\b")


def _extract_file_paths(query: str, router: ModelRouter) -> list[str]:
    """Extract file paths from the query; falls back to an LLM extraction call."""
    paths = _FILE_PATH_RE.findall(query)
    # findall returns the capture group only — re-scan for full match
    paths = _FILE_PATH_RE.findall(query)
    full_paths: list[str] = []
    for m in _FILE_PATH_RE.finditer(query):
        full_paths.append(m.group(0))
    if full_paths:
        return full_paths

    # LLM fallback
    try:
        resp = router.call(
            task_type=TaskType.CHECK,
            system_prompt=(
                "Extract every file path mentioned in the user message. "
                "Reply with one path per line, nothing else. "
                "If no paths are found, reply with an empty line."
            ),
            messages=[{"role": "user", "content": query}],
        )
        return [line.strip() for line in resp.content.splitlines() if line.strip()]
    except Exception as exc:
        logger.warning("MainOrchestrator: file-path extraction LLM failed (%s)", exc)
        return []


# ─── MainOrchestrator ─────────────────────────────────────────────────────────

class MainOrchestrator:
    """Routes each user turn to the correct agent flow.

    Constructed once in AgentLoop.__init__ (Phase 7 wiring).
    For now, Planner and Orchestrator are passed in so the GENERATE branch
    delegates directly.  Phase 1–6 agent instances will be injected the same way.
    """

    def __init__(
        self,
        router: ModelRouter,
        planner,          # Planner instance — injected by AgentLoop
        orchestrator,     # Orchestrator instance — injected by AgentLoop
        # Phase 1+ (passed as None until each phase lands)
        system_info_agent=None,
        checker_agent=None,
        repair_agent=None,
        research_agent=None,
        kg_agent=None,
        security_agent=None,
        git_manager_agent=None,
        # Optional gate callback (Phase 2)
        permission_callback: Optional[Callable[[str], bool]] = None,
        # For RUN_TESTS: source of currently tracked project files
        indexer=None,
        vector_store=None,
    ) -> None:
        self._router = router
        self._planner = planner
        self._orchestrator = orchestrator
        self._system_info = system_info_agent
        self._checker = checker_agent
        self._repair = repair_agent
        self._research = research_agent
        self._kg = kg_agent
        self._security = security_agent
        self._git = git_manager_agent
        self._permission_cb = permission_callback
        self._indexer = indexer
        self._store = vector_store

    # ─── Public API ───────────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        conversation_history: list[dict],
        episodic_context: str = "",
    ) -> MainRoute:
        """Classify the query into a MainRoute.

        Fast heuristic check first; LLM call only on miss.
        """
        fast = _heuristic_route(query)
        if fast is not None:
            logger.debug("MainOrchestrator: heuristic route → %s", fast.value)
            return fast
        result = _llm_route(query, conversation_history, self._router)
        logger.debug("MainOrchestrator: LLM route → %s", result.value)
        return result

    def run(
        self,
        query: str,
        conversation_history: list[dict],
        episodic_context: str = "",
        codebase_summary: str = "",
        memory_coordinator=None,
        _reentry: bool = False,
    ) -> AgentResult:
        """Top-level entry point replacing the direct Planner call in AgentLoop.

        Args:
            query:                 The user's turn text.
            conversation_history:  Full session history so far.
            episodic_context:      Episodic memory string from MemoryCoordinator.
            codebase_summary:      Retrieval result from vector store.
            memory_coordinator:    MemoryCoordinator for checker toggle / facts.
            _reentry:              Internal flag — True when called a second time
                                   after a Phase-1 check failure in the same turn.
                                   Prevents infinite loops.

        Returns:
            AgentResult with .route set for observability.
        """
        chosen_route = self.route(query, conversation_history, episodic_context)

        logger.info(
            "MainOrchestrator.run [reentry=%s]: route=%s query=%.80s",
            _reentry, chosen_route.value, query,
        )

        if chosen_route == MainRoute.GENERATE:
            result = self._run_generate(
                query=query,
                conversation_history=conversation_history,
                episodic_context=episodic_context,
                codebase_summary=codebase_summary,
                memory_coordinator=memory_coordinator,
            )

        elif chosen_route == MainRoute.RUN_TESTS:
            result = self._run_tests(
                query=query,
                conversation_history=conversation_history,
                memory_coordinator=memory_coordinator,
            )

        else:  # REPAIR_FILES
            result = self._run_repair(
                query=query,
                conversation_history=conversation_history,
                memory_coordinator=memory_coordinator,
            )

        result.route = chosen_route

        # ── Re-entry guard: Phase-1 check failure loop-back ───────────────────
        # If the pipeline ran a static check and it failed, and this is not
        # already a re-entry, invoke MainOrchestrator once more for the same
        # turn so the routing decision is re-evaluated from scratch.
        if (
            not _reentry
            and result.check_result is not None
            and not result.check_result.passed
        ):
            logger.info(
                "MainOrchestrator: static check failed — re-entering (once) for turn"
            )
            reentry_result = self.run(
                query=query,
                conversation_history=conversation_history,
                episodic_context=episodic_context,
                codebase_summary=codebase_summary,
                memory_coordinator=memory_coordinator,
                _reentry=True,
            )
            return reentry_result

        return result

    # ─── Route handlers ───────────────────────────────────────────────────────

    def _run_generate(
        self,
        query: str,
        conversation_history: list[dict],
        episodic_context: str,
        codebase_summary: str,
        memory_coordinator,
    ) -> AgentResult:
        """GENERATE route: existing Planner → Orchestrator flow."""
        plan = self._planner.plan(
            query=query,
            codebase_summary=codebase_summary,
            conversation_history=conversation_history,
            episodic_context=episodic_context,
        )
        logger.info("MainOrchestrator [GENERATE]: plan=%s", plan.summary())
        result = self._orchestrator.execute(
            plan=plan,
            conversation_history=conversation_history,
            memory_coordinator=memory_coordinator,
        )
        # TODO (Phase 3): post-execute KGAgent.update() hook
        # TODO (Phase 5): post-execute SecurityAgent.scan() hook
        # TODO (Phase 6): post-execute GitManagerAgent.commit() hook
        return result

    def _run_tests(
        self,
        query: str,
        conversation_history: list[dict],
        memory_coordinator,
    ) -> AgentResult:
        """RUN_TESTS route: skip planning; run the checker directly.

        Phase 1 will inject SystemInfoAgent.gather() before the check.
        Phase 2 will apply the permission_callback gate.
        """
        logger.info("MainOrchestrator [RUN_TESTS]: bypassing Planner/Orchestrator")

        # TODO (Phase 1): call self._system_info.gather() to detect entry command
        # TODO (Phase 2): apply self._permission_cb gate before executing tests

        if self._checker is not None:
            # Phase 1+ path: real checker available
            # TODO (Phase 2): pass permission_callback to checker
            check_result = self._checker.check_pipeline(
                files=[],   # TODO (Phase 1): populate from indexer/manifest
            )
            return AgentResult(
                response=(
                    "Tests passed." if check_result.passed
                    else f"Tests failed:\n{check_result.stderr}"
                ),
                mode=AgentMode.DIRECT,
                plan=None,
                all_traces=[],
                files_modified=[],
                commands_run=[check_result.command_used] if check_result else [],
                success=check_result.passed,
                check_result=check_result,
            )

        # Stub path (Phase 0): checker not yet wired
        logger.warning(
            "MainOrchestrator [RUN_TESTS]: checker not available (Phase 1 not yet wired)"
        )
        return AgentResult(
            response=(
                "Test runner not yet available — checker will be wired in Phase 1. "
                "Please run tests manually for now."
            ),
            mode=AgentMode.DIRECT,
            plan=None,
            all_traces=[],
            files_modified=[],
            commands_run=[],
            success=False,
            error="checker_not_wired",
        )

    def _run_repair(
        self,
        query: str,
        conversation_history: list[dict],
        memory_coordinator,
    ) -> AgentResult:
        """REPAIR_FILES route: skip planning; diagnose + repair named files.

        Phase 2 will wire the real RepairAgent.  For now we extract paths and
        return a structured stub result so downstream code can verify routing.
        """
        file_paths = _extract_file_paths(query, self._router)
        logger.info(
            "MainOrchestrator [REPAIR_FILES]: extracted paths=%s", file_paths
        )

        if self._repair is not None and file_paths:
            # Phase 2+ path: real RepairAgent available
            # TODO (Phase 2): for each path, run check_static/check_pipeline,
            #                 build RepairRequest, call self._repair.repair(req)
            # TODO (Phase 2): apply self._permission_cb gate
            pass

        # Stub path (Phase 0): RepairAgent not yet wired
        if not file_paths:
            response = (
                "I couldn't identify specific file paths in your message. "
                "Please mention the file(s) you'd like repaired, e.g. 'fix the bug in src/auth.py'."
            )
            return AgentResult(
                response=response,
                mode=AgentMode.DIRECT,
                plan=None,
                all_traces=[],
                files_modified=[],
                commands_run=[],
                success=False,
                error="no_files_identified",
            )

        return AgentResult(
            response=(
                f"Repair requested for: {', '.join(file_paths)}. "
                "RepairAgent will be wired in Phase 2 — manual fix required for now."
            ),
            mode=AgentMode.DIRECT,
            plan=None,
            all_traces=[],
            files_modified=[],
            commands_run=[],
            success=False,
            error="repair_agent_not_wired",
        )
