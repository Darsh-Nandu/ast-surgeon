"""
Tests for MainOrchestrator (Phase 0B).

Verifies:
  - route() classifies sample queries correctly via heuristics
  - route() falls back to LLM and returns a valid MainRoute
  - RUN_TESTS route never calls Planner.plan
  - REPAIR_FILES route extracts filenames and never calls Orchestrator.execute
  - GENERATE route calls both Planner.plan and Orchestrator.execute
  - Single re-entry guard prevents more than one loop-back per turn
"""

from __future__ import annotations

import sys
import os
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.components.agent.main_orchestrator import (
    MainOrchestrator,
    _heuristic_route,
    _extract_file_paths,
)
from src.components.agent.models import (
    AgentMode,
    AgentResult,
    MainRoute,
    TaskPlan,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_agent_result(success: bool = True, check_passed: bool = True) -> AgentResult:
    from src.components.agent.models import PipelineCheckResult
    cr = PipelineCheckResult(
        passed=check_passed,
        command_used="pytest",
        stdout="ok",
        stderr="" if check_passed else "FAILED",
    )
    return AgentResult(
        response="ok",
        mode=AgentMode.DIRECT,
        plan=None,
        all_traces=[],
        files_modified=[],
        commands_run=[],
        success=success,
        check_result=cr if not check_passed else None,
    )


def _make_plan() -> TaskPlan:
    from src.components.agent.models import AgentMode
    return TaskPlan(mode=AgentMode.DIRECT, subtasks=[], original_query="q", reasoning="r")


def _make_orchestrator(generate_result=None, run_tests_result=None, repair_result=None):
    """Build a MainOrchestrator with mocked Planner + Orchestrator."""
    router = MagicMock()
    router.call.return_value = MagicMock(content="GENERATE", is_error=False)

    planner = MagicMock()
    planner.plan.return_value = _make_plan()

    orchestrator = MagicMock()
    orchestrator.execute.return_value = generate_result or _make_agent_result()

    mo = MainOrchestrator(
        router=router,
        planner=planner,
        orchestrator=orchestrator,
    )
    return mo, planner, orchestrator, router


# ─── Heuristic routing ────────────────────────────────────────────────────────

@pytest.mark.parametrize("query,expected", [
    ("run the tests", MainRoute.RUN_TESTS),
    ("run tests please", MainRoute.RUN_TESTS),
    ("execute the test suite", MainRoute.RUN_TESTS),
    ("run pytest", MainRoute.RUN_TESTS),
    ("fix the bug in src/auth.py", MainRoute.REPAIR_FILES),
    ("repair errors in app/models.ts", MainRoute.REPAIR_FILES),
    ("correct the issue in utils.js", MainRoute.REPAIR_FILES),
    ("write a login function", None),
    ("explain how the auth module works", None),
    ("refactor the payment service", None),
])
def test_heuristic_route(query, expected):
    assert _heuristic_route(query) == expected


# ─── LLM fallback routing ─────────────────────────────────────────────────────

@pytest.mark.parametrize("llm_response,expected_route", [
    ("GENERATE", MainRoute.GENERATE),
    ("RUN_TESTS", MainRoute.RUN_TESTS),
    ("REPAIR_FILES", MainRoute.REPAIR_FILES),
    ("something unexpected", MainRoute.GENERATE),   # default on ambiguity
])
def test_route_llm_fallback(llm_response, expected_route):
    mo, _, _, router = _make_orchestrator()
    router.call.return_value = MagicMock(content=llm_response, is_error=False)
    # Use a query that won't hit heuristics
    result = mo.route("write a new module", [], "")
    assert result == expected_route


def test_route_llm_error_defaults_to_generate():
    mo, _, _, router = _make_orchestrator()
    router.call.side_effect = RuntimeError("network error")
    result = mo.route("some ambiguous query", [], "")
    assert result == MainRoute.GENERATE


# ─── GENERATE route ───────────────────────────────────────────────────────────

def test_generate_calls_planner_and_orchestrator():
    mo, planner, orchestrator, _ = _make_orchestrator()
    result = mo.run("write a new auth service", [], _reentry=False)
    planner.plan.assert_called_once()
    orchestrator.execute.assert_called_once()
    assert result.route == MainRoute.GENERATE


# ─── RUN_TESTS route ─────────────────────────────────────────────────────────

def test_run_tests_never_calls_planner():
    mo, planner, orchestrator, _ = _make_orchestrator()
    result = mo.run("run the tests", [], _reentry=False)
    planner.plan.assert_not_called()
    orchestrator.execute.assert_not_called()
    assert result.route == MainRoute.RUN_TESTS


def test_run_tests_stub_returns_informative_response():
    mo, _, _, _ = _make_orchestrator()
    result = mo.run("run tests", [], _reentry=False)
    assert "Phase 1" in result.response or "checker" in result.response.lower()


# ─── REPAIR_FILES route ───────────────────────────────────────────────────────

def test_repair_extracts_filename_and_skips_orchestrator():
    mo, planner, orchestrator, _ = _make_orchestrator()
    result = mo.run("fix the bug in src/auth.py", [], _reentry=False)
    planner.plan.assert_not_called()
    orchestrator.execute.assert_not_called()
    assert result.route == MainRoute.REPAIR_FILES
    assert "src/auth.py" in result.response


def test_repair_no_files_identified():
    mo, planner, orchestrator, router = _make_orchestrator()
    # Make LLM path return empty too
    router.call.return_value = MagicMock(content="", is_error=False)
    result = mo.run("fix the bug in my code", [], _reentry=False)
    assert result.route == MainRoute.REPAIR_FILES
    assert result.error == "no_files_identified"


@pytest.mark.parametrize("query,expected_files", [
    ("fix the bug in src/auth.py", ["src/auth.py"]),
    ("repair errors in app/models.ts and lib/utils.js", ["app/models.ts", "lib/utils.js"]),
    ("correct the issue in main.py", ["main.py"]),
])
def test_extract_file_paths_regex(query, expected_files):
    router = MagicMock()
    paths = _extract_file_paths(query, router)
    for f in expected_files:
        assert f in paths, f"Expected {f!r} in extracted paths {paths}"


# ─── Re-entry guard ───────────────────────────────────────────────────────────

def test_reentry_guard_prevents_infinite_loop():
    """A failed check_result triggers at most ONE re-entry."""
    fail_result = _make_agent_result(success=False, check_passed=False)
    mo, planner, orchestrator, _ = _make_orchestrator(generate_result=fail_result)

    call_count = {"n": 0}
    original_run_generate = mo._run_generate

    def counting_generate(**kwargs):
        call_count["n"] += 1
        return fail_result

    mo._run_generate = counting_generate

    # Force GENERATE route so the re-entry logic is exercised
    with patch.object(mo, "route", return_value=MainRoute.GENERATE):
        mo.run("write something", [], _reentry=False)

    # Should have been called twice total: once original + once re-entry
    assert call_count["n"] == 2, (
        f"Expected exactly 2 calls (original + 1 re-entry), got {call_count['n']}"
    )


def test_reentry_flag_blocks_second_reentry():
    """When _reentry=True, no further re-entry happens even on check failure."""
    fail_result = _make_agent_result(success=False, check_passed=False)
    mo, _, _, _ = _make_orchestrator(generate_result=fail_result)

    call_count = {"n": 0}

    def counting_generate(**kwargs):
        call_count["n"] += 1
        return fail_result

    mo._run_generate = counting_generate

    with patch.object(mo, "route", return_value=MainRoute.GENERATE):
        mo.run("write something", [], _reentry=True)   # already a re-entry

    assert call_count["n"] == 1, "Re-entry guard should have blocked a second loop"
