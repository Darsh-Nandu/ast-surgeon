# Sovereign-Code: Full Implementation Plan (Standalone)

This is a complete, self-contained handoff plan for a sequence of LLM sessions
that will evolve Sovereign-Code from its current state to the target
architecture shown in the system diagram. Each phase below is fully specified
— a fresh LLM session with no other context, given only this document and the
repo, should be able to execute one phase end-to-end.

**Global rules for every phase:**
- Work inside `Sovereign-Code/src/components/agent/` unless told otherwise.
- New agents follow the existing `SubAgent`/`Orchestrator`/`CheckerAgent` style:
  dataclasses/enums in `models.py`, routing entries in `router.py`, agent
  classes as their own files in `agent/`.
- Every new agent must emit `StepTrace`/`HealthSignal` entries the same way
  `SubAgent` does, so existing health-aggregation keeps working.
- After each phase: run `pytest tests/` and do a manual smoke run via
  `src/cli/main.py` if possible. Note any new env vars/dependencies in
  `pyproject.toml` and `README.md`.
- Do **not** attempt two phases in one session — each is sized for one context
  window with room for iteration.
- When a phase says a component "already exists", verify by reading the file
  before writing new code — implementations may have evolved.

---

## 1. Current State of the Codebase

Pipeline today: `Planner → Orchestrator → SubAgent(s) → CheckerAgent (Phase 1 +
Phase 2 checks) → Synthesiser`.

- **`loop.py`** — `AgentLoop`, top-level entry, owns session + memory + the
  `checker_enabled` toggle (persisted via episodic memory, settable via CLI
  `--check` flag or `/checker on|off`).
- **`planner.py`** — turns a user query into a `TaskPlan` (DAG of `SubTask`s).
- **`orchestrator.py`** — executes the DAG (direct or parallel), aggregates
  `PipelineHealth`, calls `CheckerAgent.check_pipeline()` after all subtasks
  finish (~line 91-100), and has an **old stub** `_repair_hook` (~line 306) for
  `SubAgentSignal.SLEEPING` health-based failures (separate from checker
  failures — logs only, does nothing yet).
- **`subagent.py`** — ReAct-style loop agent calling tools (`file_tools.py`).
  After every `write_file`/`edit_file`, calls `checker.check_static(path,
  content)` if the checker is enabled (~line 373-380). Has sleep-mode detection
  (`PipelineHealth`, `SleepReason`, `SubAgentSignal`).
- **`checker.py`** (`CheckerAgent`, 710 lines) — **already substantially
  implemented**:
  - `check_static(path, content) -> StaticCheckResult`: per-file syntax check
    (`py_compile` / `tsc --noEmit` / `node --check`), content-hash cached via
    `CheckCache` at `.sovereign/sessions/<id>/check_cache.json`. Zero LLM calls.
    This **is** the diagram's "PHASE 1 CHECKER".
  - `check_pipeline(all_files_written, memory_coordinator, turn_number) ->
    PipelineCheckResult`: infers an entry point/run command (heuristics first,
    then `TaskType.CHECK` LLM call), executes it via `RunCommandTool`'s
    allowlist, and on failure runs `_diagnose_failure()` (`TaskType.DEBUG`)
    producing `error_file`, `error_line`, `root_cause`, `repair_prompt`. This
    **is** the diagram's "PHASE 2 CHECKER" combined with most of "Error
    Detailer". Result cached by `frozenset(file_hashes)`.
  - `_record_failure()` calls `memory_coordinator.record_check_failure()`,
    which writes to episodic memory with `do_not_repeat=True` so the **next
    Planner turn** routes a `TaskType.REPAIR` subtask (mode=direct). This is a
    **cross-turn** loop only — no in-turn retry yet.
  - `verify_repair(all_files_written, memory_coordinator, turn_number, attempt)`
    exists with `REPAIR_RETRIES = 2`, but **nothing calls it yet** — there is no
    standalone `RepairAgent` and no orchestrator-side retry loop.
  - Toggle fully wired end-to-end: `CheckerAgent.enable()/disable()`,
    `MemoryCoordinator.checker` (lazy-created), `enable_checker()`/
    `disable_checker()`/`checker_enabled` property, CLI `--check` flag and
    `/checker on|off` slash command in `cli/session.py`. **Reuse this exact
    pattern for the Security Agent toggle.**
- **`router.py`** — `ModelRouter`, rule-based `TaskType → ModelConfig` table.
  `TaskType.CHECK` and `TaskType.REPAIR` rows already exist.
- **`synthesiser.py`** — composes final response from traces.
- **`models.py`** — shared dataclasses/enums: `TaskType` (includes `CHECK`,
  `REPAIR`), `SubTask`, `TaskPlan`, `PipelineHealth`, `SleepReason`,
  `SubAgentSignal`, `AgentResult` (already has `check_result:
  Optional[PipelineCheckResult]`), `StepTrace`, `StaticCheckResult`,
  `PipelineCheckResult`.
- **`tools/file_tools.py`** — `ToolRegistry` with read/write/edit/list/search/run
  tools; `RunCommandTool` has an allowlist (`_is_allowed`) used by
  `check_pipeline`'s `_execute_pipeline`.
- **`memory/`** — working + episodic memory, `MemoryCoordinator` (owns the
  lazily-created `CheckerAgent` and the checker toggle/persistence).
- **`sync/`** — file indexer + manifest + watcher feeding Qdrant.
- **Not yet implemented anywhere**: GitHub integration, Research Agent, KG
  Maker Agent, System Info Manager Agent, Security Agent, standalone
  RepairAgent, in-turn repair retry loop, typed/specialized subagents
  (frontend/logic/test/docs), human-in-the-loop permission gates before running
  code or pushing to git.

## 2. Target Architecture (from the diagram)

```
INPUT → MAIN ORCHESTRATOR ─┬─> SystemInfoAgent.gather() (venv/deps/RAM/OS, cached)
                            ├─> PLANNER → ORCHESTRATOR → [SUB AGENT 1..N (typed: frontend/
                            │       backend_logic/test/docs)] — needs user permission to run
                            │       └─ write_file/edit_file → Phase 1 Checker (check_static)
                            │             └─ fail → RepairAgent (static) → re-check (1 retry)
                            ├─> Phase 2 Checker (check_pipeline; smart — checks if project is
                            │       complete enough to run; needs user permission to execute;
                            │       uses SystemInfoAgent summary for venv/RAM context)
                            │       └─ fail → Error Detailer (_diagnose_failure, already mostly
                            │             built) → RepairAgent (pipeline) → verify_repair
                            │             [loop, bounded by REPAIR_RETRIES]
                            ├─> RESEARCH AGENT — on-demand, callable via shared callback from
                            │       SubAgent, RepairAgent, CheckerAgent's diagnosis step
                            ├─> KG MAKER AGENT — runs after check_pipeline passes; incremental,
                            │       shared KG injected into Planner + SubAgent prompts
                            ├─> SECURITY AGENT — optional/toggleable (same pattern as checker
                            │       toggle), human selects which findings to fix → RepairAgent
                            ├─> SYNTHESISER
                            └─> GIT MANAGER AGENT — sole holder of GitHub tools (separate
                                    registry); commit always, push needs permission
                            → OUTPUT
```

---

## PHASE 0 — Shared Models & Stubs

**Goal:** Add every shared dataclass/enum/router-row/stub module that later
phases depend on, without changing runtime behavior.

**Tasks:**

1. In `models.py`, **verify what already exists** (`TaskType.CHECK`/`REPAIR`,
   `StaticCheckResult`, `PipelineCheckResult`, `AgentResult.check_result`) —
   don't duplicate. Then add:
   - New `TaskType` entries: `RESEARCH`, `KG_BUILD`, `SECURITY_SCAN`,
     `SYSTEM_INFO`, `GIT_OPS`.
   - New enum `SubAgentRole`: `GENERAL`, `FRONTEND`, `BACKEND_LOGIC`, `TEST`,
     `DOCS`.
   - Add `role: SubAgentRole = SubAgentRole.GENERAL` field to `SubTask`.
   - New `RepairRequest` dataclass — a thin wrapper around the existing check
     result types:
     ```python
     @dataclass
     class RepairRequest:
         source: str  # "static" | "pipeline"
         static_result: Optional["StaticCheckResult"] = None
         pipeline_result: Optional["PipelineCheckResult"] = None
         attempt: int = 1
         give_up: bool = False
     ```
   - Extend `AgentResult` with: `repair_attempts: int = 0`,
     `kg_updated: bool = False`,
     `security_findings: list[dict] = field(default_factory=list)`,
     `git_summary: Optional[dict] = None`.

2. In `router.py`, add routing-table rows for the five new `TaskType`s above
   (don't touch existing `CHECK`/`REPAIR` rows):
   - `RESEARCH` → `llama-3.1-8b-instant` (groq) — cheap query formulation.
   - `KG_BUILD` → `gemini-2.0-flash` — structured summarization.
   - `SECURITY_SCAN` → `llama-3.3-70b-versatile` (groq).
   - `SYSTEM_INFO` → `llama-3.1-8b-instant` (groq) — light summarization only.
   - `GIT_OPS` → `llama-3.1-8b-instant` (groq) — commit message generation.

3. Create empty stub modules (docstring + `TODO` + minimal imports so import
   paths resolve for later phases):
   - `agent/system_info_agent.py`
   - `agent/repair_agent.py`
   - `agent/research_agent.py`
   - `agent/kg_agent.py`
   - `agent/security_agent.py`
   - `agent/git_manager_agent.py`

**Verify:** `python -c "from src.components.agent import models, router"`
succeeds; `pytest tests/` passes unchanged.

---

## PHASE 0B — Main Orchestrator (top-level router)

**Goal:** The diagram's top box under INPUT is **"MAIN SMART ORCHESTRATOR"** —
a routing layer above the existing `Orchestrator` (which is really the
plan-DAG executor, i.e. the diagram's second-level "ORCHESTRATOR" that
dispatches SUB AGENT 1..N). Today `AgentLoop.run()` hardcodes
`Planner → Orchestrator → Synthesiser` for every query. We need a new
`MainOrchestrator` that sits between `AgentLoop` and everything else, and
decides **which top-level flow** a query triggers, per the diagram's three
annotated entry conditions:

- "THIS IS INITIATED TO GENERATE NEW CONTENT" → normal flow:
  `Planner → Orchestrator(SubAgents) → checks → KG → security → synth → git`.
- "THIS IS INITIATED IF USER ASKS TO RUN TESTS" → skip planning/generation,
  go straight to `Phase2Checker` (`check_pipeline`), with the permission gate.
- "THIS IS INITIATED IF USER STATED ONE/MANY FILES HAVE ERRORS AND HAVE
  REPAIRING TO DO" → skip planning, go straight to the Error
  Detailer/RepairAgent loop (`_diagnose_failure` + `RepairAgent` +
  `verify_repair`) on the named file(s), then `check_pipeline`.
- "IF PHASE ONE CHECKS FAIL" (loop-back arrow to top) → re-enter
  `MainOrchestrator` after a static-check failure triggers a repair, so the
  next routing decision re-evaluates from scratch (handles cases where a
  repair itself needs new files generated).

**File:** `agent/main_orchestrator.py`

**Tasks:**

1. New enum in `models.py`: `MainRoute(str, Enum)` with values `GENERATE`,
   `RUN_TESTS`, `REPAIR_FILES`. Add `route: Optional[MainRoute] = None` field
   to `AgentResult` for observability/logging.

2. `MainOrchestrator.route(query: str, conversation_history: list[dict],
   episodic_context: str) -> MainRoute`:
   - One cheap LLM call (`TaskType.PLANNING` or reuse `TaskType.CHECK`'s
     fast 8b model) with a short classification prompt: given the query +
     recent history, classify as `GENERATE` / `RUN_TESTS` / `REPAIR_FILES`.
   - Heuristic fast-paths before the LLM call (no-cost): if the query matches
     patterns like `r"\brun (the )?tests?\b"` → `RUN_TESTS`; if it matches
     `r"\b(fix|repair|error|bug) in .*\.(py|js|ts)"` → `REPAIR_FILES`;
     otherwise fall through to the LLM classification.
   - Default to `GENERATE` on ambiguity or LLM error.

3. `MainOrchestrator.run(query, ...) -> AgentResult` — the new top-level
   entry point that `AgentLoop.run()` calls instead of going straight to
   `Planner`:
   - `route = self.route(query, ...)`.
   - `GENERATE`: existing flow — `Planner.plan()` → `Orchestrator.execute()`
     → (Phase 1/2 checks, repair loop, KG, security — wired in later phases).
   - `RUN_TESTS`: skip `Planner`/`Orchestrator` entirely. Call
     `SystemInfoAgent.gather()` (Phase 1) then `CheckerAgent.check_pipeline()`
     directly on the current set of tracked project files (use the indexer/
     manifest from `sync/` to get the current file list + contents, since
     there's no `all_files_written` from a fresh plan). Apply the
     `permission_callback` gate (Phase 2) before executing.
   - `REPAIR_FILES`: parse file path(s) mentioned in the query (simple regex
     for `\S+\.(py|js|ts|tsx|jsx)` tokens, falling back to an LLM extraction
     call if none found). Read each file, run `check_static`/`check_pipeline`
     as appropriate to produce a `CheckResult`/`PipelineCheckResult`, run
     `_diagnose_failure` if not already failing-with-prompt, then invoke
     `RepairAgent` directly (Phase 2's class) — bypassing
     `Planner`/`Orchestrator`'s subagent dispatch entirely, matching the
     diagram's loop-back arrow straight into "Error detailer" → "Repair
     Agent".
   - In all branches, after the branch-specific work completes, if
     `check_pipeline` was run and it **failed at the static (Phase 1) level**,
     set `AgentResult.route` and have `AgentLoop` **re-invoke
     `MainOrchestrator.run()`** once more for the same turn (the diagram's "IF
     PHASE ONE CHECKS FAIL" loop back to INPUT/MAIN ORCHESTRATOR) — guard with
     a `_reentry: bool = False` internal flag to prevent infinite loops (max
     one re-entry per turn).

4. `AgentLoop.run()` (`loop.py`) — replace the direct
   `self._planner.plan(...)` / `self._orchestrator.execute(...)` calls with a
   single `self._main_orchestrator.run(query, ...)` call. Construct
   `MainOrchestrator` in `AgentLoop.__init__`/`_build`, passing it the
   `Planner`, `Orchestrator`, `router`, `vector_store`/`indexer` (for
   `RUN_TESTS`'s file-listing), and (once they exist in later phases)
   `SystemInfoAgent`/`CheckerAgent`/`RepairAgent` references.

**Note on later phases:** Phases 1–6 build the agents (`SystemInfoAgent`,
`RepairAgent`, `ResearchAgent`, `KGAgent`, `SecurityAgent`,
`GitManagerAgent`). Wherever those phases say "Orchestrator wiring", that
wiring point is still correct for the `GENERATE` route (the plan-DAG
executor), but `MainOrchestrator`'s `RUN_TESTS`/`REPAIR_FILES` routes will
also need access to the same agent instances — Phase 7 (final integration)
must ensure `MainOrchestrator` and `Orchestrator` **share single instances**
of `SystemInfoAgent`/`CheckerAgent`/`RepairAgent`/`ResearchAgent` per run
(construct once in `AgentLoop`, pass down to both) rather than each creating
their own.

**Verify:** `tests/test_main_orchestrator.py` — for each route, mock the
downstream calls and assert `MainOrchestrator.route()` classifies sample
queries correctly (heuristics + a mocked LLM fallback case); assert
`RUN_TESTS` route never calls `Planner.plan`; assert `REPAIR_FILES` route
extracts filenames from a query like `"fix the bug in src/auth.py"` and
calls `RepairAgent` without calling `Orchestrator.execute`; assert the
single-re-entry guard prevents more than one loop-back per turn.

---

## PHASE 1 — System Info Manager Agent

**Goal:** An agent that creates/activates a venv, checks/installs
dependencies, and reports OS/RAM info — feeding both the Planner (can a model
run here?) and the Phase 2 Checker (good communication, point 9 of the spec).

**File:** `agent/system_info_agent.py`

**Tasks:**

1. Add tools (new file `tools/system_tools.py`, registered into
   `ToolRegistry` alongside the existing tools in `file_tools.py`):
   - `GetSystemInfoTool`: returns OS name/version, Python version, total/
     available RAM in MB. Check `pyproject.toml` for `psutil`; if absent, add
     it (cleanest cross-platform RAM source) rather than hand-parsing
     `/proc/meminfo`.
   - `EnsureVenvTool`: creates `.venv` in project root if absent
     (`python -m venv .venv`), returns `{venv_path, python_executable}`.
   - `InstallDependenciesTool`: runs `pip install -r requirements.txt` (or
     resolves deps from `pyproject.toml` if no requirements file) inside the
     venv via subprocess (reuse `RunCommandTool`'s subprocess pattern), returns
     `{success, stdout, stderr}`.

2. `SystemInfoAgent` class — mostly deterministic tool orchestration, thin LLM
   summary at the end:
   - `gather() -> dict`: calls the three tools above, returns
     `{os, python_version, ram_total_mb, ram_available_mb, venv_path,
     python_executable, deps_status}`.
   - `can_run_model(estimated_model_ram_mb: int) -> tuple[bool, str]`: compares
     against `ram_available_mb`, returns `(bool, reasoning)`.
   - `summary_for_orchestrator() -> str`: short human-readable string for
     prompt injection (e.g. `"OS: Linux, Python 3.11, venv at .venv (python:
     .venv/bin/python), RAM: 14.2GB available / 16GB total"`).

3. **Orchestrator wiring**:
   - Add `self._system_info_agent` to `Orchestrator.__init__`.
   - In `execute()`, call `gather()` **once per run**, cache the result on the
     instance for that run.
   - Pass `system_info.summary_for_orchestrator()` into
     `CheckerAgent.check_pipeline(..., system_context: str = "")` — add this
     new optional kwarg to `check_pipeline` and thread it into
     `_infer_entry_point` (so the LLM knows to use `.venv/bin/python` in
     `command`) and `_diagnose_failure` (so it knows available RAM, can flag
     OOM-type failures).
   - **Fast-path dependency fix**: if `_diagnose_failure`'s `root_cause` /
     stderr contains `ModuleNotFoundError` or `ImportError`, the orchestrator
     should call `system_info_agent` to run `InstallDependenciesTool` (or
     `pip install <missing-module>` parsed from the error) and call
     `verify_repair()` again **without** invoking `RepairAgent` — cheapest
     possible fix path, doesn't count against `REPAIR_RETRIES`.

4. Add `TaskType.SYSTEM_INFO` usage: Planner may emit a subtask of this type
   when the query implies "set up the project"/"install deps"/"run this" —
   `SubAgent` should special-case this task type to call `SystemInfoAgent`
   directly instead of running a generic ReAct loop (cheap, deterministic).

**Verify:** `tests/test_system_info_agent.py` — mock subprocess calls, assert
`gather()` returns the expected dict shape; assert `can_run_model()` logic on
synthetic RAM values; assert `check_pipeline`'s prompts include
`system_context` when provided (mock `ModelRouter.call`, inspect the
`messages`/`system_prompt` passed).

---

## PHASE 2 — RepairAgent + In-Turn Repair Retry Loop + Permission Gate

**Goal:** This is the highest-value gap. Today a checker failure only sets up
context for the *next user turn* via `record_check_failure`. The diagram wants
an automatic detail→repair→recheck loop **within the same turn**, bounded by
`REPAIR_RETRIES`, plus a human-permission gate before executing code.

**File:** `agent/repair_agent.py`

**Tasks:**

1. **Investigate current REPAIR consumption** first: grep `planner.py` and
   `subagent.py` for `TaskType.REPAIR` and `do_not_repeat` to see how a
   repair-routed subtask is currently executed (likely just a normal `SubAgent`
   run with `task_type=REPAIR`). Document the finding in a comment at the top
   of `repair_agent.py` — `RepairAgent` below is a *new, more targeted* path
   that runs within the same turn, distinct from (and complementary to) that
   cross-turn mechanism.

2. **`RepairAgent` class**:
   - `__init__(router: ModelRouter, project_root: Path, tool_registry,
     research_callback: Optional[Callable[[str, str], str]] = None)`.
   - Per the `checker.py` docstring: "RepairAgent uses a single big LLM call,
     NO new SubAgents." Implement exactly that:
     - `repair(repair_request: RepairRequest, all_files_written: dict[str, str])
       -> dict[str, str]`:
       1. Build the failure description: for `source == "pipeline"`, use
          `repair_request.pipeline_result.repair_prompt` (+ `error_file`/
          `error_line`); for `source == "static"`, use
          `repair_request.static_result.error`.
       2. If the description contains the literal marker `NEEDS_RESEARCH:`
          (followed by a query) and `research_callback` is set, call
          `research_callback(query, context=description)` and append the
          result to the prompt before the main repair call.
       3. Gather full current content of the implicated file(s) from
          `all_files_written` (plus `error_file` if not already in the set —
          read via the tool registry).
       4. One LLM call, `TaskType.REPAIR`, system prompt instructing: "You are
          given a precise description of what's wrong. Output ONLY corrected
          file(s) as JSON: `{\"path\": \"<full new file content>\", ...}` — no
          markdown fences, no explanation."
       5. Parse JSON (reuse the fence-stripping `re.sub` pattern already used
          in `checker.py`'s `_infer_entry_point`/`_diagnose_failure`).
       6. Write each file via `tool_registry`'s `write_file` (so existing
          on-write hooks — re-indexing, `check_static` — fire normally).
       7. Return the `{path: new_content}` map. On parse failure or empty
          response, return `{}`.

3. **Orchestrator wiring — `_run_checker_repair_cycle`**:
   - After `check_pipeline()` returns `passed=False` (existing call site
     ~orchestrator.py line 91-100), instead of just storing the result:
     ```python
     attempt = 1
     while attempt <= REPAIR_RETRIES and not check_result.passed:
         if "ModuleNotFoundError" in check_result.stderr or "ImportError" in check_result.stderr:
             # Phase 1 fast-path: try dependency install first (doesn't consume attempt)
             ... (see Phase 1 step 3) ...
             check_result = checker.verify_repair(all_files_written, memory_coordinator, turn_number, attempt=attempt)
             if check_result.passed:
                 break
         repair_request = RepairRequest(source="pipeline", pipeline_result=check_result, attempt=attempt)
         updated = repair_agent.repair(repair_request, all_files_written)
         if not updated:
             repair_request.give_up = True
             break
         all_files_written.update(updated)
         check_result = checker.verify_repair(all_files_written, memory_coordinator, turn_number, attempt=attempt)
         attempt += 1
     result.repair_attempts = attempt - 1
     result.check_result = check_result
     ```
   - If still failing after the loop: existing `_record_failure`/
     `verify_repair` already wrote to episodic memory (cross-turn fallback
     stays intact). Additionally, `Synthesiser` should surface: "Automatic
     repair attempted `{repair_attempts}` time(s) but the pipeline still fails:
     `{check_result.root_cause}`." — extend `synthesiser.py`'s prompt assembly
     to include this when `not result.check_result.passed`.

4. **Static-check repair (Phase 1 of the diagram)** — in `subagent.py` at the
   existing `check_static()` call site (~line 373-380): if it fails, call
   `RepairAgent.repair(RepairRequest(source="static", static_result=static,
   attempt=1), {path: content_written})` **once** (no retry loop needed here —
   syntax errors are usually fixed in one shot), then re-run `check_static` on
   the result. If it still fails, continue the SubAgent's ReAct loop as before
   (don't block the agent — log it).

5. **Unify with sleep-mode `_repair_hook`**: update the old stub
   (~orchestrator.py line 306) to also go through `RepairAgent`. Build:
   ```python
   pipeline_result = PipelineCheckResult(
       passed=False,
       repair_prompt="\n".join(f"{s.signal.value}: {s.detail}" for s in subtask.health.signals),
   )
   repair_request = RepairRequest(source="pipeline", pipeline_result=pipeline_result, attempt=1)
   ```
   then call `repair_agent.repair(...)`. This unifies both repair paths
   (checker-failure and sleep-mode) through one `RepairAgent`.

6. **Permission gate before executing code** (diagram's "NEEDS USERS
   PERMISSION" on Phase 2 Checker):
   - Add `permission_callback: Optional[Callable[[str], bool]] = None` param to
     `CheckerAgent.check_pipeline` and `verify_repair`.
   - Before `_execute_pipeline(command)`: if `permission_callback` is set, call
     `permission_callback(f"Run: {command}?")`. If `False`, return
     `PipelineCheckResult(passed=False, command_used=command,
     repair_prompt="Execution skipped — user denied permission")` without
     running anything.
   - Default `None` → **deny** (safe default for API/non-interactive mode).
   - Thread the callback: `cli/session.py` implements it via `rich`'s
     `Confirm.ask(...)` (already used given existing `console.print` calls);
     pass it through `AgentLoop.run()` → `Orchestrator.execute()` →
     `MemoryCoordinator`/`CheckerAgent` call sites.

**Verify:**
- `tests/test_repair_agent.py` — mock router to return a JSON file-fix, assert
  `repair()` writes the file via the tool registry and returns the updated
  map; assert malformed/empty JSON → `{}`. Assert `research_callback` is
  invoked when the description contains `NEEDS_RESEARCH:`.
- `tests/test_repair_loop.py` — mock `check_pipeline` to fail once, mock
  `RepairAgent.repair` to return a fix, mock `verify_repair` to then pass.
  Assert `result.repair_attempts == 1` and `result.check_result.passed is
  True`. Also test the `give_up` path (`repair()` returns `{}` → loop exits,
  failure message present in synthesised response).
- `tests/test_checker_permission.py` — pass a callback returning `False`,
  assert `check_pipeline` returns `passed=False` with the "denied" message and
  `subprocess.run` was never called (patch and assert).
- `tests/test_static_repair.py` — mock `check_static` to fail then pass after
  one `RepairAgent.repair()` call inside `SubAgent`'s write-file flow.

---

## PHASE 3 — Research Agent + KG Maker Agent

**Goal:** A shared research capability usable by any agent, and an incremental
knowledge-graph builder triggered once the pipeline check passes.

**Files:** `agent/research_agent.py`, `agent/kg_agent.py`

### 3a. Research Agent

1. `ResearchAgent.research(query: str, context: str = "") -> str`:
   - Use `TaskType.RESEARCH` to first turn `query` + `context` into 1-3
     concrete search queries.
   - Define a `SearchProvider` protocol: `.search(query: str) -> list[dict]`
     (each dict: `title`, `url`, `snippet`).
   - Implement `StackOverflowSearchProvider` (Stack Exchange API — no key
     required for low-volume use, called out specifically in the spec) and a
     generic `WebSearchProvider` (httpx GET to a configurable endpoint, e.g.
     DuckDuckGo's HTML endpoint with basic parsing). Both wrapped in
     try/except — on failure, return `[]` rather than raising.
   - Summarize results via one LLM call into a concise answer string. Keep
     this internal/non-user-facing (used only to inform other agents' prompts).

2. **Shared access via callback** (not direct imports, to avoid circular
   dependencies): `Orchestrator.execute()` instantiates **one**
   `ResearchAgent` per run and passes a bound `research_agent.research` as
   `research_callback` to:
   - `SubAgent` (Phase 0/2 wiring) — add a `research(query)` tool to the
     ReAct tool registry, registered in `_format_tool_schemas` like any other
     tool, that calls the callback.
   - `RepairAgent` (Phase 2) — already specified to accept this.
   - `CheckerAgent._diagnose_failure` — if the diagnosis is uncertain, it can
     emit `NEEDS_RESEARCH: <query>` in its `repair_prompt`, which `RepairAgent`
     then resolves (Phase 2 step 2).

3. Emit a `StepTrace` with `tool_name="research"` whenever the callback is
   invoked, so it shows up in traces/UI like any other tool call.

### 3b. KG Maker Agent

1. On-disk format: JSON at `.sovereign/sessions/<id>/knowledge_graph.json`:
   ```json
   {"nodes": {"<file_path>": {"summary": "...", "symbols": [...], "depends_on": [...]}},
    "edges": [{"from": "...", "to": "...", "type": "imports|calls|defines"}]}
   ```

2. `KGAgent.build_or_update(files_modified: list[str], project_root: Path,
   session_dir: Path) -> dict` (the KG):
   - Load existing KG if present, else start empty.
   - For each modified file: one `TaskType.KG_BUILD` LLM call — given the file
     content (or a chunk summary via the existing `chunker/` module), produce
     `{summary, symbols, depends_on}`. Replace that file's node entirely.
   - **Incremental/smart update** (per spec point 5): files *not* modified keep
     their nodes untouched. Deleted files (diff previous node-set against a
     current directory scan) have their nodes and any referencing edges
     removed.
   - Recompute edges only for changed nodes (diff `depends_on` → edges).
   - Persist to `knowledge_graph.json`.

3. `KGAgent.context_for(query: str, top_k: int = 5) -> str`: simple
   relevance match (reuse `embed_pipeline` if available, else keyword match)
   against node summaries → formatted text block for prompt injection.

4. **Orchestrator wiring**: trigger `KGAgent.build_or_update()` immediately
   after `check_pipeline()` (post repair-loop) returns `passed=True`. Set
   `AgentResult.kg_updated = True` on success. Inject
   `KGAgent.context_for(query)` into:
   - `Planner.plan()` — add an optional `kg_context: str` param to its prompt
     assembly.
   - `SubAgent._build_initial_message()` — append KG context block.

**Verify:**
- `tests/test_research_agent.py` — mock `SearchProvider`, assert `research()`
  returns a non-empty string and never raises on provider exceptions.
- `tests/test_kg_agent.py` — build KG for 2 fake files, assert nodes/edges
  created; modify one file and rebuild, assert only that node/its edges
  changed; remove a file from the modified set entirely (simulate deletion),
  assert its node/edges are pruned.

---

## PHASE 4 — Specialized SubAgents + Smart Orchestrator Routing

**Goal:** Typed subagents (Frontend, Backend/Logic, Test, Docs) per spec point
7, with the Planner assigning roles and the Orchestrator/SubAgent honoring
them.

**Tasks:**

1. `SubAgentRole` enum and `SubTask.role` field already added in Phase 0.

2. `planner.py`: extend the plan-generation prompt so the LLM assigns a `role`
   per subtask based on its description — frontend = UI/CSS/React/HTML;
   backend_logic = APIs/business logic/data processing; test = test files;
   docs = README/docstrings/comments; default `GENERAL`. Parse `role` from the
   LLM's JSON plan output into `SubTask.role`.

3. `subagent.py`: make `_build_system_prompt()` role-aware via a
   `ROLE_PROMPTS: dict[SubAgentRole, str]` constant with role-specific
   guidance, e.g.:
   - `FRONTEND`: "Follow the project's existing component patterns; prefer
     accessible, semantic markup; keep styling consistent with existing
     CSS/Tailwind conventions."
   - `BACKEND_LOGIC`: "Prioritize correctness, explicit error handling, and
     type hints; avoid side effects in pure functions."
   - `TEST`: "Write pytest-style tests; cover edge cases and failure paths;
     mock external dependencies."
   - `DOCS`: "Write clear, concise documentation matching the project's
     existing tone; avoid restating code verbatim."
   Append the role prompt to the base system prompt.

4. Optional model override per role: add a small `ROLE_OVERRIDES` dict in
   `router.py` keyed by `(TaskType, SubAgentRole)` → `model_id`, applied in
   `ModelRouter.call` if a `role` kwarg is passed. Thread `subtask.role` from
   `SubAgent` into its `router.call(...)` invocations.

5. Confirm `Orchestrator._run_one_task` already has access to `subtask.role`
   when constructing `SubAgent` (it has `subtask` already — just verify
   `subagent.py` reads `self._subtask.role`).

**Verify:** `tests/test_subagent_roles.py` — construct subtasks with each
role, assert `_build_system_prompt()` output contains role-specific text; mock
planner LLM response including role assignments, assert `Planner.plan()`
populates `SubTask.role` correctly for each.

---

## PHASE 5 — Security Agent (Toggleable, Human-in-the-Loop)

**Goal:** Spec point 8 — optional security scanning with a curated
common-mistakes checklist, human selects which fixes to apply.

**File:** `agent/security_agent.py`

**Tasks:**

1. Create `agent/data/security_checklist.md` — a curated list of common
   LLM-generated-code security issues: hardcoded secrets/API keys, SQL
   injection via string formatting/concatenation, missing authentication/
   authorization checks, insecure `eval`/`exec` usage, weak or missing password
   hashing, missing input validation/sanitization, CORS misconfiguration,
   insecure deserialization, missing rate limiting on sensitive endpoints,
   logging of sensitive data.

2. **Toggle — copy the existing `checker_enabled` pattern exactly**:
   - `SecurityAgent.enable()`/`.disable()`, an `enabled` property.
   - `MemoryCoordinator`: add `security_agent` (lazy-created, mirroring
     `.checker`), `enable_security_agent()`/`disable_security_agent()`/
     `security_agent_enabled` property, persisted via episodic fact
     `security_agent_enabled`.
   - CLI: `--security` flag in `cli/main.py` (mirrors `--check`), `/security
     on|off` slash command in `cli/session.py` (mirrors `/checker on|off`).
   - Config: `.sovereign/config.json` key `security_agent_enabled` (default
     `True` per spec — note this differs from checker's default-off; confirm
     against `.sovereign/config.json` schema and document clearly).

3. `SecurityAgent.scan(files: list[str], project_root: Path) -> list[dict]`:
   - Returns `[]` immediately if `not self.enabled`.
   - For each file: one `TaskType.SECURITY_SCAN` LLM call with
     `security_checklist.md` injected as reference, asking for
     `[{file, line, issue, severity, suggested_fix}, ...]`. Parse JSON
     (reuse the fence-stripping pattern).

4. **Orchestrator wiring**: run `SecurityAgent.scan()` on `files_modified`
   after `check_pipeline()` passes and (if Phase 3 KG step ran) after the KG
   update. Store results in `AgentResult.security_findings`.

5. **Human-in-the-loop**: in `cli/session.py`, after a turn completes, if
   `result.security_findings` is non-empty, present a numbered list and let
   the user select which `suggested_fix`es to apply (e.g. comma-separated
   indices, or "all"/"none"). For each selected finding, build:
   ```python
   static_result = StaticCheckResult(path=finding["file"], passed=False,
                                       error=f"{finding['issue']} (line {finding['line']}): {finding['suggested_fix']}")
   repair_request = RepairRequest(source="static", static_result=static_result, attempt=1)
   ```
   and call `RepairAgent.repair()` (Phase 2) — this is "this info will go again
   to main orchestrator" from the spec, implemented by re-entering the repair
   path with these specific fixes. After applying, re-run `check_static` on
   each touched file.

**Verify:** `tests/test_security_agent.py` — `enabled=False` → `scan()`
returns `[]` without calling the router; mock LLM response with findings,
assert correct parsing into the expected dict shape; assert config toggle is
read correctly from `.sovereign/config.json` and the `/security on|off`
slash command updates `MemoryCoordinator.security_agent_enabled` and persists
via episodic fact.

---

## PHASE 6 — Git Manager Agent (Sole Holder of GitHub Tools)

**Goal:** Spec point 3 — final stage before output; this is the **only** agent
with GitHub/git tool access.

**Files:** `agent/git_manager_agent.py`, new `tools/git_tools.py`

**Tasks:**

1. `tools/git_tools.py` — new tool classes registered into a **dedicated**
   `GitToolRegistry`, separate from the general `ToolRegistry` used by
   `SubAgent`/`RepairAgent`/`SecurityAgent`/etc., to enforce "no other agent
   will have access to these tools":
   - `GitInitTool` (`git init` if no `.git` dir present).
   - `GitStatusTool`, `GitAddTool`, `GitCommitTool`, `GitPushTool` — subprocess
     calls scoped to `project_root` (reuse `RunCommandTool`'s subprocess
     pattern from `file_tools.py`).
   - `GitHubCreateRepoTool` / `GitHubPushTool`: GitHub REST API via `httpx`,
     authenticated with `GITHUB_TOKEN` env var (`api.github.com` is reachable
     in this environment's network config).

2. `GitManagerAgent`:
   - A focused loop (similar shape to `SubAgent` but using only
     `GitToolRegistry`).
   - Input: the turn's `AgentResult` summary (files modified, plan summary,
     `result.check_result.passed`).
   - One `TaskType.GIT_OPS` LLM call to generate a commit message describing
     the changes, and to decide: create a new repo (first run, no `.git`) vs.
     commit to existing.
   - Sequence: `git add <files_modified>` → `git commit -m <generated message>`
     → if `.sovereign/config.json` has `github_remote` configured and
     `auto_push: true`, push (subject to permission gate below); otherwise
     leave committed locally and report what a push would do.
   - Returns `AgentResult.git_summary = {"committed": bool, "commit_sha": str,
     "pushed": bool, "remote": Optional[str], "message": str}`.

3. **Orchestrator/`loop.py` wiring**: `GitManagerAgent` runs as the **last**
   step in `AgentLoop.run()`, after `Synthesiser`, only if
   `result.files_modified` is non-empty and (`result.check_result is None or
   result.check_result.passed` OR the user explicitly forces it via a future
   flag). Commit can happen automatically/locally; **push requires the
   `permission_callback`** from Phase 2 (`permission_callback("Push commit
   <sha> to <remote>?")`).

4. Add a prominent comment at the top of `git_tools.py` stating the
   single-consumer constraint explicitly, and grep the rest of the codebase
   after implementation to confirm no other agent file imports it.

**Verify:** `tests/test_git_manager_agent.py` — use a temp git repo (real
subprocess calls against a tmpdir are fine/cheap), mock
`GitHubCreateRepoTool`'s HTTP call, assert `git add`+`commit` actually happen
on disk and `git_summary` has the expected shape; assert push is skipped when
`permission_callback` returns `False` or is `None`.

---

## PHASE 7 — Final Integration, Config & Docs Pass

**Goal:** Wire every phase together into one coherent pipeline, finalize
config schema, update docs.

**Tasks:**

1. `loop.py` — full pipeline order, matching the diagram:
   ```
   SystemInfoAgent.gather()              (cached per run; Phase 1)
   → Planner.plan(..., kg_context=KGAgent.context_for(query))   (Phase 3)
   → Orchestrator.execute(plan)
        → per-subtask: SubAgent.run()    (role-aware; Phase 4; research_callback wired; Phase 3)
             → write_file/edit_file → check_static()            (existing)
                  → fail → RepairAgent (static, 1 retry)         (Phase 2)
        → check_pipeline(system_context=..., permission_callback=...)  (Phase 1 + 2)
             → fail → RepairAgent (pipeline) loop → verify_repair       (Phase 2)
        → if passed: KGAgent.build_or_update()                   (Phase 3)
        → SecurityAgent.scan() (if enabled) → human selects → RepairAgent (static)  (Phase 5)
   → Synthesiser
   → GitManagerAgent (commit always if files changed; push needs permission)  (Phase 6)
   ```

2. `.sovereign/config.json` — finalize and document schema. Existing key:
   `checker_enabled` (default `False`). New keys: `security_agent_enabled`
   (default `True`), `github_remote` (string or null), `auto_push` (default
   `False`), `research_provider` (`"stackoverflow" | "web" | "none"`, default
   `"stackoverflow"`).

3. `cli/session.py`/`cli/main.py`: implement `permission_callback` (used by
   `check_pipeline`, `verify_repair`, and `GitManagerAgent`'s push step) as an
   interactive `Confirm.ask(...)` prompt; in non-interactive/API mode default
   to deny and surface "permission required for: `<action>`" in
   `AgentResult`.

4. Remove now-dead code: old `_repair_hook` stub comments (now replaced by
   real `RepairAgent` calls per Phase 2 step 5), Phase 0 stub-module
   docstrings/TODOs once filled in.

5. Update top-level `README.md`: new architecture diagram (link to image or
   ASCII rendering of the pipeline above), required env vars (`GITHUB_TOKEN`,
   any research-provider keys), full `.sovereign/config.json` schema, and the
   new `/security on|off` and `--security` flags alongside existing
   `/checker`/`--check` docs.

6. Full test run: `pytest tests/ -v`. Add
   `tests/test_e2e_pipeline.py` — run a trivial query through `AgentLoop` with
   all external calls (LLM, subprocess, git, http) mocked, asserting the
   pipeline reaches `GitManagerAgent` and produces a populated `AgentResult`
   (`check_result`, `repair_attempts`, `kg_updated`, `security_findings`,
   `git_summary` all present and consistent with the mocked scenario,
   including a scenario where `check_pipeline` fails once and `repair_attempts
   == 1` after a successful retry).

---

## Suggested Session Boundaries (if token-constrained)

- **Phase 2** is the largest and most critical — split into:
  - **2a**: `RepairAgent` class + its unit tests only.
  - **2b**: orchestrator/SubAgent retry-loop wiring + permission gate +
    integration tests.
- **Phase 7** can split into:
  - **7a**: pipeline wiring in `loop.py`/`orchestrator.py`.
  - **7b**: docs, config schema finalization, e2e test.
- Phases 0, 1, 3, 4, 5, 6 each fit comfortably in one session as written.