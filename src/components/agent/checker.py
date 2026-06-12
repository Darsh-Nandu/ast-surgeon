"""
CheckerAgent — two-phase code verification with caching and redundancy prevention.

PHASE 1: Static check (per-file, fast, inline in SubAgent after each write)
  - Syntax check: python -m py_compile / tsc --noEmit / node --check
  - Result cached by (path, content_hash) — same content never re-checked
  - Runs in ~50-100ms, zero LLM calls

PHASE 2: Pipeline execution check (once, after all SubAgents done)
  - Step 1: LLM reads ALL files_written → infers entry point + run command
  - Step 2: Executes the command, captures stdout/stderr
  - Step 3: On failure, LLM diagnoses which file/line failed and why
  - Step 4: Builds a structured repair_prompt for the RepairAgent
  - Result cached by frozenset(file_hashes) — same set of files never re-run
  - RepairAgent uses a single big LLM call, NO new SubAgents

TOGGLE: Checker is disabled by default. Enable via:
  - CLI flag:  sovereign chat --check
  - Config:    .sovereign/config.json  { "checker_enabled": true }
  - Runtime:   session /checker on | /checker off
  - Per-turn:  Automatic for complex multi-file tasks, skipped for simple ones

REDUNDANCY:
  Both phases use content-hash-keyed caches persisted to:
    .sovereign/sessions/<id>/check_cache.json

  If a file was already statically checked with the same content, Phase 1 is
  a dict lookup (no subprocess). If the exact same set of files was already
  pipeline-checked successfully, Phase 2 is skipped entirely.

  This means: re-running the same query twice costs zero checker overhead.

REPAIR ROUTING:
  On Phase 2 failure:
    - repair_prompt is written to EpisodicMemory.failed_approaches
    - A planner hint is added: "REPAIR mode, single LLM call"
    - Next Planner turn routes to TaskType.REPAIR with mode=direct
    - RepairAgent (one LLM call) patches only the broken parts
    - Phase 2 re-runs to verify (max REPAIR_RETRIES times)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, TYPE_CHECKING

from .models import StaticCheckResult, PipelineCheckResult, TaskType
from .router import ModelRouter

if TYPE_CHECKING:
    from ..memory.coordinator import MemoryCoordinator

logger = logging.getLogger(__name__)

# ─── Tunables ─────────────────────────────────────────────────────────────────

STATIC_CHECK_TIMEOUT  = 15    # seconds per file
PIPELINE_RUN_TIMEOUT  = 60    # seconds for the full pipeline run
REPAIR_RETRIES        = 2     # max re-check attempts after a repair
MAX_OUTPUT_CHARS      = 4000  # cap on stdout/stderr sent to LLM

# ─── LLM prompts ──────────────────────────────────────────────────────────────

ENTRY_POINT_SYSTEM = """\
You are a code execution expert. Given a set of source files, determine:
1. The correct command to run the entire program / pipeline end-to-end
2. The likely execution order of the files (dependency order)

Respond ONLY with a valid JSON object — no markdown, no explanation:
{
  "command": "python main.py",
  "order": ["ingester.py", "transformer.py", "exporter.py"],
  "reasoning": "one sentence"
}

Rules:
- command must be a single shell command that runs from the project root
- order lists files in execution/dependency order, entry point first
- If there is no clear entry point, pick the most likely one
- Prefer: python main.py > python -m module > python src/main.py
- For JS/TS: node index.js > npx ts-node src/index.ts > node dist/index.js
"""

DIAGNOSE_SYSTEM = """\
You are a debugging expert. A code pipeline failed. Given the source files and
the error output, diagnose exactly what went wrong.

Respond ONLY with a valid JSON object — no markdown, no explanation:
{
  "error_file": "src/transformer.py",
  "error_line": 42,
  "error_type": "NameError",
  "root_cause": "one sentence describing the root cause",
  "repair_prompt": "Detailed instructions for fixing the bug. Include: which file, which line/function, what to change, why. Be specific enough that another LLM can fix it without seeing the error again."
}

Rules:
- error_file: the file that CAUSED the error (not always the one in the traceback)
- error_line: the line number (0 if unknown)
- repair_prompt: must be self-contained — include file name, function name, the bad code, and the fix
- If multiple files are broken, describe all fixes in repair_prompt
"""


# ─── Cache ────────────────────────────────────────────────────────────────────

def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def _files_hash(files: dict[str, str]) -> str:
    """Stable hash of a {path: content} dict — order-independent."""
    parts = sorted(f"{p}:{_content_hash(c)}" for p, c in files.items())
    return hashlib.sha256("|".join(parts).encode()).hexdigest()[:16]


class CheckCache:
    """
    Persistent cache for check results.

    Stored at: .sovereign/sessions/<id>/check_cache.json

    Two sub-caches:
      static_checks:   {file_hash → StaticCheckResult dict}
      pipeline_checks: {files_hash → PipelineCheckResult dict}

    On cache hit: result is returned immediately, subprocess/LLM skipped.
    Cache is invalidated automatically when content changes (hash mismatch).
    """

    def __init__(self, cache_path: Path):
        self._path = cache_path
        self._static: dict[str, dict] = {}      # content_hash → result
        self._pipeline: dict[str, dict] = {}    # files_hash → result
        self._load()

    def get_static(self, content_hash: str) -> Optional[StaticCheckResult]:
        data = self._static.get(content_hash)
        if data:
            result = StaticCheckResult(**data)
            result.cached = True
            return result
        return None

    def set_static(self, content_hash: str, result: StaticCheckResult) -> None:
        self._static[content_hash] = asdict(result)
        self._save()

    def get_pipeline(self, files_hash: str) -> Optional[PipelineCheckResult]:
        data = self._pipeline.get(files_hash)
        if data:
            result = PipelineCheckResult(**data)
            result.cached = True
            return result
        return None

    def set_pipeline(self, files_hash: str, result: PipelineCheckResult) -> None:
        self._pipeline[files_hash] = asdict(result)
        self._save()

    def _load(self) -> None:
        if self._path.exists():
            try:
                data = json.loads(self._path.read_text(encoding="utf-8"))
                self._static = data.get("static_checks", {})
                self._pipeline = data.get("pipeline_checks", {})
                logger.debug(
                    "CheckCache loaded: %d static, %d pipeline entries",
                    len(self._static), len(self._pipeline),
                )
            except Exception as exc:
                logger.warning("CheckCache load failed: %s", exc)

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {"static_checks": self._static, "pipeline_checks": self._pipeline}
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._path)


# ─── CheckerAgent ─────────────────────────────────────────────────────────────

class CheckerAgent:
    """
    Two-phase code checker with caching and repair routing.

    Usage:
        checker = CheckerAgent(
            project_root=Path("/my/project"),
            router=router,
            cache_path=session_dir / "check_cache.json",
            enabled=True,          # can be toggled at runtime
        )

        # Phase 1 — after each SubAgent write_file:
        result = checker.check_static(path="src/auth.py", content="...")
        if not result.passed:
            # inject error back into SubAgent or abort

        # Phase 2 — after all SubAgents done:
        result = checker.check_pipeline(
            all_files_written={"src/auth.py": "...", "main.py": "..."},
            memory_coordinator=memory,
            turn_number=3,
        )
        if not result.passed:
            # repair_prompt is in result.repair_prompt
            # memory already updated — Planner will route to REPAIR next turn
    """

    def __init__(
        self,
        project_root: Path,
        router: ModelRouter,
        cache_path: Path,
        enabled: bool = False,
    ):
        self._root = project_root
        self._router = router
        self._cache = CheckCache(cache_path)
        self.enabled = enabled

    # ─── Toggle ───────────────────────────────────────────────────────────────

    def enable(self) -> None:
        self.enabled = True
        logger.info("CheckerAgent: ENABLED")

    def disable(self) -> None:
        self.enabled = False
        logger.info("CheckerAgent: DISABLED")

    # ─── Phase 1: Static check ────────────────────────────────────────────────

    def check_static(self, path: str, content: str) -> StaticCheckResult:
        """
        Fast per-file syntax check. Returns cached result if content unchanged.

        Called by SubAgent immediately after a write_file or edit_file tool call.
        Zero LLM calls. Uses py_compile / tsc / node --check depending on extension.
        """
        if not self.enabled:
            return StaticCheckResult(path=path, passed=True, check_type="disabled")

        ch = _content_hash(content)

        # Cache hit — same content was already checked
        cached = self._cache.get_static(ch)
        if cached:
            logger.debug("Phase 1 cache hit: %s [%s]", path, ch)
            return cached

        result = self._run_static_check(path, content)
        self._cache.set_static(ch, result)
        return result

    def _run_static_check(self, path: str, content: str) -> StaticCheckResult:
        ext = Path(path).suffix.lower()

        if ext == ".py":
            return self._static_python(path, content)
        elif ext in (".ts", ".tsx"):
            return self._static_typescript(path, content)
        elif ext in (".js", ".jsx", ".mjs"):
            return self._static_javascript(path, content)
        else:
            # Unsupported extension — skip check, mark as passed
            return StaticCheckResult(path=path, passed=True, check_type="unsupported")

    def _static_python(self, path: str, content: str) -> StaticCheckResult:
        """Use py_compile on the content string (no temp file needed)."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp = f.name

        try:
            result = subprocess.run(
                ["python", "-m", "py_compile", tmp],
                capture_output=True, text=True, timeout=STATIC_CHECK_TIMEOUT,
            )
            passed = result.returncode == 0
            # Normalise error: replace temp path with real path
            error = (result.stderr or "").replace(tmp, path).strip()
            return StaticCheckResult(
                path=path, passed=passed, error=error, check_type="syntax"
            )
        except Exception as exc:
            return StaticCheckResult(
                path=path, passed=False, error=str(exc), check_type="syntax"
            )
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _static_typescript(self, path: str, content: str) -> StaticCheckResult:
        """Quick tsc typecheck on a single file."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ts", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp = f.name
        try:
            result = subprocess.run(
                ["tsc", "--noEmit", "--allowJs", "--checkJs", "--strict", tmp],
                capture_output=True, text=True, timeout=STATIC_CHECK_TIMEOUT,
            )
            passed = result.returncode == 0
            error = (result.stdout + result.stderr).replace(tmp, path).strip()
            return StaticCheckResult(path=path, passed=passed, error=error, check_type="syntax")
        except FileNotFoundError:
            # tsc not installed — skip
            return StaticCheckResult(path=path, passed=True, check_type="tsc_missing")
        except Exception as exc:
            return StaticCheckResult(path=path, passed=False, error=str(exc), check_type="syntax")
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    def _static_javascript(self, path: str, content: str) -> StaticCheckResult:
        """node --check for basic JS syntax."""
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".js", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            tmp = f.name
        try:
            result = subprocess.run(
                ["node", "--check", tmp],
                capture_output=True, text=True, timeout=STATIC_CHECK_TIMEOUT,
            )
            passed = result.returncode == 0
            error = (result.stderr or "").replace(tmp, path).strip()
            return StaticCheckResult(path=path, passed=passed, error=error, check_type="syntax")
        except FileNotFoundError:
            return StaticCheckResult(path=path, passed=True, check_type="node_missing")
        except Exception as exc:
            return StaticCheckResult(path=path, passed=False, error=str(exc), check_type="syntax")
        finally:
            try:
                os.unlink(tmp)
            except Exception:
                pass

    # ─── Phase 2: Pipeline execution check ───────────────────────────────────

    def check_pipeline(
        self,
        all_files_written: dict[str, str],
        memory_coordinator: Optional["MemoryCoordinator"],
        turn_number: int,
    ) -> PipelineCheckResult:
        """
        Full pipeline execution check. Steps:
          1. Cache check — if same files were already verified, return immediately
          2. LLM call — infer entry point + run command from file set
          3. Execute the command, capture output
          4. On failure: LLM diagnoses error + builds repair_prompt
          5. Write failure to EpisodicMemory so Planner routes to REPAIR

        Returns PipelineCheckResult with .passed and .repair_prompt.
        """
        if not self.enabled:
            return PipelineCheckResult(passed=True, command_used="(checker disabled)")

        if not all_files_written:
            return PipelineCheckResult(passed=True, command_used="(no files to check)")

        fh = _files_hash(all_files_written)

        # Cache hit — same exact set of files was already checked and passed
        cached = self._cache.get_pipeline(fh)
        if cached:
            if cached.passed:
                logger.info("Phase 2 cache hit (PASS): %s", fh)
                return cached
            else:
                # Failed before — don't re-run, return the cached failure
                # (The repair should have changed the content, producing a new hash)
                logger.info("Phase 2 cache hit (FAIL): %s — repair needed", fh)
                return cached

        logger.info(
            "Phase 2: checking %d files [hash=%s]", len(all_files_written), fh
        )

        # Step 1: Infer entry point
        entry = self._infer_entry_point(all_files_written)
        if not entry:
            result = PipelineCheckResult(
                passed=False,
                command_used="(could not determine entry point)",
                repair_prompt=(
                    "The checker could not determine how to run the generated files. "
                    "Please add a clear entry point (main.py / index.js / main.sh) "
                    "that runs the full pipeline."
                ),
                files_checked=list(all_files_written.keys()),
            )
            self._record_failure(result, memory_coordinator, turn_number, all_files_written)
            self._cache.set_pipeline(fh, result)
            return result

        command = entry["command"]
        logger.info("Phase 2: inferred command=%r", command)

        # Step 2: Execute
        exec_result = self._execute_pipeline(command)

        if exec_result["passed"]:
            result = PipelineCheckResult(
                passed=True,
                command_used=command,
                stdout=exec_result["stdout"],
                stderr=exec_result["stderr"],
                exit_code=exec_result["exit_code"],
                files_checked=list(all_files_written.keys()),
            )
            self._cache.set_pipeline(fh, result)
            logger.info("Phase 2: PASSED (exit_code=0)")
            return result

        # Step 3: Diagnose failure
        logger.warning(
            "Phase 2: FAILED (exit_code=%d) — diagnosing...",
            exec_result["exit_code"]
        )
        diagnosis = self._diagnose_failure(
            all_files_written=all_files_written,
            command=command,
            stdout=exec_result["stdout"],
            stderr=exec_result["stderr"],
        )

        result = PipelineCheckResult(
            passed=False,
            command_used=command,
            stdout=exec_result["stdout"],
            stderr=exec_result["stderr"],
            exit_code=exec_result["exit_code"],
            files_checked=list(all_files_written.keys()),
            error_file=diagnosis.get("error_file", ""),
            error_line=diagnosis.get("error_line", 0),
            repair_prompt=diagnosis.get("repair_prompt", exec_result["stderr"][:500]),
        )

        self._record_failure(result, memory_coordinator, turn_number, all_files_written)
        self._cache.set_pipeline(fh, result)
        return result

    # ─── Entry point inference ────────────────────────────────────────────────

    def _infer_entry_point(self, files: dict[str, str]) -> Optional[dict]:
        """
        Ask the LLM to read all file names + first 20 lines of each,
        then return the run command and dependency order.

        Falls back to heuristic detection if LLM call fails.
        """
        # Heuristic fast path — check common entry points first (no LLM needed)
        heuristic = self._heuristic_entry_point(files)
        if heuristic:
            logger.debug("Entry point heuristic: %s", heuristic["command"])
            return heuristic

        # LLM inference
        file_summaries = []
        for path, content in files.items():
            preview = "\n".join(content.splitlines()[:20])
            file_summaries.append(f"### {path}\n```\n{preview}\n```")

        user_prompt = (
            f"These files were just written by a coding agent.\n\n"
            f"{'---'.join(file_summaries)}\n\n"
            f"What command runs this program end-to-end? Respond with JSON only."
        )

        response = self._router.call(
            task_type=TaskType.CHECK,
            system_prompt=ENTRY_POINT_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        if response.is_error:
            logger.warning("Entry point LLM call failed: %s", response.content[:100])
            return None

        try:
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response.content).strip()
            return json.loads(cleaned)
        except Exception as exc:
            logger.warning("Entry point JSON parse failed: %s", exc)
            return None

    def _heuristic_entry_point(self, files: dict[str, str]) -> Optional[dict]:
        """
        Fast heuristic before calling LLM. Detects common entry point patterns.
        Returns entry dict or None if no clear match.
        """
        paths = list(files.keys())
        names = {Path(p).name: p for p in paths}

        # Python entry points
        for candidate in ["main.py", "app.py", "run.py", "pipeline.py", "start.py"]:
            if candidate in names:
                return {"command": f"python {names[candidate]}", "order": paths, "reasoning": "heuristic"}

        # Python module
        if "__main__.py" in names:
            # find the package dir
            pkg = str(Path(names["__main__.py"]).parent)
            return {"command": f"python -m {pkg.replace('/', '.')}", "order": paths, "reasoning": "heuristic __main__"}

        # JS/TS entry points
        for candidate in ["index.js", "main.js", "app.js", "index.ts", "main.ts"]:
            if candidate in names:
                ext = Path(candidate).suffix
                cmd = f"node {names[candidate]}" if ext == ".js" else f"npx ts-node {names[candidate]}"
                return {"command": cmd, "order": paths, "reasoning": "heuristic"}

        # Shell scripts
        for candidate in ["run.sh", "start.sh", "pipeline.sh"]:
            if candidate in names:
                return {"command": f"bash {names[candidate]}", "order": paths, "reasoning": "heuristic"}

        return None

    # ─── Pipeline execution ───────────────────────────────────────────────────

    def _execute_pipeline(self, command: str) -> dict:
        """Run the pipeline command and return {passed, stdout, stderr, exit_code}."""
        # Safety: only allow commands from the RunCommandTool allowlist
        from ..tools.file_tools import RunCommandTool
        dummy = RunCommandTool(self._root)
        if not dummy._is_allowed(command):
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"Command not in allowlist: {command!r}",
                "exit_code": -1,
            }

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=PIPELINE_RUN_TIMEOUT,
            )
            stdout = result.stdout[:MAX_OUTPUT_CHARS]
            stderr = result.stderr[:MAX_OUTPUT_CHARS]
            return {
                "passed": result.returncode == 0,
                "stdout": stdout,
                "stderr": stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {
                "passed": False,
                "stdout": "",
                "stderr": f"Pipeline timed out after {PIPELINE_RUN_TIMEOUT}s",
                "exit_code": -1,
            }
        except Exception as exc:
            return {
                "passed": False,
                "stdout": "",
                "stderr": str(exc),
                "exit_code": -1,
            }

    # ─── Failure diagnosis ────────────────────────────────────────────────────

    def _diagnose_failure(
        self,
        all_files_written: dict[str, str],
        command: str,
        stdout: str,
        stderr: str,
    ) -> dict:
        """
        Ask the LLM to read all source files + the error output and
        pinpoint exactly what went wrong and how to fix it.

        Returns diagnosis dict with keys: error_file, error_line, root_cause, repair_prompt.
        Falls back to raw stderr if LLM call fails.
        """
        file_blocks = []
        for path, content in all_files_written.items():
            # Number the lines so LLM can reference them precisely
            numbered = "\n".join(
                f"{i+1:4d}  {line}"
                for i, line in enumerate(content.splitlines())
            )
            file_blocks.append(f"### {path}\n```\n{numbered}\n```")

        user_prompt = (
            f"Command run: {command}\n\n"
            f"STDOUT:\n{stdout[:1000] or '(empty)'}\n\n"
            f"STDERR:\n{stderr[:2000] or '(empty)'}\n\n"
            f"Source files:\n\n{'---'.join(file_blocks)}\n\n"
            f"Diagnose the failure and provide a repair prompt. Respond with JSON only."
        )

        response = self._router.call(
            task_type=TaskType.DEBUG,
            system_prompt=DIAGNOSE_SYSTEM,
            messages=[{"role": "user", "content": user_prompt}],
        )

        if response.is_error:
            return {
                "error_file": "",
                "error_line": 0,
                "root_cause": "LLM diagnosis failed",
                "repair_prompt": (
                    f"Pipeline failed with exit code ≠ 0.\n"
                    f"Command: {command}\n"
                    f"Error output:\n{stderr[:1000]}"
                ),
            }

        try:
            cleaned = re.sub(r"```(?:json)?\s*|\s*```", "", response.content).strip()
            return json.loads(cleaned)
        except Exception:
            return {
                "error_file": "",
                "error_line": 0,
                "root_cause": response.content[:200],
                "repair_prompt": response.content[:500],
            }

    # ─── Memory integration ───────────────────────────────────────────────────

    def _record_failure(
        self,
        result: PipelineCheckResult,
        memory_coordinator: Optional["MemoryCoordinator"],
        turn_number: int,
        all_files_written: dict[str, str],
    ) -> None:
        """Write check failure into EpisodicMemory so Planner routes to REPAIR."""
        if memory_coordinator is None:
            return

        description = (
            f"Pipeline check failed: {result.command_used!r} exited {result.exit_code}"
            + (f" in {result.error_file}:{result.error_line}" if result.error_file else "")
        )
        memory_coordinator.record_check_failure(
            turn_number=turn_number,
            description=description,
            reason=result.repair_prompt[:300],
            files_involved=list(all_files_written.keys()),
        )

    # ─── Repair verification ──────────────────────────────────────────────────

    def verify_repair(
        self,
        all_files_written: dict[str, str],
        memory_coordinator: Optional["MemoryCoordinator"],
        turn_number: int,
        attempt: int = 1,
    ) -> PipelineCheckResult:
        """
        Re-run Phase 2 after a repair. Called by Orchestrator after RepairAgent
        completes. Max REPAIR_RETRIES attempts.

        The files_hash will be different (repair changed the content),
        so the cache will miss and the check will re-execute.
        """
        if attempt > REPAIR_RETRIES:
            return PipelineCheckResult(
                passed=False,
                command_used="(repair retries exhausted)",
                repair_prompt=(
                    f"Repair failed after {REPAIR_RETRIES} attempt(s). "
                    f"Manual intervention required."
                ),
                files_checked=list(all_files_written.keys()),
            )

        logger.info(
            "CheckerAgent: verify_repair attempt %d/%d", attempt, REPAIR_RETRIES
        )
        return self.check_pipeline(
            all_files_written=all_files_written,
            memory_coordinator=memory_coordinator,
            turn_number=turn_number,
        )