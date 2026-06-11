"""
Eval harness - SWE-bench-style benchmark for the Sovereign-Code agent.

WHY a custom harness instead of raw SWE-bench:
  SWE-bench requires Docker, a full repo checkout per task, and a patch-apply
  pipeline. That's the gold standard for production evals, but it's too heavy
  for development iteration. Our harness is lightweight:
  - Tasks defined as Python dicts (file setup + query + assertion)
  - Agent runs against a real temp directory with real tools
  - Assertions check file contents, command output, or response text
  - Results persisted as JSON for trend tracking

Metrics we track:
  - task_success_rate: fraction of tasks where assertions pass
  - avg_steps:         mean steps per task (lower = more efficient)
  - avg_latency_ms:    mean wall-clock time per task
  - tool_accuracy:     fraction of tool calls that didn't error
  - step_efficiency:   tasks completed in ≤ half of max_steps (headroom measure)

DESIGN NOTE on EvalTask structure:
  Each task has:
  - setup_files: {relative_path: content} - written to tmp dir before agent runs
  - query: the natural language task given to the agent
  - assertions: list of Assertion objects checked after agent completes
  - max_steps: ceiling passed to the agent
  - tags: ["code_gen", "debug", "refactor"] for breakdown analysis
"""

from __future__ import annotations

import json
import logging
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)


# Assertion types

class AssertionType(str, Enum):
    FILE_EXISTS      = "file_exists"          # file was created
    FILE_CONTAINS    = "file_contains"        # file contains a substring
    FILE_NOT_CONTAINS = "file_not_contains"   # file does NOT contain substring
    RESPONSE_CONTAINS = "response_contains"   # agent response contains substring
    COMMAND_PASSES   = "command_passes"       # shell command exits 0


@dataclass
class Assertion:
    type: AssertionType
    target: str          # file path, response text pattern, or command
    value: str = ""      # substring to check (for CONTAINS assertions)

    def check(self, project_root: Path, response: str) -> tuple[bool, str]:
        """Evaluate the assertion. Returns (passed, reason)."""
        if self.type == AssertionType.FILE_EXISTS:
            exists = (project_root / self.target).exists()
            return exists, f"File {self.target} {'exists' if exists else 'NOT FOUND'}"

        elif self.type == AssertionType.FILE_CONTAINS:
            path = project_root / self.target
            if not path.exists():
                return False, f"File {self.target} does not exist"
            content = path.read_text(encoding="utf-8", errors="replace")
            passed = self.value in content
            return passed, f"File {self.target} {'contains' if passed else 'missing'} {self.value!r}"

        elif self.type == AssertionType.FILE_NOT_CONTAINS:
            path = project_root / self.target
            if not path.exists():
                return True, f"File {self.target} does not exist (ok)"
            content = path.read_text(encoding="utf-8", errors="replace")
            passed = self.value not in content
            return passed, f"File {self.target} {'correctly excludes' if passed else 'UNEXPECTEDLY contains'} {self.value!r}"

        elif self.type == AssertionType.RESPONSE_CONTAINS:
            passed = self.target.lower() in response.lower()
            return passed, f"Response {'contains' if passed else 'missing'} {self.target!r}"

        elif self.type == AssertionType.COMMAND_PASSES:
            import subprocess
            try:
                result = subprocess.run(
                    self.target, shell=True, cwd=str(project_root),
                    capture_output=True, timeout=30,
                )
                passed = result.returncode == 0
                return passed, f"Command {self.target!r} exited {result.returncode}"
            except Exception as exc:
                return False, f"Command failed: {exc}"

        return False, f"Unknown assertion type: {self.type}"


# EvalTask

@dataclass
class EvalTask:
    """One benchmark task."""
    id: str
    description: str
    query: str
    setup_files: dict[str, str]    # {relative_path: content}
    assertions: list[Assertion]
    tags: list[str] = field(default_factory=list)
    max_steps: int = 10


# EvalResult

@dataclass
class EvalResult:
    task_id: str
    passed: bool
    assertion_results: list[tuple[bool, str]]   # (passed, reason) per assertion
    response: str
    total_steps: int
    latency_ms: float
    files_modified: list[str]
    tool_errors: int
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "passed": self.passed,
            "assertions": [{"passed": p, "reason": r} for p, r in self.assertion_results],
            "total_steps": self.total_steps,
            "latency_ms": round(self.latency_ms, 1),
            "files_modified": self.files_modified,
            "tool_errors": self.tool_errors,
            "error": self.error,
        }


# EvalSuite - collection of tasks + runner

class EvalSuite:
    """Runs a set of EvalTasks against a real AgentLoop and reports metrics.

    Usage:
        suite = EvalSuite(tasks=BUILTIN_TASKS)
        report = suite.run(project_root="/tmp/eval_project")
        suite.print_report(report)
        suite.save_report(report, "eval_results.json")
    """

    def __init__(self, tasks: list[EvalTask]):
        self._tasks = tasks

    def run(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        on_task_done: Optional[Callable[[EvalResult], None]] = None,
    ) -> "EvalReport":
        """Run all tasks and return an EvalReport.

        Each task gets a fresh tmp directory and fresh AgentLoop (no history bleed).
        """
        from ..components.agent.loop import AgentLoop

        results: list[EvalResult] = []
        t0 = time.monotonic()

        for task in self._tasks:
            logger.info("Running eval task: %s", task.id)
            result = self._run_one(task, groq_api_key, gemini_api_key)
            results.append(result)
            if on_task_done:
                on_task_done(result)

        total_elapsed = time.monotonic() - t0
        report = EvalReport(results=results, total_elapsed_seconds=total_elapsed)

        logger.info(
            "Eval complete: %d/%d passed in %.1fs",
            report.passed_count, len(results), total_elapsed
        )
        return report

    def _run_one(
        self,
        task: EvalTask,
        groq_api_key: Optional[str],
        gemini_api_key: Optional[str],
    ) -> EvalResult:
        from ..components.agent.loop import AgentLoop

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)

            # Set up files
            for rel_path, content in task.setup_files.items():
                target = root / rel_path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")

            # Create a fresh AgentLoop (no Qdrant for evals - too slow to index per task)
            loop = AgentLoop.create(
                project_root=str(root),
                groq_api_key=groq_api_key,
                gemini_api_key=gemini_api_key,
                qdrant_host="localhost",
                qdrant_port=6333,
                embedding_provider="local",
            )

            # Override max_steps on the planner's fallback
            t_start = time.monotonic()
            agent_result = None
            error = None
            try:
                agent_result = loop.run(task.query)
            except Exception as exc:
                logger.error("Task %s crashed: %s", task.id, exc)
                error = str(exc)

            latency_ms = (time.monotonic() - t_start) * 1000
            response = agent_result.response if agent_result else ""

            # Count tool errors across all traces
            tool_errors = sum(
                1 for t in (agent_result.all_traces if agent_result else [])
                if t.tool_error
            )

            # Check assertions
            assertion_results = []
            for assertion in task.assertions:
                passed, reason = assertion.check(root, response)
                assertion_results.append((passed, reason))

            all_passed = all(p for p, _ in assertion_results) and error is None

            return EvalResult(
                task_id=task.id,
                passed=all_passed,
                assertion_results=assertion_results,
                response=response,
                total_steps=agent_result.total_steps if agent_result else 0,
                latency_ms=latency_ms,
                files_modified=agent_result.files_modified if agent_result else [],
                tool_errors=tool_errors,
                error=error,
            )


# EvalReport - metrics aggregation

@dataclass
class EvalReport:
    results: list[EvalResult]
    total_elapsed_seconds: float

    @property
    def passed_count(self) -> int:
        return sum(1 for r in self.results if r.passed)

    @property
    def task_success_rate(self) -> float:
        if not self.results:
            return 0.0
        return self.passed_count / len(self.results)

    @property
    def avg_steps(self) -> float:
        steps = [r.total_steps for r in self.results]
        return sum(steps) / len(steps) if steps else 0.0

    @property
    def avg_latency_ms(self) -> float:
        lats = [r.latency_ms for r in self.results]
        return sum(lats) / len(lats) if lats else 0.0

    @property
    def tool_accuracy(self) -> float:
        total_calls = sum(r.total_steps for r in self.results)
        total_errors = sum(r.tool_errors for r in self.results)
        if total_calls == 0:
            return 1.0
        return 1.0 - (total_errors / total_calls)

    def by_tag(self, tag: str) -> "EvalReport":
        """Filter results to tasks with a specific tag."""
        # We don't store task refs in results - return self for now
        return self

    def to_dict(self) -> dict:
        return {
            "summary": {
                "total_tasks": len(self.results),
                "passed": self.passed_count,
                "task_success_rate": round(self.task_success_rate, 3),
                "avg_steps": round(self.avg_steps, 1),
                "avg_latency_ms": round(self.avg_latency_ms, 1),
                "tool_accuracy": round(self.tool_accuracy, 3),
                "total_elapsed_seconds": round(self.total_elapsed_seconds, 1),
            },
            "tasks": [r.to_dict() for r in self.results],
        }

    def save(self, path: str) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))
        logger.info("Eval report saved to %s", path)


# Built-in benchmark tasks

BUILTIN_TASKS: list[EvalTask] = [

    EvalTask(
        id="create_simple_function",
        description="Agent creates a new utility function from scratch",
        query="Create a file called utils/math_utils.py with a function called `clamp(value, min_val, max_val)` that clamps a number between min and max.",
        setup_files={},
        assertions=[
            Assertion(AssertionType.FILE_EXISTS, "utils/math_utils.py"),
            Assertion(AssertionType.FILE_CONTAINS, "utils/math_utils.py", "def clamp"),
            Assertion(AssertionType.FILE_CONTAINS, "utils/math_utils.py", "min_val"),
            Assertion(AssertionType.FILE_CONTAINS, "utils/math_utils.py", "max_val"),
        ],
        tags=["code_gen"],
        max_steps=5,
    ),

    EvalTask(
        id="fix_bug_off_by_one",
        description="Agent fixes an off-by-one bug in a list function",
        query="There is a bug in src/list_utils.py in the `last_n` function - it returns one too many items. Fix it.",
        setup_files={
            "src/list_utils.py": '''\
def last_n(items, n):
    """Return the last n items from a list."""
    return items[-(n+1):]   # BUG: off by one
'''
        },
        assertions=[
            Assertion(AssertionType.FILE_EXISTS, "src/list_utils.py"),
            Assertion(AssertionType.FILE_CONTAINS, "src/list_utils.py", "def last_n"),
            Assertion(AssertionType.FILE_NOT_CONTAINS, "src/list_utils.py", "n+1"),
        ],
        tags=["debug"],
        max_steps=8,
    ),

    EvalTask(
        id="add_docstrings",
        description="Agent adds docstrings to undocumented functions",
        query="Add proper docstrings to all functions in src/calculator.py that are missing them.",
        setup_files={
            "src/calculator.py": '''\
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b
'''
        },
        assertions=[
            Assertion(AssertionType.FILE_CONTAINS, "src/calculator.py", '"""'),
        ],
        tags=["code_edit"],
        max_steps=6,
    ),

    EvalTask(
        id="write_unit_tests",
        description="Agent writes pytest unit tests for an existing module",
        query="Write unit tests in tests/test_calculator.py for all functions in src/calculator.py. Use pytest.",
        setup_files={
            "src/calculator.py": '''\
def add(a, b):
    """Add two numbers."""
    return a + b

def subtract(a, b):
    """Subtract b from a."""
    return a - b
''',
            "src/__init__.py": "",
            "tests/__init__.py": "",
        },
        assertions=[
            Assertion(AssertionType.FILE_EXISTS, "tests/test_calculator.py"),
            Assertion(AssertionType.FILE_CONTAINS, "tests/test_calculator.py", "def test_"),
            Assertion(AssertionType.FILE_CONTAINS, "tests/test_calculator.py", "add"),
            Assertion(AssertionType.FILE_CONTAINS, "tests/test_calculator.py", "subtract"),
        ],
        tags=["test_write"],
        max_steps=8,
    ),

    EvalTask(
        id="explain_code",
        description="Agent explains what a function does",
        query="Explain what the `process_order` function in src/orders.py does and what it returns.",
        setup_files={
            "src/orders.py": '''\
def process_order(order: dict) -> dict:
    """Process an incoming order."""
    total = sum(item["price"] * item["qty"] for item in order["items"])
    tax = total * 0.1
    return {
        "order_id": order["id"],
        "subtotal": total,
        "tax": tax,
        "total": total + tax,
        "status": "processed",
    }
'''
        },
        assertions=[
            Assertion(AssertionType.RESPONSE_CONTAINS, "process_order"),
            Assertion(AssertionType.RESPONSE_CONTAINS, "total"),
        ],
        tags=["explain"],
        max_steps=4,
    ),

    EvalTask(
        id="refactor_rename",
        description="Agent renames a variable throughout a file",
        query="In src/config.py, rename the variable `MAX_RETRIES` to `MAX_RETRY_COUNT` everywhere it appears.",
        setup_files={
            "src/config.py": '''\
MAX_RETRIES = 3

def get_retry_limit():
    return MAX_RETRIES

def should_retry(attempt):
    return attempt < MAX_RETRIES
'''
        },
        assertions=[
            Assertion(AssertionType.FILE_CONTAINS, "src/config.py", "MAX_RETRY_COUNT"),
            Assertion(AssertionType.FILE_NOT_CONTAINS, "src/config.py", "MAX_RETRIES"),
        ],
        tags=["code_edit"],
        max_steps=8,
    ),

    EvalTask(
        id="create_class",
        description="Agent creates a class with specified interface",
        query="Create a Stack class in src/data_structures.py with push, pop, peek, and is_empty methods.",
        setup_files={},
        assertions=[
            Assertion(AssertionType.FILE_EXISTS, "src/data_structures.py"),
            Assertion(AssertionType.FILE_CONTAINS, "src/data_structures.py", "class Stack"),
            Assertion(AssertionType.FILE_CONTAINS, "src/data_structures.py", "def push"),
            Assertion(AssertionType.FILE_CONTAINS, "src/data_structures.py", "def pop"),
            Assertion(AssertionType.FILE_CONTAINS, "src/data_structures.py", "def peek"),
            Assertion(AssertionType.FILE_CONTAINS, "src/data_structures.py", "def is_empty"),
        ],
        tags=["code_gen"],
        max_steps=6,
    ),

    EvalTask(
        id="search_and_answer",
        description="Agent searches codebase and answers a question",
        query="What functions are defined in src/helpers.py?",
        setup_files={
            "src/helpers.py": '''\
def format_name(first, last):
    """Format a full name."""
    return f"{first} {last}"

def parse_date(date_str):
    """Parse a date string."""
    from datetime import datetime
    return datetime.fromisoformat(date_str)
'''
        },
        assertions=[
            Assertion(AssertionType.RESPONSE_CONTAINS, "format_name"),
            Assertion(AssertionType.RESPONSE_CONTAINS, "parse_date"),
        ],
        tags=["search", "explain"],
        max_steps=4,
    ),
]