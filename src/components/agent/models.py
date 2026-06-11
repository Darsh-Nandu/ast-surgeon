"""
Agent data models — the shared language between every layer of the agent system.

CHANGES in this version:
  - Added PipelineHealth, SleepReason, and HealthSignal for sleep-mode detection
  - Added SubAgentSignal enum so parent agent can reason about child agent state
  - StepTrace now records tool_result_raw (untruncated) and health_flag
  - AgentResult exposes sleep_mode and health_report
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


# ─── Core Enums ──────────────────────────────────────────────────────────────

class TaskType(str, Enum):
    PLANNING     = "planning"
    CODE_GEN     = "code_gen"
    CODE_EDIT    = "code_edit"
    CODE_REVIEW  = "code_review"
    DEBUG        = "debug"
    TEST_WRITE   = "test_write"
    SEARCH       = "search"
    RUN          = "run"
    EXPLAIN      = "explain"
    SYNTHESISE   = "synthesise"


class SubTaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    SKIPPED    = "skipped"


class AgentMode(str, Enum):
    DIRECT   = "direct"
    PARALLEL = "parallel"


# ─── Health / Sleep models ────────────────────────────────────────────────────

class SleepReason(str, Enum):
    """Why the pipeline entered sleep mode."""
    LLM_PARSE_FAILURES   = "llm_parse_failures"    # too many JSON parse errors
    TOOL_LOOP_DETECTED   = "tool_loop_detected"     # agent stuck calling same tool
    SUBAGENT_TIMEOUT     = "subagent_timeout"       # child agent timed out
    CONSECUTIVE_ERRORS   = "consecutive_errors"     # N consecutive tool errors
    LLM_ERROR_RATE       = "llm_error_rate"         # LLM calls returning errors
    ANOMALY_DETECTED     = "anomaly_detected"       # generic anomaly signal


@dataclass
class HealthSignal:
    """One health observation emitted during execution."""
    timestamp: float = field(default_factory=time.time)
    agent_id: str = ""
    signal: SleepReason = SleepReason.ANOMALY_DETECTED
    detail: str = ""
    severity: int = 1   # 1=warn, 2=error, 3=critical


@dataclass
class PipelineHealth:
    """Aggregated health state for one agent run."""
    sleep_mode: bool = False
    sleep_reason: Optional[SleepReason] = None
    signals: list[HealthSignal] = field(default_factory=list)
    consecutive_tool_errors: int = 0
    parse_failures: int = 0
    total_llm_errors: int = 0

    def record(self, signal: HealthSignal) -> None:
        self.signals.append(signal)

    def should_sleep(self) -> tuple[bool, Optional[SleepReason]]:
        """Evaluate whether conditions warrant sleep mode."""
        if self.parse_failures >= 3:
            return True, SleepReason.LLM_PARSE_FAILURES
        if self.consecutive_tool_errors >= 4:
            return True, SleepReason.CONSECUTIVE_ERRORS
        if self.total_llm_errors >= 3:
            return True, SleepReason.LLM_ERROR_RATE
        return False, None

    def summary(self) -> str:
        if self.sleep_mode:
            return f"⚠ SLEEP({self.sleep_reason.value}): {len(self.signals)} signals"
        return f"Healthy: {len(self.signals)} signals, {self.consecutive_tool_errors} consecutive errors"


# ─── SubAgent inter-agent signalling ─────────────────────────────────────────

class SubAgentSignal(str, Enum):
    """Status signal a child SubAgent sends back to its parent."""
    PENDING   = "pending"   # not yet started
    RUNNING   = "running"   # in progress
    DONE      = "done"      # completed successfully
    FAILED    = "failed"    # completed with error
    SLEEPING  = "sleeping"  # entered sleep mode, needs intervention


# ─── StepTrace ────────────────────────────────────────────────────────────────

@dataclass
class StepTrace:
    """Records one Observe→Think→Act cycle in an agent loop.

    Now includes:
      - tool_result_raw: full untruncated result for replay
      - health_flag: True if this step contributed a health signal
      - child_signal: signal from a child subagent if spawned this step
    """
    step_number: int
    agent_id: str
    task_type: TaskType
    model_used: str
    thought: str
    tool_name: Optional[str]
    tool_args: dict[str, Any]
    tool_result: Optional[str]          # truncated for display
    tool_result_raw: Optional[str]      # full result for replay
    tool_error: bool
    latency_ms: float
    timestamp: float = field(default_factory=time.time)
    health_flag: bool = False
    child_signal: Optional[SubAgentSignal] = None   # set when a child was spawned

    def summary(self) -> str:
        status = "❌" if self.tool_error else "✓"
        health = " ⚠" if self.health_flag else ""
        tool_part = (
            f" → {self.tool_name}({', '.join(f'{k}={v!r}' for k, v in list(self.tool_args.items())[:2])})"
            if self.tool_name else " → (no tool)"
        )
        return f"[{self.agent_id}] step {self.step_number}{tool_part} {status}{health}"


# ─── SubTask ──────────────────────────────────────────────────────────────────

@dataclass
class SubTask:
    """One unit of work in the TaskPlan DAG."""
    id: str
    description: str
    task_type: TaskType
    dependencies: list[str]
    spawn_subagent: bool
    context_hint: str = ""
    max_steps: int = 10

    # Set during execution
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Optional[str] = None
    error: Optional[str] = None
    traces: list[StepTrace] = field(default_factory=list)
    agent_id: Optional[str] = None
    signal: SubAgentSignal = SubAgentSignal.PENDING   # live status for parent
    health: PipelineHealth = field(default_factory=PipelineHealth)


# ─── TaskPlan ─────────────────────────────────────────────────────────────────

@dataclass
class TaskPlan:
    """The Planner's output: a DAG of subtasks + execution mode decision."""
    mode: AgentMode
    subtasks: list[SubTask]
    original_query: str
    reasoning: str

    def get(self, task_id: str) -> Optional[SubTask]:
        return next((t for t in self.subtasks if t.id == task_id), None)

    def ready_tasks(self, completed_ids: set[str]) -> list[SubTask]:
        return [
            t for t in self.subtasks
            if t.status == SubTaskStatus.PENDING
            and t.id not in completed_ids
            and all(dep in completed_ids for dep in t.dependencies)
        ]

    def is_complete(self) -> bool:
        return all(
            t.status in (SubTaskStatus.DONE, SubTaskStatus.FAILED, SubTaskStatus.SKIPPED)
            for t in self.subtasks
        )

    def summary(self) -> str:
        done = sum(1 for t in self.subtasks if t.status == SubTaskStatus.DONE)
        return f"TaskPlan({self.mode.value}, {len(self.subtasks)} tasks, {done} done)"


# ─── AgentResult ─────────────────────────────────────────────────────────────

@dataclass
class AgentResult:
    """The final output of one complete agent turn."""
    response: str
    mode: AgentMode
    plan: Optional[TaskPlan]
    all_traces: list[StepTrace]
    files_modified: list[str]
    commands_run: list[str]
    success: bool
    error: Optional[str] = None
    total_steps: int = 0
    total_latency_ms: float = 0.0
    sleep_mode: bool = False
    health_report: Optional[PipelineHealth] = None

    def step_summary(self) -> str:
        lines = [
            f"Mode: {self.mode.value} | Steps: {self.total_steps} | "
            f"{self.total_latency_ms:.0f}ms"
            + (" | ⚠ SLEEP MODE" if self.sleep_mode else "")
        ]
        for trace in self.all_traces:
            lines.append(f"  {trace.summary()}")
        return "\n".join(lines)