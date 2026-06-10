"""
Agent data models — the shared language between every layer of the agent system.

WHY a rich model layer:
  Every component (Planner, SubAgent, Orchestrator, Synthesiser) communicates
  through these typed structures. No passing raw dicts between layers.
  This means:
  - The eval harness (Phase 5) can replay any session from its StepTrace
  - The CLI can render live progress from AgentResult.traces
  - Tests can assert on specific fields without parsing LLM output strings

DESIGN NOTE on TaskPlan as a DAG:
  SubTask.dependencies is a list of other subtask IDs. The Orchestrator
  resolves execution order by topological sort. Tasks with empty dependencies
  run immediately and can be parallelised. This is how tools like Airflow and
  GitHub Actions model pipelines — proven pattern for task graphs.

DESIGN NOTE on StepTrace:
  Every action every agent takes is recorded here — tool name, args, result,
  latency, model used. This is the audit trail for debugging and evals.
  A senior engineer at Anthropic would flag the absence of this immediately:
  without a trace you cannot reproduce failures or measure agent quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional
import time


# Enums

class TaskType(str, Enum):
    """What kind of work a task requires — used by the Router to pick a model."""
    PLANNING     = "planning"       # break down a problem, reason about architecture
    CODE_GEN     = "code_gen"       # write new code
    CODE_EDIT    = "code_edit"      # modify existing code
    CODE_REVIEW  = "code_review"    # read and explain code
    DEBUG        = "debug"          # find and fix bugs
    TEST_WRITE   = "test_write"     # write tests
    SEARCH       = "search"         # retrieve / explore codebase
    RUN          = "run"            # execute commands / tests
    EXPLAIN      = "explain"        # answer questions about code
    SYNTHESISE   = "synthesise"     # merge / summarise multiple results


class SubTaskStatus(str, Enum):
    PENDING    = "pending"
    RUNNING    = "running"
    DONE       = "done"
    FAILED     = "failed"
    SKIPPED    = "skipped"   # dependency failed, so this was skipped


class AgentMode(str, Enum):
    DIRECT   = "direct"    # single agent loop, no subagents
    PARALLEL = "parallel"  # orchestrator spawns subagents


# Step trace - one action in an agent's loop

@dataclass
class StepTrace:
    """Records one Observe→Think→Act cycle in an agent loop.

    DESIGN NOTE: we record latency_ms so the eval harness can flag slow steps
    and we can tune which model to use for which task type.
    """
    step_number: int
    agent_id: str                    # which agent took this step
    task_type: TaskType
    model_used: str                  # e.g. "llama-3.3-70b-versatile"
    thought: str                     # LLM's reasoning before acting
    tool_name: Optional[str]         # None if LLM decided no tool needed
    tool_args: dict[str, Any]        # args passed to the tool
    tool_result: Optional[str]       # ToolResult.content (truncated if huge)
    tool_error: bool                 # ToolResult.is_error
    latency_ms: float
    timestamp: float = field(default_factory=time.time)

    def summary(self) -> str:
        status = "❌" if self.tool_error else "✓"
        tool_part = f" → {self.tool_name}({', '.join(f'{k}={v!r}' for k,v in list(self.tool_args.items())[:2])})" if self.tool_name else " → (no tool)"
        return f"[{self.agent_id}] step {self.step_number}{tool_part} {status}"


# SubTask - one node in the TaskPlan DAG

@dataclass
class SubTask:
    """One unit of work in the TaskPlan.

    Can be executed by a SubAgent (if complex) or directly by the Orchestrator
    (if simple). The Planner decides which by setting spawn_subagent=True.

    DESIGN NOTE on dependencies:
      dependencies is a list of subtask IDs that must complete before this
      one starts. Empty list = can run immediately (and in parallel with other
      zero-dependency tasks). This is the core of the DAG execution model.
    """
    id: str                          # e.g. "task_0", "task_1"
    description: str                 # what this subtask should do
    task_type: TaskType
    dependencies: list[str]          # IDs of subtasks that must complete first
    spawn_subagent: bool             # True = needs own SubAgent loop
    context_hint: str = ""           # hint for retrieval (e.g. "AuthService login")
    max_steps: int = 10              # hard ceiling on agent loop iterations

    # Set during execution
    status: SubTaskStatus = SubTaskStatus.PENDING
    result: Optional[str] = None     # final output of this subtask
    error: Optional[str] = None
    traces: list[StepTrace] = field(default_factory=list)
    agent_id: Optional[str] = None   # which agent ran this


# TaskPlan - the full DAG returned by the Planner

@dataclass
class TaskPlan:
    """The Planner's output: a DAG of subtasks + execution mode decision.

    DESIGN NOTE on mode selection:
      The Planner sets mode=PARALLEL only when there are 2+ subtasks with
      no dependency between them AND each is complex enough to warrant its
      own agent loop. For simple tasks, mode=DIRECT and subtasks has one entry.
    """
    mode: AgentMode
    subtasks: list[SubTask]
    original_query: str
    reasoning: str                   # why the planner chose this structure

    def get(self, task_id: str) -> Optional[SubTask]:
        return next((t for t in self.subtasks if t.id == task_id), None)

    def ready_tasks(self, completed_ids: set[str]) -> list[SubTask]:
        """Return PENDING tasks whose dependencies are all in completed_ids."""
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


# AgentResult - what the full loop returns to the session

@dataclass
class AgentResult:
    """The final output of one complete agent turn.

    This is what session.py receives and renders to the user.
    The full traces are available for the eval harness.
    """
    response: str                    # final answer / summary to show the user
    mode: AgentMode
    plan: Optional[TaskPlan]         # None for trivial single-step responses
    all_traces: list[StepTrace]      # every step from every agent
    files_modified: list[str]        # paths written/edited this turn
    commands_run: list[str]          # commands executed this turn
    success: bool
    error: Optional[str] = None
    total_steps: int = 0
    total_latency_ms: float = 0.0

    def step_summary(self) -> str:
        lines = [f"Mode: {self.mode.value} | Steps: {self.total_steps} | {self.total_latency_ms:.0f}ms"]
        for trace in self.all_traces:
            lines.append(f"  {trace.summary()}")
        return "\n".join(lines)