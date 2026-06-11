"""
Layer 1: SubAgent Working Memory (RAM, one task lifetime)

This is the agent's short-term RAM for one SubTask execution. It tracks:
  - Files read and their content snapshots
  - Files written with final content
  - Commands run and their outputs
  - Errors encountered and how they were handled
  - Intermediate observations

LIFETIME: Created when a SubAgent starts, discarded when it finishes.
          The files_written dict is attached to the SubTask for Layer 2 to harvest.

INJECTION: Serialised into the SubAgent's initial message as a context block
           so the LLM always knows what it has already done in this task.

WHY RAM only (no disk):
  Each subtask is one focused unit of work. Its working memory is small enough
  to fit in the LLM context window. Writing it to disk would be overhead with
  no benefit — it lives only for the duration of one agent.run() call.

FUTURE (checker integration):
  files_written maps path → final_content. The CheckerAgent will pull this
  dict from the completed SubTask and run the code before reporting back.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class FileSnapshot:
    """Content of a file at a point in time."""
    path: str
    content: str
    timestamp: float = field(default_factory=time.time)
    operation: str = "read"   # "read" | "write" | "edit"


@dataclass
class CommandRecord:
    """One shell command and its output."""
    command: str
    output: str
    exit_ok: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ErrorRecord:
    """An error observed during the task."""
    step: int
    tool: str
    message: str
    resolved: bool = False
    resolution: str = ""
    timestamp: float = field(default_factory=time.time)


class WorkingMemory:
    """
    Ephemeral per-SubAgent memory. Tracks what the agent has seen and done.

    Usage in SubAgent:
        wm = WorkingMemory(task_description="Refactor auth.py")
        wm.record_file_read("src/auth.py", content)
        wm.record_file_write("src/auth.py", new_content)
        wm.record_command("python -m pytest", output, ok=True)
        wm.record_error(step=3, tool="run_command", message="SyntaxError: ...")
        wm.mark_error_resolved(step=3, resolution="Fixed missing colon")

        # Inject into LLM initial message:
        context_block = wm.to_context_block()

        # After task: attach to SubTask for Layer 2
        subtask.files_written = wm.files_written
        subtask.working_memory_summary = wm.summary()
    """

    def __init__(self, task_description: str, agent_id: str = ""):
        self.task_description = task_description
        self.agent_id = agent_id
        self.started_at = time.time()

        self._files_read: dict[str, FileSnapshot] = {}       # path → snapshot
        self._files_written: dict[str, FileSnapshot] = {}    # path → snapshot
        self._commands: list[CommandRecord] = []
        self._errors: list[ErrorRecord] = []
        self._observations: list[str] = []    # freeform notes from the agent

    # ─── Recording ────────────────────────────────────────────────────────────

    def record_file_read(self, path: str, content: str) -> None:
        self._files_read[path] = FileSnapshot(
            path=path, content=content, operation="read"
        )

    def record_file_write(self, path: str, content: str, operation: str = "write") -> None:
        self._files_written[path] = FileSnapshot(
            path=path, content=content, operation=operation
        )
        # Also update read cache so agent sees its own writes
        self._files_read[path] = FileSnapshot(
            path=path, content=content, operation=operation
        )

    def record_command(self, command: str, output: str, ok: bool) -> None:
        self._commands.append(CommandRecord(
            command=command, output=output, exit_ok=ok
        ))

    def record_error(self, step: int, tool: str, message: str) -> None:
        self._errors.append(ErrorRecord(
            step=step, tool=tool, message=message
        ))

    def mark_error_resolved(self, step: int, resolution: str) -> None:
        """Mark an error from a given step as resolved."""
        for err in self._errors:
            if err.step == step and not err.resolved:
                err.resolved = True
                err.resolution = resolution
                break

    def add_observation(self, note: str) -> None:
        """Record a freeform observation (e.g. 'tests are passing now')."""
        self._observations.append(note)

    # ─── Export ───────────────────────────────────────────────────────────────

    @property
    def files_written(self) -> dict[str, str]:
        """Return {path: content} for all files written this task.
        Attached to SubTask for CheckerAgent and Layer 2 harvesting."""
        return {p: snap.content for p, snap in self._files_written.items()}

    @property
    def files_read_paths(self) -> list[str]:
        return list(self._files_read.keys())

    @property
    def commands_run(self) -> list[str]:
        return [c.command for c in self._commands]

    @property
    def unresolved_errors(self) -> list[ErrorRecord]:
        return [e for e in self._errors if not e.resolved]

    def summary(self) -> str:
        """One-paragraph summary for attaching to SubTask."""
        elapsed = time.time() - self.started_at
        parts = [
            f"Task: {self.task_description[:80]}",
            f"Duration: {elapsed:.1f}s",
            f"Files read: {len(self._files_read)}",
            f"Files written: {len(self._files_written)} ({', '.join(self._files_written) or 'none'})",
            f"Commands run: {len(self._commands)}",
            f"Errors: {len(self._errors)} ({len(self.unresolved_errors)} unresolved)",
        ]
        return " | ".join(parts)

    def to_context_block(self) -> str:
        """
        Serialise working memory into a text block for injection into the
        SubAgent's initial LLM message (Layer 1 injection).

        Only includes entries that are actually useful to surface:
          - Files already read (agent doesn't need to re-read them)
          - Files already written (agent knows what it has produced)
          - Commands run and whether they succeeded
          - Unresolved errors (agent should know what went wrong so far)
        """
        if not (self._files_read or self._files_written or self._commands or self._errors):
            return ""

        lines = ["=== Working Memory (current task) ==="]

        if self._files_read:
            lines.append(f"\nFiles already read ({len(self._files_read)}):")
            for path in self._files_read:
                lines.append(f"  • {path}")

        if self._files_written:
            lines.append(f"\nFiles written this task:")
            for path, snap in self._files_written.items():
                lines.append(f"  • {path} ({snap.operation}, {len(snap.content)} chars)")

        if self._commands:
            lines.append(f"\nCommands run ({len(self._commands)}):")
            for cmd in self._commands[-5:]:   # last 5 to keep context bounded
                status = "✓" if cmd.exit_ok else "✗"
                lines.append(f"  {status} {cmd.command[:80]}")
                if not cmd.exit_ok and cmd.output:
                    lines.append(f"    Error: {cmd.output[:200]}")

        unresolved = self.unresolved_errors
        if unresolved:
            lines.append(f"\nUnresolved errors ({len(unresolved)}):")
            for err in unresolved:
                lines.append(f"  [step {err.step}] {err.tool}: {err.message[:120]}")

        if self._observations:
            lines.append(f"\nObservations:")
            for note in self._observations[-3:]:
                lines.append(f"  • {note}")

        lines.append("=== End Working Memory ===")
        return "\n".join(lines)