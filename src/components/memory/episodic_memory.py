"""
Layer 2: Session Episodic Memory (disk, whole session)

This is the agent's medium-term memory for an entire session. It persists:
  - Project facts: language, framework, key file locations, conventions
  - File registry: every file the system has read or written, with metadata
  - Turn summaries: compressed summaries of each completed turn
  - Failed approaches: what was tried and why it didn't work
  - Planner hints: patterns the planner should be aware of next time

LIFETIME: Created at session start, persisted after every turn, loaded on --resume.

INJECTION: Serialised into the Planner's user prompt so the Planner has full
           project context without needing to re-read everything from disk.

DISK FORMAT: .sovereign/sessions/<id>/episodic.json

WHY disk (not RAM):
  Sessions can span hours and dozens of turns. Reloading the codebase from
  Qdrant on every turn is expensive and lossy. Episodic memory is the cheap
  structured complement — facts and summaries rather than full content.

FUTURE (checker integration):
  When the CheckerAgent reports a failure, the orchestrator calls
  episodic.record_failed_approach() with the error details. The Planner
  reads this on the next turn and avoids the same approach.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── Data models ─────────────────────────────────────────────────────────────

@dataclass
class ProjectFact:
    """A factual discovery about the project."""
    key: str              # e.g. "primary_language", "test_framework"
    value: str            # e.g. "python", "pytest"
    confidence: float = 1.0
    source: str = ""      # e.g. "detected from pyproject.toml"
    timestamp: float = field(default_factory=time.time)


@dataclass
class FileRecord:
    """Metadata about a file the agent has interacted with."""
    path: str
    last_read: Optional[float] = None
    last_written: Optional[float] = None
    purpose: str = ""       # human-readable role, e.g. "main auth module"
    key_symbols: list[str] = field(default_factory=list)  # top-level classes/functions
    write_count: int = 0


@dataclass
class TurnSummary:
    """Compressed record of one completed agent turn."""
    turn_number: int
    query_snippet: str        # first 80 chars of the user query
    outcome: str              # "success" | "partial" | "failed" | "sleeping"
    files_modified: list[str]
    commands_run: list[str]
    key_finding: str          # most important thing that happened
    timestamp: float = field(default_factory=time.time)


@dataclass
class FailedApproach:
    """Something the agent tried that didn't work."""
    turn_number: int
    description: str           # what was attempted
    reason: str                # why it failed (error message or explanation)
    files_involved: list[str]
    do_not_repeat: bool = True  # hint for Planner
    timestamp: float = field(default_factory=time.time)


# ─── EpisodicMemory ───────────────────────────────────────────────────────────

class EpisodicMemory:
    """
    Session-scoped episodic memory persisted to disk as JSON.

    Usage in AgentLoop:
        em = EpisodicMemory.load_or_create(session_dir)

        # Before planning:
        planner_context = em.to_planner_context()

        # After a turn:
        em.record_turn(turn_number, query, result)
        em.save()

        # When a file is read/written by an agent:
        em.update_file_record(path, written=True, purpose="auth module")

        # When something fails:
        em.record_failed_approach(turn, "Tried rewriting X", "Import error", files)
    """

    def __init__(self, session_dir: Path):
        self._path = session_dir / "episodic.json"
        self.session_dir = session_dir

        self.project_facts: dict[str, ProjectFact] = {}
        self.file_registry: dict[str, FileRecord] = {}
        self.turn_summaries: list[TurnSummary] = []
        self.failed_approaches: list[FailedApproach] = []
        self.planner_hints: list[str] = []   # freeform hints for the Planner
        self.created_at: float = time.time()
        self.last_updated: float = time.time()

    # ─── Factories ────────────────────────────────────────────────────────────

    @classmethod
    def load_or_create(cls, session_dir: Path) -> "EpisodicMemory":
        """Load from disk if exists, otherwise create empty."""
        em = cls(session_dir)
        path = session_dir / "episodic.json"
        if path.exists():
            try:
                em._load(path)
            except Exception:
                pass   # corrupt file → start fresh
        return em

    # ─── Recording ────────────────────────────────────────────────────────────

    def set_fact(self, key: str, value: str, confidence: float = 1.0, source: str = "") -> None:
        """Record or update a project fact (e.g. language, framework, entry point)."""
        self.project_facts[key] = ProjectFact(
            key=key, value=value, confidence=confidence, source=source
        )

    def update_file_record(
        self,
        path: str,
        read: bool = False,
        written: bool = False,
        purpose: str = "",
        key_symbols: Optional[list[str]] = None,
    ) -> None:
        """Update the file registry after an agent reads or writes a file."""
        rec = self.file_registry.get(path) or FileRecord(path=path)
        now = time.time()
        if read:
            rec.last_read = now
        if written:
            rec.last_written = now
            rec.write_count += 1
        if purpose:
            rec.purpose = purpose
        if key_symbols:
            rec.key_symbols = key_symbols[:20]  # cap to keep memory bounded
        self.file_registry[path] = rec

    def record_turn(
        self,
        turn_number: int,
        query: str,
        outcome: str,
        files_modified: list[str],
        commands_run: list[str],
        key_finding: str,
    ) -> None:
        """Summarise and store one completed agent turn."""
        summary = TurnSummary(
            turn_number=turn_number,
            query_snippet=query[:80],
            outcome=outcome,
            files_modified=files_modified,
            commands_run=commands_run[:10],
            key_finding=key_finding[:200],
        )
        self.turn_summaries.append(summary)
        # Keep only the last 30 turn summaries to avoid unbounded growth
        if len(self.turn_summaries) > 30:
            self.turn_summaries = self.turn_summaries[-30:]

        # Auto-update file registry for anything touched this turn
        for path in files_modified:
            self.update_file_record(path, written=True)

    def record_failed_approach(
        self,
        turn_number: int,
        description: str,
        reason: str,
        files_involved: list[str],
        do_not_repeat: bool = True,
    ) -> None:
        """Record something that was tried and failed. Planner reads these."""
        self.failed_approaches.append(FailedApproach(
            turn_number=turn_number,
            description=description,
            reason=reason,
            files_involved=files_involved,
            do_not_repeat=do_not_repeat,
        ))
        # Keep last 20 failures
        if len(self.failed_approaches) > 20:
            self.failed_approaches = self.failed_approaches[-20:]

    def add_planner_hint(self, hint: str) -> None:
        """Add a freeform hint for the Planner (e.g. 'Always run mypy after edits')."""
        if hint not in self.planner_hints:
            self.planner_hints.append(hint)
        if len(self.planner_hints) > 15:
            self.planner_hints = self.planner_hints[-15:]

    # ─── Planner injection ────────────────────────────────────────────────────

    def to_planner_context(self) -> str:
        """
        Serialise episodic memory into a text block for injection into the
        Planner's prompt (Layer 2 injection).

        Structured to be concise but complete: Planner reads this once per turn.
        """
        lines = ["=== Session Memory (Episodic) ==="]

        # Project facts
        if self.project_facts:
            lines.append("\nProject facts:")
            for key, fact in self.project_facts.items():
                conf = f" (confidence: {fact.confidence:.0%})" if fact.confidence < 1.0 else ""
                lines.append(f"  {key}: {fact.value}{conf}")

        # File registry (most recently touched files)
        relevant_files = sorted(
            self.file_registry.values(),
            key=lambda r: max(r.last_read or 0, r.last_written or 0),
            reverse=True,
        )[:20]
        if relevant_files:
            lines.append(f"\nFile registry ({len(self.file_registry)} total, showing 20 most recent):")
            for rec in relevant_files:
                tags = []
                if rec.last_written:
                    tags.append(f"written×{rec.write_count}")
                if rec.purpose:
                    tags.append(rec.purpose)
                if rec.key_symbols:
                    tags.append(f"symbols: {', '.join(rec.key_symbols[:5])}")
                tag_str = f" [{', '.join(tags)}]" if tags else ""
                lines.append(f"  {rec.path}{tag_str}")

        # Recent turn summaries
        if self.turn_summaries:
            lines.append(f"\nRecent turns (last {min(5, len(self.turn_summaries))}):")
            for ts in self.turn_summaries[-5:]:
                lines.append(
                    f"  Turn {ts.turn_number} [{ts.outcome}]: {ts.query_snippet!r}"
                    + (f" → {ts.key_finding}" if ts.key_finding else "")
                )

        # Failed approaches — critical for Planner to avoid repeating mistakes
        do_not_repeat = [fa for fa in self.failed_approaches if fa.do_not_repeat]
        if do_not_repeat:
            lines.append(f"\n⚠ Failed approaches (do NOT repeat):")
            for fa in do_not_repeat[-5:]:
                lines.append(f"  • Turn {fa.turn_number}: {fa.description[:80]}")
                lines.append(f"    Reason: {fa.reason[:100]}")

        # Planner hints
        if self.planner_hints:
            lines.append(f"\nPlanner hints:")
            for hint in self.planner_hints:
                lines.append(f"  • {hint}")

        lines.append("=== End Session Memory ===")
        return "\n".join(lines)

    # ─── Persistence ──────────────────────────────────────────────────────────

    def save(self) -> None:
        """Atomic write to disk."""
        self.last_updated = time.time()
        self.session_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "created_at": self.created_at,
            "last_updated": self.last_updated,
            "project_facts": {k: asdict(v) for k, v in self.project_facts.items()},
            "file_registry": {k: asdict(v) for k, v in self.file_registry.items()},
            "turn_summaries": [asdict(ts) for ts in self.turn_summaries],
            "failed_approaches": [asdict(fa) for fa in self.failed_approaches],
            "planner_hints": self.planner_hints,
        }
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self, path: Path) -> None:
        """Deserialise from disk."""
        data = json.loads(path.read_text(encoding="utf-8"))
        self.created_at = data.get("created_at", time.time())
        self.last_updated = data.get("last_updated", time.time())

        self.project_facts = {
            k: ProjectFact(**v)
            for k, v in data.get("project_facts", {}).items()
        }
        self.file_registry = {
            k: FileRecord(**v)
            for k, v in data.get("file_registry", {}).items()
        }
        self.turn_summaries = [TurnSummary(**ts) for ts in data.get("turn_summaries", [])]
        self.failed_approaches = [FailedApproach(**fa) for fa in data.get("failed_approaches", [])]
        self.planner_hints = data.get("planner_hints", [])

    # ─── Utilities ────────────────────────────────────────────────────────────

    def detect_and_set_project_facts(self, project_root: Path) -> None:
        """
        Auto-detect common project facts from disk.
        Called once at session start; supplements Qdrant retrieval.
        """
        root = project_root

        # Language / runtime detection
        if (root / "pyproject.toml").exists() or (root / "setup.py").exists():
            self.set_fact("primary_language", "python", source="pyproject.toml/setup.py")
        elif (root / "package.json").exists():
            self.set_fact("primary_language", "javascript/typescript", source="package.json")
        elif (root / "go.mod").exists():
            self.set_fact("primary_language", "go", source="go.mod")
        elif (root / "Cargo.toml").exists():
            self.set_fact("primary_language", "rust", source="Cargo.toml")

        # Test framework
        if (root / "pytest.ini").exists() or (root / "pyproject.toml").exists():
            try:
                content = (root / "pyproject.toml").read_text() if (root / "pyproject.toml").exists() else ""
                if "pytest" in content:
                    self.set_fact("test_framework", "pytest", source="pyproject.toml")
            except Exception:
                pass
        if (root / "jest.config.js").exists() or (root / "jest.config.ts").exists():
            self.set_fact("test_framework", "jest", source="jest.config")

        # Entry points
        for candidate in ["src/main.py", "main.py", "app.py", "src/app.py", "index.ts", "index.js"]:
            if (root / candidate).exists():
                self.set_fact("entry_point", candidate, source="filesystem scan")
                break

        # Source layout
        if (root / "src").is_dir():
            self.set_fact("source_layout", "src/", source="filesystem scan")
        if (root / "tests").is_dir() or (root / "test").is_dir():
            self.set_fact("test_dir", "tests/" if (root / "tests").is_dir() else "test/",
                          source="filesystem scan")