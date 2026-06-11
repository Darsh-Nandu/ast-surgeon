"""
SessionStore — persistent storage for one chat session.

WHY per-session JSON files (not a single DB):
  Each session is small (< 1 MB of chat history), human-readable, and
  trivially diffable in git. A single SQLite DB would add a dependency and
  require locking. JSON files have no concurrent-write risk because each
  sovereign chat process owns one session.

FILE FORMAT (sessions/<id>.json):
  {
    "session_id":   "ses-abc123",
    "project_root": "/absolute/path/to/project",
    "created_at":   1718000000.0,
    "last_active":  1718005000.0,
    "title":        "Refactor auth module",   // auto-set from first message
    "history": [
      {"role": "user",      "content": "..."},
      {"role": "assistant", "content": "..."},
      ...
    ],
    "files_modified": ["src/auth.py", ...],
    "stats": {
      "total_turns": 12,
      "total_tokens_approx": 14200
    }
  }

DESIGN NOTE on history truncation:
  We store the full history in the file (up to MAX_HISTORY_TURNS=100 turns).
  AgentLoop only *sends* the last N turns to the LLM (its own rolling window),
  but we keep the full record so /history can show the entire conversation and
  future summarisation can use it.
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Optional

MAX_HISTORY_TURNS = 100   # max turns stored on disk (200 messages)
_SESSION_DIR_NAME = "sessions"


def _new_session_id() -> str:
    return "ses-" + uuid.uuid4().hex[:10]


class SessionStore:
    """Manages persistent state for one chat session.

    Usage:
        store = SessionStore.create(sovereign_dir=Path(".sovereign"))
        # or resume:
        store = SessionStore.load(sovereign_dir, session_id="ses-abc123")

        store.append_turn("user", "How does auth work?")
        store.append_turn("assistant", "The auth module lives in src/auth.py ...")
        store.save()
    """

    def __init__(
        self,
        sovereign_dir: Path,
        session_id: str,
        project_root: str,
        created_at: float,
        last_active: float,
        title: str,
        history: list[dict],
        files_modified: list[str],
        stats: dict,
    ):
        self._dir = sovereign_dir
        self.session_id = session_id
        self.project_root = project_root
        self.created_at = created_at
        self.last_active = last_active
        self.title = title
        self.history: list[dict] = history
        self.files_modified: list[str] = files_modified
        self.stats: dict = stats

    # ── Factories ────────────────────────────────────────────────────────────

    @classmethod
    def create(cls, sovereign_dir: Path, project_root: str) -> "SessionStore":
        """Create a brand-new session and save it immediately."""
        now = time.time()
        store = cls(
            sovereign_dir=sovereign_dir,
            session_id=_new_session_id(),
            project_root=project_root,
            created_at=now,
            last_active=now,
            title="",
            history=[],
            files_modified=[],
            stats={"total_turns": 0, "total_tokens_approx": 0},
        )
        store.save()
        return store

    @classmethod
    def load(cls, sovereign_dir: Path, session_id: str) -> "SessionStore":
        """Load an existing session from disk. Raises FileNotFoundError if missing."""
        path = cls._session_path(sovereign_dir, session_id)
        data = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            sovereign_dir=sovereign_dir,
            session_id=data["session_id"],
            project_root=data["project_root"],
            created_at=data["created_at"],
            last_active=data["last_active"],
            title=data.get("title", ""),
            history=data.get("history", []),
            files_modified=data.get("files_modified", []),
            stats=data.get("stats", {"total_turns": 0, "total_tokens_approx": 0}),
        )

    @classmethod
    def exists(cls, sovereign_dir: Path, session_id: str) -> bool:
        return cls._session_path(sovereign_dir, session_id).exists()

    @classmethod
    def list_sessions(cls, sovereign_dir: Path) -> list[dict]:
        """Return lightweight metadata dicts for all sessions, newest first."""
        sessions_dir = sovereign_dir / _SESSION_DIR_NAME
        if not sessions_dir.exists():
            return []
        results = []
        for path in sorted(sessions_dir.glob("ses-*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append({
                    "session_id": data["session_id"],
                    "title": data.get("title", ""),
                    "last_active": data.get("last_active", 0),
                    "total_turns": data.get("stats", {}).get("total_turns", 0),
                    "project_root": data.get("project_root", ""),
                })
            except Exception:
                pass
        return results

    # ── Mutation ─────────────────────────────────────────────────────────────

    def append_turn(self, role: str, content: str) -> None:
        """Append one message and update stats. Call save() afterwards."""
        self.history.append({"role": role, "content": content})
        if role == "user":
            self.stats["total_turns"] = self.stats.get("total_turns", 0) + 1
            # Auto-title from the first user message
            if not self.title and content:
                self.title = content[:60].strip()
        self.stats["total_tokens_approx"] = self.stats.get("total_tokens_approx", 0) + len(content) // 4

        # Trim history if needed (keep most recent turns)
        max_msgs = MAX_HISTORY_TURNS * 2
        if len(self.history) > max_msgs:
            self.history = self.history[-max_msgs:]

    def record_files_modified(self, paths: list[str]) -> None:
        """Add to the cumulative list of files this session touched."""
        for p in paths:
            if p not in self.files_modified:
                self.files_modified.append(p)

    def save(self) -> None:
        """Persist the session to disk (atomic write via temp file)."""
        self.last_active = time.time()
        sessions_dir = self._dir / _SESSION_DIR_NAME
        sessions_dir.mkdir(parents=True, exist_ok=True)

        path = self._session_path(self._dir, self.session_id)
        data = {
            "session_id": self.session_id,
            "project_root": self.project_root,
            "created_at": self.created_at,
            "last_active": self.last_active,
            "title": self.title,
            "history": self.history,
            "files_modified": self.files_modified,
            "stats": self.stats,
        }
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @property
    def recent_history(self) -> list[dict]:
        """Return the last 40 messages (20 turns) for sending to the LLM."""
        return self.history[-40:]

    def summary_line(self) -> str:
        import datetime
        dt = datetime.datetime.fromtimestamp(self.last_active).strftime("%Y-%m-%d %H:%M")
        title = self.title or "(no title)"
        turns = self.stats.get("total_turns", 0)
        return f"{self.session_id}  [{dt}]  {turns} turns  — {title}"

    @staticmethod
    def _session_path(sovereign_dir: Path, session_id: str) -> Path:
        return sovereign_dir / _SESSION_DIR_NAME / f"{session_id}.json"