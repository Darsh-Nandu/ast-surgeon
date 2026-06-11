"""
SessionManager — session lifecycle for a specific project.

RESPONSIBILITIES:
  - Create a new session when `sovereign init` is run in a directory.
  - Resume the active session when `sovereign chat` is run.
  - List all sessions for a project.
  - Switch active session.

ACTIVE SESSION TRACKING:
  .sovereign/active_session  is a plain text file containing just the session ID
  (e.g. "ses-abc123de12"). On `init`, a new ID is written here. On `chat`, this
  file is read. If it's missing or the session file is gone, a new session is
  created automatically.

WHY track active session in a separate file (not config.json):
  config.json is for infrastructure config (Qdrant host, embedding provider).
  Session lifecycle is a separate concern. Keeping them separate means `init`
  can re-index without wiping conversation history.

SESSION ISOLATION BY FOLDER:
  Each unique absolute project_root gets its own `.sovereign/` directory (same
  pattern as git). So /home/user/projectA and /home/user/projectB each have
  their own sessions. Running `sovereign init` in a different folder creates
  a new session there automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .session_store import SessionStore

_ACTIVE_SESSION_FILE = "active_session"


class SessionManager:
    """Manages session lifecycle for one project (one `.sovereign/` directory).

    Usage:
        mgr = SessionManager(Path("/my/project/.sovereign"), project_root="/my/project")

        # On init:
        session = mgr.new_session()

        # On chat:
        session = mgr.get_or_create_session()

        # List past sessions:
        for meta in mgr.list_sessions():
            print(meta["session_id"], meta["title"])
    """

    def __init__(self, sovereign_dir: Path, project_root: str):
        self._dir = sovereign_dir
        self._project_root = project_root

    # ── Public API ────────────────────────────────────────────────────────────

    def new_session(self) -> SessionStore:
        """Create a fresh session and mark it as active."""
        session = SessionStore.create(
            sovereign_dir=self._dir,
            project_root=self._project_root,
        )
        self._set_active(session.session_id)
        return session

    def get_or_create_session(self) -> tuple[SessionStore, bool]:
        """Return (session, is_new).

        Tries to resume the active session. Falls back to creating a new one
        if the active session file is missing or its JSON is gone/corrupt.
        """
        active_id = self._get_active_id()
        if active_id and SessionStore.exists(self._dir, active_id):
            try:
                session = SessionStore.load(self._dir, active_id)
                return session, False
            except Exception:
                pass  # corrupt file → create fresh
        # No valid active session → create one
        session = self.new_session()
        return session, True

    def resume_session(self, session_id: str) -> SessionStore:
        """Load a specific session and make it the active session."""
        session = SessionStore.load(self._dir, session_id)  # raises if missing
        self._set_active(session_id)
        return session

    def list_sessions(self) -> list[dict]:
        """Return metadata for all sessions, newest first."""
        return SessionStore.list_sessions(self._dir)

    def active_session_id(self) -> Optional[str]:
        return self._get_active_id()

    # ── Internal ─────────────────────────────────────────────────────────────

    def _get_active_id(self) -> Optional[str]:
        path = self._dir / _ACTIVE_SESSION_FILE
        if not path.exists():
            return None
        raw = path.read_text(encoding="utf-8").strip()
        return raw if raw else None

    def _set_active(self, session_id: str) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / _ACTIVE_SESSION_FILE).write_text(session_id, encoding="utf-8")