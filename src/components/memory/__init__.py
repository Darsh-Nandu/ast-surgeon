"""
Persistent session memory for Sovereign-Code.

Sessions are scoped to a project root (resolved absolute path).
Each project can have many sessions; the active one is tracked in
.sovereign/active_session.

Storage layout under .sovereign/:
  sessions/
    <session_id>.json   ← one file per session
  active_session        ← plain text file: "ses-<id>"
"""
from .session_store import SessionStore
from .session_manager import SessionManager

__all__ = ["SessionStore", "SessionManager"]