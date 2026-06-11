"""
SessionManager — per-session lifecycle: create, resume, list, delete.

DESIGN:
  Each session is assigned a UUID on creation. Two things are scoped to a
  session:

  1. CONVERSATION HISTORY (on disk)
     Saved to .sovereign/sessions/<session_id>/history.json after every
     agent turn. Loaded back when --resume <session_id> is used.

  2. QDRANT COLLECTION (isolated vectors)
     Collection name: sovereign_<session_id[:8]>
     Every session's codebase index lives in its own Qdrant collection.
     Sessions never share vectors — one session's indexed files don't bleed
     into another session's retrieval results.

  Session metadata (id, project root, created_at, collection name, last
  active, turn count) is stored in .sovereign/sessions/<id>/meta.json.

WHY per-session collections:
  The agent indexes code during operation (new files written, files edited).
  Without isolation, session A's half-written refactor would pollute session
  B's search results. Isolation means each session sees only what IT indexed.

COLLECTION LIFECYCLE:
  - Created lazily on first vector operation in the session.
  - NOT auto-deleted on session end (user may resume).
  - Deleted explicitly via SessionManager.delete(session_id).

RESUMING:
  SessionManager.resume(session_id) returns a SessionInfo whose history and
  collection_name are already populated. AgentLoop.create() accepts the
  collection_name so Qdrant uses the right namespace.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional


# ─── Data model ───────────────────────────────────────────────────────────────

@dataclass
class SessionInfo:
    """Metadata + conversation history for one session."""
    session_id: str
    project_root: str
    collection_name: str          # Qdrant collection: sovereign_<id[:8]>
    created_at: float
    last_active: float
    turn_count: int = 0
    history: list[dict] = field(default_factory=list)   # [{role, content}, ...]
    files_modified: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)

    @property
    def session_dir(self) -> Path:
        return (
            Path(self.project_root)
            / ".sovereign"
            / "sessions"
            / self.session_id
        )

    def age_str(self) -> str:
        delta = time.time() - self.last_active
        if delta < 60:
            return f"{int(delta)}s ago"
        if delta < 3600:
            return f"{int(delta/60)}m ago"
        if delta < 86400:
            return f"{int(delta/3600)}h ago"
        return f"{int(delta/86400)}d ago"


# ─── Manager ──────────────────────────────────────────────────────────────────

class SessionManager:
    """
    Create, load, list, and delete sessions for a project.

    Usage:
        sm = SessionManager(project_root="/path/to/project")

        # New session
        info = sm.create()

        # Resume existing
        info = sm.resume("abc12345-...")

        # After agent turn: persist updated history
        sm.save(info)

        # List all sessions
        for s in sm.list_sessions():
            print(s.session_id, s.turn_count, s.age_str())

        # Delete (also tells caller to drop the Qdrant collection)
        sm.delete("abc12345-...")
    """

    def __init__(self, project_root: str | Path):
        self._root = Path(project_root).resolve()
        self._sessions_dir = self._root / ".sovereign" / "sessions"
        self._sessions_dir.mkdir(parents=True, exist_ok=True)

    # ─── Create ───────────────────────────────────────────────────────────────

    def create(self) -> SessionInfo:
        """Create a fresh session with a new UUID and empty history."""
        sid = str(uuid.uuid4())
        collection = f"sovereign_{sid[:8]}"
        now = time.time()

        info = SessionInfo(
            session_id=sid,
            project_root=str(self._root),
            collection_name=collection,
            created_at=now,
            last_active=now,
        )

        info.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_meta(info)
        return info

    # ─── Resume ───────────────────────────────────────────────────────────────

    def resume(self, session_id: str) -> SessionInfo:
        """Load an existing session by ID.

        Raises:
            FileNotFoundError: if session_id doesn't exist.
        """
        meta_path = self._sessions_dir / session_id / "meta.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                f"Session {session_id!r} not found in {self._sessions_dir}"
            )

        meta = json.loads(meta_path.read_text())
        history = self._load_history(session_id)

        return SessionInfo(
            session_id=meta["session_id"],
            project_root=meta["project_root"],
            collection_name=meta["collection_name"],
            created_at=meta["created_at"],
            last_active=meta["last_active"],
            turn_count=meta.get("turn_count", 0),
            history=history,
            files_modified=meta.get("files_modified", []),
            commands_run=meta.get("commands_run", []),
        )

    # ─── Save ─────────────────────────────────────────────────────────────────

    def save(self, info: SessionInfo) -> None:
        """Persist session metadata + conversation history to disk.

        Call this after every agent turn so sessions survive crashes/exits.
        """
        info.last_active = time.time()
        info.session_dir.mkdir(parents=True, exist_ok=True)
        self._write_meta(info)
        self._write_history(info)

    # ─── List ─────────────────────────────────────────────────────────────────

    def list_sessions(self) -> list[SessionInfo]:
        """Return all sessions for this project, newest first."""
        sessions = []
        for d in self._sessions_dir.iterdir():
            meta_path = d / "meta.json"
            if not meta_path.exists():
                continue
            try:
                meta = json.loads(meta_path.read_text())
                sessions.append(SessionInfo(
                    session_id=meta["session_id"],
                    project_root=meta["project_root"],
                    collection_name=meta["collection_name"],
                    created_at=meta["created_at"],
                    last_active=meta["last_active"],
                    turn_count=meta.get("turn_count", 0),
                    history=[],   # don't load history for listing
                    files_modified=meta.get("files_modified", []),
                    commands_run=meta.get("commands_run", []),
                ))
            except Exception:
                continue
        return sorted(sessions, key=lambda s: s.last_active, reverse=True)

    # ─── Delete ───────────────────────────────────────────────────────────────

    def delete(
        self,
        session_id: str,
        drop_qdrant_collection: bool = True,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
    ) -> None:
        """Delete session files and optionally its Qdrant collection."""
        session_dir = self._sessions_dir / session_id
        if not session_dir.exists():
            raise FileNotFoundError(f"Session {session_id!r} not found")

        # Read collection name before deleting meta
        meta_path = session_dir / "meta.json"
        collection_name = None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            collection_name = meta.get("collection_name")

        # Remove session directory
        import shutil
        shutil.rmtree(session_dir)

        # Drop Qdrant collection
        if drop_qdrant_collection and collection_name:
            try:
                from qdrant_client import QdrantClient
                client = QdrantClient(host=qdrant_host, port=qdrant_port)
                existing = {c.name for c in client.get_collections().collections}
                if collection_name in existing:
                    client.delete_collection(collection_name)
            except Exception:
                pass   # non-fatal — Qdrant may not be running

    # ─── Helpers ──────────────────────────────────────────────────────────────

    def _write_meta(self, info: SessionInfo) -> None:
        meta = {
            "session_id":      info.session_id,
            "project_root":    info.project_root,
            "collection_name": info.collection_name,
            "created_at":      info.created_at,
            "last_active":     info.last_active,
            "turn_count":      info.turn_count,
            "files_modified":  info.files_modified,
            "commands_run":    info.commands_run,
        }
        (info.session_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def _write_history(self, info: SessionInfo) -> None:
        (info.session_dir / "history.json").write_text(
            json.dumps(info.history, indent=2)
        )

    def _load_history(self, session_id: str) -> list[dict]:
        history_path = self._sessions_dir / session_id / "history.json"
        if not history_path.exists():
            return []
        try:
            return json.loads(history_path.read_text())
        except Exception:
            return []