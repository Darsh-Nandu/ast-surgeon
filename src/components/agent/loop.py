"""
AgentLoop — top-level entry point that wires the full agent pipeline.

SESSIONS:
  Every AgentLoop is now scoped to a SessionInfo object.
  The session owns:
    - conversation history (persisted to .sovereign/sessions/<id>/history.json)
    - Qdrant collection (sovereign_<session_id[:8]>)

  Use AgentLoop.for_new_session() to start fresh.
  Use AgentLoop.for_session(session_id) to resume.
  Both save history to disk after every turn automatically.

DESIGN:
  Planner → Orchestrator → (SubAgents) → Synthesiser
  session.py calls loop.run(query) → AgentResult.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

from .models import AgentResult, AgentMode
from .planner import Planner
from .router import ModelRouter
from .orchestrator import Orchestrator
from .session_manager import SessionManager, SessionInfo

logger = logging.getLogger(__name__)


class AgentLoop:
    """Top-level agent pipeline scoped to one session.

    Usage — new session:
        loop = AgentLoop.for_new_session(project_root="/path/to/project")
        result = loop.run("Refactor the auth module")
        print(loop.session.session_id)   # save this to resume later

    Usage — resume session:
        loop = AgentLoop.for_session(
            session_id="abc12345-...",
            project_root="/path/to/project",
        )
        result = loop.run("Continue the refactor")
    """

    def __init__(
        self,
        project_root: str | Path,
        router: ModelRouter,
        session: SessionInfo,
        session_manager: SessionManager,
        vector_store=None,
        embed_pipeline=None,
        indexer=None,
    ):
        self._root = Path(project_root)
        self._router = router
        self._store = vector_store
        self._pipeline = embed_pipeline
        self._indexer = indexer
        self._session = session
        self._session_manager = session_manager

        self._planner = Planner(router)
        self._orchestrator = Orchestrator(
            router=router,
            project_root=project_root,
            vector_store=vector_store,
            embed_pipeline=embed_pipeline,
            indexer=indexer,
        )

    # ─── Factories ────────────────────────────────────────────────────────────

    @classmethod
    def for_new_session(
        cls,
        project_root: str | Path,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_provider: Optional[str] = None,
    ) -> "AgentLoop":
        """Create a fresh session and return a wired AgentLoop."""
        project_root = Path(project_root)
        sm = SessionManager(project_root)
        session = sm.create()
        logger.info("New session created: %s (collection=%s)", session.session_id, session.collection_name)
        return cls._build(
            project_root=project_root,
            session=session,
            session_manager=sm,
            groq_api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_provider=embedding_provider,
        )

    @classmethod
    def for_session(
        cls,
        session_id: str,
        project_root: str | Path,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_provider: Optional[str] = None,
    ) -> "AgentLoop":
        """Resume an existing session by ID."""
        project_root = Path(project_root)
        sm = SessionManager(project_root)
        session = sm.resume(session_id)
        logger.info(
            "Resuming session: %s (%d turns, collection=%s)",
            session.session_id, session.turn_count, session.collection_name,
        )
        return cls._build(
            project_root=project_root,
            session=session,
            session_manager=sm,
            groq_api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_provider=embedding_provider,
        )

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_provider: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> "AgentLoop":
        """
        Backward-compatible factory.

        Pass session_id to resume; omit for a new session.
        """
        if session_id:
            return cls.for_session(
                session_id=session_id,
                project_root=project_root,
                groq_api_key=groq_api_key,
                gemini_api_key=gemini_api_key,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                embedding_provider=embedding_provider,
            )
        return cls.for_new_session(
            project_root=project_root,
            groq_api_key=groq_api_key,
            gemini_api_key=gemini_api_key,
            qdrant_host=qdrant_host,
            qdrant_port=qdrant_port,
            embedding_provider=embedding_provider,
        )

    # ─── Internal builder ─────────────────────────────────────────────────────

    @classmethod
    def _build(
        cls,
        project_root: Path,
        session: SessionInfo,
        session_manager: SessionManager,
        groq_api_key: Optional[str],
        gemini_api_key: Optional[str],
        qdrant_host: str,
        qdrant_port: int,
        embedding_provider: Optional[str],
    ) -> "AgentLoop":
        router = ModelRouter(
            groq_api_key=groq_api_key or os.environ.get("GROQ_API_KEY"),
            gemini_api_key=(
                gemini_api_key
                or os.environ.get("GEMINI_API_KEY")
                or os.environ.get("GOOGLE_API_KEY")
            ),
        )

        # Per-session Qdrant collection — isolated from all other sessions
        vector_store = None
        embed_pipeline = None
        try:
            from ..vectorstore.qdrant_store import VectorStore
            from ..embeddings.providers import get_provider
            from ..embeddings.pipeline import EmbeddingPipeline

            vector_store = VectorStore.connect(
                host=qdrant_host,
                port=qdrant_port,
                collection=session.collection_name,  # <── session-scoped
            )
            vector_store.ensure_collection()

            provider = get_provider(embedding_provider)
            embed_pipeline = EmbeddingPipeline(provider=provider)

            logger.info(
                "AgentLoop: vector search enabled (collection=%s)",
                session.collection_name,
            )
        except Exception as exc:
            logger.warning("AgentLoop: vector search disabled (%s)", exc)

        # Build a session-scoped Indexer so every write_file/edit_file call
        # immediately re-embeds the file into this session's Qdrant collection.
        # The Indexer reuses the same VectorStore instance already connected
        # to session.collection_name — no second connection needed.
        # Manifest is stored in .sovereign/sessions/<id>/manifest.json so
        # each session tracks its own chunk state independently.
        indexer = None
        if vector_store is not None and embed_pipeline is not None:
            try:
                from ..sync.indexer import Indexer
                from ..sync.manifest import ManifestStore

                session_manifest = ManifestStore(
                    project_root=project_root,
                    manifest_path=session.session_dir / "manifest.json",
                )
                indexer = Indexer(
                    project_root=project_root,
                    store=vector_store,
                    pipeline=embed_pipeline,
                    manifest_store=session_manifest,
                )
                indexer.load_manifest()
                logger.info(
                    "AgentLoop: live-reindex enabled (session manifest in %s)",
                    session.session_dir,
                )
            except Exception as exc:
                logger.warning("AgentLoop: live-reindex disabled (%s)", exc)

        return cls(
            project_root=project_root,
            router=router,
            session=session,
            session_manager=session_manager,
            vector_store=vector_store,
            embed_pipeline=embed_pipeline,
            indexer=indexer,
        )

    # ─── Public API ───────────────────────────────────────────────────────────

    def run(self, query: str) -> AgentResult:
        """Process one user turn end-to-end and persist session to disk."""
        logger.info("AgentLoop.run [%s]: %s", self._session.session_id[:8], query[:80])

        codebase_summary = self._retrieve_summary(query)

        plan = self._planner.plan(
            query=query,
            codebase_summary=codebase_summary,
            conversation_history=self._session.history,
        )
        logger.info("Plan: %s", plan.summary())

        result = self._orchestrator.execute(
            plan=plan,
            conversation_history=self._session.history,
        )

        # ── Update session state ──────────────────────────────────────────────
        self._session.history.append({"role": "user", "content": query})
        self._session.history.append({"role": "assistant", "content": result.response})

        # Cap history at 40 messages (20 turns) to avoid context bloat
        if len(self._session.history) > 40:
            self._session.history = self._session.history[-40:]

        self._session.turn_count += 1
        self._session.files_modified.extend(result.files_modified)
        self._session.commands_run.extend(result.commands_run)

        # ── Persist to disk ───────────────────────────────────────────────────
        self._session_manager.save(self._session)

        return result

    def clear_history(self) -> None:
        """Reset conversation history for this session (keeps session alive)."""
        self._session.history.clear()
        self._session_manager.save(self._session)

    # ─── Properties ───────────────────────────────────────────────────────────

    @property
    def session(self) -> SessionInfo:
        return self._session

    @property
    def session_id(self) -> str:
        return self._session.session_id

    @property
    def history(self) -> list[dict]:
        return list(self._session.history)

    # ─── Internal ─────────────────────────────────────────────────────────────

    def _retrieve_summary(self, query: str, top_k: int = 6) -> str:
        """Quick vector search to give the Planner codebase context."""
        if not self._store or not self._pipeline:
            return ""
        try:
            query_vec = self._pipeline.embed_query(query)
            results = self._store.search(query_vec, top_k=top_k)
            if not results:
                return ""
            lines = []
            for r in results:
                c = r.chunk
                lines.append(
                    f"- {c.name or 'block'} in {c.file_path}:{c.start_line}"
                    + (f" — {c.docstring[:80]}" if c.docstring else "")
                )
            return "\n".join(lines)
        except Exception as exc:
            logger.debug("Retrieval summary failed: %s", exc)
            return ""