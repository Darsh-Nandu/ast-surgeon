"""
AgentLoop — top-level entry point that wires the full agent pipeline.

This is the single class session.py calls. It owns:
  Planner → Orchestrator → (SubAgents) → Synthesiser

And hides all of that complexity behind one method: run(query) → AgentResult.

WHY this wrapper layer:
  session.py doesn't need to know about Planners, Orchestrators, or SubAgents.
  It calls loop.run(query) and gets back an AgentResult. This makes testing
  trivial (mock the loop) and lets us swap internal architecture without
  touching the CLI.

DESIGN NOTE on codebase summary for the Planner:
  We do a quick vector search before calling the Planner so it has real context
  about the codebase when deciding how to decompose the task. A Planner that
  knows "AuthService is in src/auth.py and has 3 methods" makes much better
  subtask decisions than one working blind.
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

logger = logging.getLogger(__name__)


class AgentLoop:
    """Top-level agent pipeline: Planner → Orchestrator → SubAgents → Synthesiser.

    Usage:
        loop = AgentLoop.create(project_root="/path/to/project")
        result = loop.run("Refactor the auth module and add tests")
        print(result.response)
        print(result.step_summary())
    """

    def __init__(
        self,
        project_root: str | Path,
        router: ModelRouter,
        vector_store=None,
        embed_pipeline=None,
    ):
        self._root = Path(project_root)
        self._router = router
        self._store = vector_store
        self._pipeline = embed_pipeline

        self._planner = Planner(router)
        self._orchestrator = Orchestrator(
            router=router,
            project_root=project_root,
            vector_store=vector_store,
            embed_pipeline=embed_pipeline,
        )

        # Session-level conversation history (multi-turn context)
        self._history: list[dict] = []


    # Factory

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_provider: Optional[str] = None,
    ) -> "AgentLoop":
        """Wire up all dependencies and return a ready AgentLoop.

        Falls back gracefully if Qdrant isn't running (vector search disabled).
        """
        router = ModelRouter(
            groq_api_key=groq_api_key or os.environ.get("GROQ_API_KEY"),
            gemini_api_key=gemini_api_key or os.environ.get("GEMINI_API_KEY")
                           or os.environ.get("GOOGLE_API_KEY"),
        )

        # Try to connect to Qdrant - non-fatal if unavailable
        vector_store = None
        embed_pipeline = None
        try:
            from ..vectorstore.qdrant_store import VectorStore
            from ..embeddings.providers import get_provider
            from ..embeddings.pipeline import EmbeddingPipeline

            vector_store = VectorStore.connect(host=qdrant_host, port=qdrant_port)
            provider = get_provider(embedding_provider)
            embed_pipeline = EmbeddingPipeline(provider=provider)
            logger.info("AgentLoop: vector search enabled")
        except Exception as exc:
            logger.warning("AgentLoop: vector search disabled (%s)", exc)

        return cls(
            project_root=project_root,
            router=router,
            vector_store=vector_store,
            embed_pipeline=embed_pipeline,
        )


    # Public API

    def run(self, query: str) -> AgentResult:
        """Process one user turn end-to-end.

        Args:
            query: The user's message / task description.

        Returns:
            AgentResult with response, traces, and side-effect lists.
        """
        logger.info("AgentLoop.run: %s", query[:80])

        # Quick retrieval to give the Planner codebase context
        codebase_summary = self._retrieve_summary(query)

        # Plan the work
        plan = self._planner.plan(
            query=query,
            codebase_summary=codebase_summary,
            conversation_history=self._history,
        )
        logger.info("Plan: %s", plan.summary())

        # Execute
        result = self._orchestrator.execute(
            plan=plan,
            conversation_history=self._history,
        )

        # Update session history for next turn
        self._history.append({"role": "user", "content": query})
        self._history.append({"role": "assistant", "content": result.response})

        # Keep history bounded (last 20 turns = 40 messages)
        if len(self._history) > 40:
            self._history = self._history[-40:]

        return result

    def clear_history(self) -> None:
        """Reset conversation history for a fresh session."""
        self._history.clear()

    @property
    def history(self) -> list[dict]:
        return list(self._history)


    # Internal

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