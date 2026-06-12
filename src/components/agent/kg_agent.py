"""
KGAgent (Knowledge-Graph Agent) — builds and updates a structured knowledge graph
from generated/modified code and research findings.

Phase 3 will implement:
  - Receives file content + research summaries after generation/repair
  - Calls LLM (TaskType.KG_BUILD → gemini-2.0-flash) for structured extraction
  - Persists extracted entities and relationships to the vector store
  - Sets AgentResult.kg_updated = True on success

TODO (Phase 3): implement KGAgent.update()
"""

from __future__ import annotations

# TODO (Phase 3): from .models import TaskType, AgentResult
# TODO (Phase 3): from .router import ModelRouter


class KGAgent:
    """Stub — to be implemented in Phase 3."""

    def update(self, content: str, context: str = "") -> dict:
        """Extract entities/relationships from *content* and persist to the KG.

        Args:
            content: Source text (code, docs, research summary) to process.
            context: Optional surrounding context for better extraction.

        Returns:
            dict summarising extracted nodes and edges.

        TODO (Phase 3): implement structured extraction + vector store write.
        """
        raise NotImplementedError("KGAgent.update() — implement in Phase 3")
