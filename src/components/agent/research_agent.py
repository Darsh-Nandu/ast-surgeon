"""
ResearchAgent — formulates and executes knowledge-retrieval queries.

Phase 3 will implement:
  - Receives a research question from the Orchestrator or MainOrchestrator
  - Calls LLM (TaskType.RESEARCH → llama-3.1-8b-instant) to formulate
    structured search queries
  - Integrates with the vector store and/or external search tools
  - Returns a structured summary for use by KGAgent or SubAgents

TODO (Phase 3): implement ResearchAgent.research()
"""

from __future__ import annotations

# TODO (Phase 3): from .models import TaskType
# TODO (Phase 3): from .router import ModelRouter


class ResearchAgent:
    """Stub — to be implemented in Phase 3."""

    def research(self, query: str) -> str:
        """Formulate search queries and return a synthesised research summary.

        Args:
            query: The research question or topic.

        Returns:
            A string summary of research findings.

        TODO (Phase 3): implement query formulation + retrieval + synthesis.
        """
        raise NotImplementedError("ResearchAgent.research() — implement in Phase 3")
