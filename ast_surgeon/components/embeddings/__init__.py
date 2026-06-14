"""Embedding package - provider abstraction and batching pipeline."""

from .providers import (
    EmbeddingProvider, EmbeddingError,
    VoyageProvider, OpenAIProvider, CohereProvider,
    GeminiProvider, MistralProvider, LocalProvider,
    get_provider, list_providers,
)
from .pipeline import EmbeddingPipeline, EmbedStats

__all__ = [
    "EmbeddingProvider", "EmbeddingError",
    "VoyageProvider", "OpenAIProvider", "CohereProvider",
    "GeminiProvider", "MistralProvider", "LocalProvider",
    "get_provider", "list_providers",
    "EmbeddingPipeline", "EmbedStats",
]