"""Embedding package - provider abstraction and batching pipeline."""

from .providers import (
    EmbeddingProvider, EmbeddingError,
    VoyageProvider, OpenAIProvider, LocalProvider,
    get_provider, EMBED_DIM,
)
from .pipeline import EmbeddingPipeline, EmbedStats

__all__ = [
    "EmbeddingProvider", "EmbeddingError",
    "VoyageProvider", "OpenAIProvider", "LocalProvider",
    "get_provider", "EMBED_DIM",
    "EmbeddingPipeline", "EmbedStats",
]