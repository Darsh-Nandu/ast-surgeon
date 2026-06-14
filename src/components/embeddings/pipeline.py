"""
Embedding pipeline - batching, retry, and progress reporting.

DESIGN NOTE on retry strategy:
  Truncated exponential backoff with jitter: wait = 2^attempt + random(0,1)s.
  Jitter prevents thundering-herd on batch jobs hitting rate limits simultaneously.
  Max 4 retries ≈ 17s total wait - covers transient blips without stalling the watcher.

DESIGN NOTE on zero-vector fallback:
  When a batch fails after all retries, we emit zero-vectors for those chunks
  and record their names in EmbedStats.failed_chunks. We never silently drop work -
  failed chunks appear in the manifest with no vector_id, so the next full
  re-index will retry them automatically.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..chunker.models import CodeChunk
from .providers import EmbeddingProvider, EmbeddingError, get_provider

logger = logging.getLogger(__name__)


# Telemetry

@dataclass
class EmbedStats:
    """Accumulated statistics for one pipeline.run() call."""
    total_chunks:   int   = 0
    total_batches:  int   = 0
    total_retries:  int   = 0
    elapsed_seconds: float = 0.0
    failed_chunks:  list[str] = field(default_factory=list)

    @property
    def chunks_per_second(self) -> float:
        return self.total_chunks / self.elapsed_seconds if self.elapsed_seconds else 0.0

    def __str__(self) -> str:
        return (
            f"EmbedStats(chunks={self.total_chunks}, "
            f"batches={self.total_batches}, retries={self.total_retries}, "
            f"{self.chunks_per_second:.1f} chunks/s, "
            f"failed={len(self.failed_chunks)})"
        )


# Pipeline

class EmbeddingPipeline:
    """
    Orchestrates chunk → vector conversion with batching and retry.

    Usage:
        provider = get_provider("cohere")
        pipeline = EmbeddingPipeline(provider=provider)
        vectors, stats = pipeline.run(chunks)
        # vectors[i] corresponds to chunks[i]
        query_vec = pipeline.embed_query("how does auth work?")
    """

    MAX_RETRIES  = 4
    BASE_BACKOFF = 1.0   # seconds

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        batch_size: int = 64,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Args:
            provider:    EmbeddingProvider to use. Auto-detected if not given.
            batch_size:  Max chunks per provider call (provider may sub-batch further).
            on_progress: Optional callback(done, total) for progress reporting.
        """
        self._provider   = provider or get_provider()
        self._batch_size = batch_size
        self._on_progress = on_progress

    @property
    def provider(self) -> EmbeddingProvider:
        return self._provider

    def run(self, chunks: list[CodeChunk]) -> tuple[list[list[float]], EmbedStats]:
        """
        Embed all chunks. Returns (vectors, stats) in the same order as input.

        Chunks that fail after MAX_RETRIES are returned as zero-vectors and
        recorded in stats.failed_chunks - never silently dropped.
        """
        if not chunks:
            return [], EmbedStats()

        stats = EmbedStats(total_chunks=len(chunks))
        t0 = time.monotonic()

        zero_vec    = [0.0] * self._provider.dimension
        all_vectors = [zero_vec[:] for _ in chunks]
        done = 0

        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            vecs  = self._embed_with_retry(batch, stats)

            for i, vec in enumerate(vecs):
                all_vectors[batch_start + i] = vec

            done += len(batch)
            stats.total_batches += 1

            if self._on_progress:
                self._on_progress(done, len(chunks))

        stats.elapsed_seconds = time.monotonic() - t0
        logger.info(
            "Embedded %d chunks via %s in %.2fs (%s)",
            stats.total_chunks,
            type(self._provider).__name__,
            stats.elapsed_seconds,
            stats,
        )
        return all_vectors, stats

    def embed_query(self, query: str) -> list[float]:
        """Embed a single search query string."""
        return self._provider.embed_query(query)

    # Internal

    def _embed_with_retry(
        self,
        batch: list[CodeChunk],
        stats: EmbedStats,
    ) -> list[list[float]]:
        """Embed a batch with exponential-backoff retry on EmbeddingError."""
        zero_vec = [0.0] * self._provider.dimension

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return self._provider.embed_chunks(batch)
            except EmbeddingError as exc:
                if attempt == self.MAX_RETRIES:
                    logger.error(
                        "Batch of %d chunks failed after %d retries: %s",
                        len(batch), self.MAX_RETRIES, exc,
                    )
                    for chunk in batch:
                        stats.failed_chunks.append(chunk.qualified_name())
                    return [zero_vec[:] for _ in batch]

                wait = (2 ** attempt) * self.BASE_BACKOFF + random.random()
                stats.total_retries += 1
                logger.warning(
                    "Embed attempt %d/%d failed (%s) - retrying in %.1fs",
                    attempt + 1, self.MAX_RETRIES, exc, wait,
                )
                time.sleep(wait)

        return [zero_vec[:] for _ in batch]  # unreachable; satisfies type checker