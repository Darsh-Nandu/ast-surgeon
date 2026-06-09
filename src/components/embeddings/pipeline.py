"""
Embedding pipeline — orchestrates batching, retry, and rate-limit handling.

DESIGN NOTE on retry strategy:
  We use truncated exponential backoff with jitter (2^attempt + random(0,1)).
  Pure exponential without jitter causes thundering-herd on batch jobs.
  Max 4 retries = up to ~17 seconds of total wait, which covers most transient
  API blips without hanging the sync daemon too long.
"""

from __future__ import annotations

import random
import time
import logging
from dataclasses import dataclass, field
from typing import Callable, Optional

from ..chunker.models import CodeChunk
from .providers import EmbeddingProvider, EmbeddingError, get_provider

logger = logging.getLogger(__name__)


# Telemetry
@dataclass
class EmbedStats:
    """Accumulated stats for one pipeline.run() call. Used by the eval harness."""
    total_chunks: int = 0
    total_batches: int = 0
    total_retries: int = 0
    elapsed_seconds: float = 0.0
    failed_chunks: list[str] = field(default_factory=list)  # qualified names

    @property
    def chunks_per_second(self) -> float:
        if self.elapsed_seconds == 0:
            return 0.0
        return self.total_chunks / self.elapsed_seconds


# Pipeline
class EmbeddingPipeline:
    """Orchestrates chunk → vector conversion with batching and retry.

    Usage:
        pipeline = EmbeddingPipeline(provider=get_provider("local"))
        vectors, stats = pipeline.run(chunks)
        # vectors[i] corresponds to chunks[i]
    """

    MAX_RETRIES = 4
    BASE_BACKOFF = 1.0   # seconds

    def __init__(
        self,
        provider: Optional[EmbeddingProvider] = None,
        batch_size: int = 64,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ):
        """
        Args:
            provider:    EmbeddingProvider to use. Defaults to auto-detected provider.
            batch_size:  Number of chunks per API call. Provider may further subdivide.
            on_progress: Optional callback(done, total) for progress reporting.
        """
        self._provider = provider or get_provider()
        self._batch_size = batch_size
        self._on_progress = on_progress

    def run(self, chunks: list[CodeChunk]) -> tuple[list[list[float]], EmbedStats]:
        """Embed all chunks and return (vectors, stats).

        Vectors are returned in the same order as input chunks.
        Chunks that fail after all retries are returned as zero-vectors and
        logged in stats.failed_chunks — we never silently drop work.

        Args:
            chunks: list of CodeChunks to embed.

        Returns:
            (vectors, stats) — vectors[i] corresponds to chunks[i].
        """
        stats = EmbedStats(total_chunks=len(chunks))
        t0 = time.monotonic()

        all_vectors: list[list[float]] = [[] for _ in chunks]
        done = 0

        for batch_start in range(0, len(chunks), self._batch_size):
            batch = chunks[batch_start : batch_start + self._batch_size]
            batch_vecs = self._embed_with_retry(batch, stats)

            for i, vec in enumerate(batch_vecs):
                all_vectors[batch_start + i] = vec

            done += len(batch)
            stats.total_batches += 1
            if self._on_progress:
                self._on_progress(done, len(chunks))

        stats.elapsed_seconds = time.monotonic() - t0
        logger.info(
            "Embedded %d chunks in %.2fs (%.1f chunks/s, %d retries)",
            stats.total_chunks,
            stats.elapsed_seconds,
            stats.chunks_per_second,
            stats.total_retries,
        )
        return all_vectors, stats

    def embed_query(self, query: str) -> list[float]:
        """Embed a single query string. Does NOT go through run()."""
        return self._provider.embed_query(query)

    def _embed_with_retry(
        self, batch: list[CodeChunk], stats: EmbedStats
    ) -> list[list[float]]:
        """Call provider with exponential-backoff retry.

        Returns a list of vectors. On total failure, returns zero-vectors
        and records chunk names in stats.failed_chunks.
        """
        zero_vec = [0.0] * self._provider.dimension

        for attempt in range(self.MAX_RETRIES + 1):
            try:
                return self._provider.embed_chunks(batch)
            except EmbeddingError as exc:
                if attempt == self.MAX_RETRIES:
                    logger.error(
                        "Batch failed after %d retries: %s", self.MAX_RETRIES, exc
                    )
                    for chunk in batch:
                        stats.failed_chunks.append(chunk.qualified_name())
                    return [zero_vec] * len(batch)

                # Backoff with jitter
                wait = (2 ** attempt) * self.BASE_BACKOFF + random.random()
                stats.total_retries += 1
                logger.warning(
                    "Embedding attempt %d failed (%s), retrying in %.1fs",
                    attempt + 1, exc, wait,
                )
                time.sleep(wait)

        return [zero_vec] * len(batch)   # unreachable but satisfies type checker