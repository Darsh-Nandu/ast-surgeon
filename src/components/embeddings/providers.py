"""
Embedding pipeline — provider abstraction + concrete implementations.

Provider priority (configured via SOVEREIGN_EMBEDDING_PROVIDER env var):
  1. "voyage"  — voyage-code-2 via Voyage AI API (best code recall)
  2. "openai"  — text-embedding-3-large via OpenAI API (fallback)
  3. "local"   — sentence-transformers all-MiniLM-L6-v2 (offline/free fallback)

DESIGN NOTE on batching:
  Embedding APIs charge per token and have throughput limits. We always batch
  chunks rather than embedding one at a time. The `embed_chunks` method accepts
  a list and returns a parallel list of vectors. Callers must not call it in a
  tight loop per-chunk.

DESIGN NOTE on dimensionality:
  voyage-code-2: 1536 dims
  text-embedding-3-large: 3072 dims (we use 1536 via truncation param)
  all-MiniLM-L6-v2: 384 dims

  We normalise to 1536 across all providers so Qdrant collection config
  never needs to change when you switch providers.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from typing import Optional
import httpx

from ..chunker.models import CodeChunk


EMBED_DIM = 1536   # canonical dimension across all providers

# Abstract base
class EmbeddingProvider(ABC):
    """Interface all embedding providers must satisfy."""

    @abstractmethod
    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """
        Return one embedding vector per chunk, in the same order.
        """

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a query string for retrieval (asymmetric if provider supports it)."""

    @property
    def dimension(self) -> int:
        return EMBED_DIM

class EmbeddingError(RuntimeError):
    """Raised when an embedding provider call fails unrecoverably."""

# Shared: chunk → text preparation
def _chunk_to_text(chunk: CodeChunk) -> str:
    """Prepare a chunk's text for embedding.

    Prepends a short header so the model sees context like:
      "python function Calculator.add\n<source>"
    This improves retrieval for natural-language queries.

    DESIGN NOTE: voyage-code-2 is trained on code with such headers, so this
    helps. For prose chunks we skip the header since the content speaks for itself.
    """
    if chunk.chunk_type.value == "text_block":
        return chunk.content

    header = f"{chunk.language} {chunk.chunk_type.value}"
    if chunk.name:
        header += f" {chunk.name}"
    return f"{header}\n\n{chunk.content}"


# Voyage AI provider
class VoyageProvider(EmbeddingProvider):
    """Voyage AI voyage-code-2 embeddings.

    DESIGN NOTE: Voyage uses asymmetric embedding — documents and queries
    use different input_type values. This measurably improves recall on code.
    """

    API_URL = "https://api.voyageai.com/v1/embeddings"
    MODEL = "voyage-code-2"
    BATCH_SIZE = 128

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self._api_key:
            raise EmbeddingError("VOYAGE_API_KEY not set")
        self._client = httpx.Client(timeout=60.0)

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        texts = [_chunk_to_text(c) for c in chunks]
        return self._embed(texts, input_type="document")

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], input_type="query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            resp = self._client.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self.MODEL, "input": batch, "input_type": input_type},
            )
            if resp.status_code != 200:
                raise EmbeddingError(
                    f"Voyage API error {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            for item in sorted(data["data"], key=lambda x: x["index"]):
                all_vectors.append(item["embedding"])
        return all_vectors


# OpenAI provider (fallback)
class OpenAIProvider(EmbeddingProvider):
    """OpenAI text-embedding-3-large with dimension truncation to 1536."""

    API_URL = "https://api.openai.com/v1/embeddings"
    MODEL = "text-embedding-3-large"
    BATCH_SIZE = 100

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise EmbeddingError("OPENAI_API_KEY not set")
        self._client = httpx.Client(timeout=60.0)

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        texts = [_chunk_to_text(c) for c in chunks]
        return self._embed(texts)

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        all_vectors: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            resp = self._client.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self.MODEL,
                    "input": batch,
                    "dimensions": EMBED_DIM,  # truncate to 1536
                    "encoding_format": "float",
                },
            )
            if resp.status_code != 200:
                raise EmbeddingError(
                    f"OpenAI API error {resp.status_code}: {resp.text[:200]}"
                )
            data = resp.json()
            for item in sorted(data["data"], key=lambda x: x["index"]):
                all_vectors.append(item["embedding"])
        return all_vectors


# Local provider — sentence-transformers (no API key required)
class LocalProvider(EmbeddingProvider):
    """Local sentence-transformers embedding — zero cost, works offline.

    DESIGN NOTE on dimension mismatch:
      all-MiniLM-L6-v2 produces 384-dim vectors. We zero-pad to 1536 so the
      Qdrant collection shape stays consistent. Padding hurts recall slightly
      but keeps the architecture uniform. In prod, switch to voyage-code-2.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.MODEL_NAME)
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed. "
                "Run: pip install sentence-transformers"
            )

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        texts = [_chunk_to_text(c) for c in chunks]
        vecs = self._model.encode(texts, normalize_embeddings=True).tolist()
        return [self._pad(v) for v in vecs]

    def embed_query(self, query: str) -> list[float]:
        vec = self._model.encode([query], normalize_embeddings=True)[0].tolist()
        return self._pad(vec)

    @staticmethod
    def _pad(vec: list[float]) -> list[float]:
        """Zero-pad to EMBED_DIM."""
        if len(vec) >= EMBED_DIM:
            return vec[:EMBED_DIM]
        return vec + [0.0] * (EMBED_DIM - len(vec))


# Factory
def get_provider(name: Optional[str] = None) -> EmbeddingProvider:
    """Instantiate the configured embedding provider.

    Resolution order:
      1. `name` argument
      2. SOVEREIGN_EMBEDDING_PROVIDER environment variable
      3. Auto-detect: try voyage → openai → local

    Args - name: "voyage" | "openai" | "local" | None
    """
    chosen = name or os.environ.get("SOVEREIGN_EMBEDDING_PROVIDER", "auto")

    if chosen == "voyage":
        return VoyageProvider()
    if chosen == "openai":
        return OpenAIProvider()
    if chosen == "local":
        return LocalProvider()
    if chosen == "auto":
        # Try each in priority order, fall back gracefully
        for Provider, env_key in [
            (VoyageProvider, "VOYAGE_API_KEY"),
            (OpenAIProvider, "OPENAI_API_KEY"),
        ]:
            if os.environ.get(env_key):
                try:
                    return Provider()
                except EmbeddingError:
                    continue
        return LocalProvider()

    raise EmbeddingError(f"Unknown provider: {chosen!r}")