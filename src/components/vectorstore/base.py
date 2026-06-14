"""
Abstract vector store interface + shared serialization helpers.

Every backend (Qdrant, ChromaDB, Pinecone, ...) implements VectorStore.
The Indexer and the rest of the sync layer talk only to this interface,
so swapping backends is a one-line change in user code.

DESIGN NOTE on dimension handling:
  Unlike the old design that hardcoded 1536 and zero-padded, each
  EmbeddingProvider now reports its own `dimension`. The VectorStore is
  initialised with that dimension via `ensure_collection(dim)`, so the
  collection schema always matches the provider — no padding, no surprises.
"""

from __future__ import annotations

import hashlib
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

from ..chunker.models import CodeChunk, ChunkType


# Search result

@dataclass
class SearchResult:
    """One result returned by VectorStore.search()."""
    vector_id: str
    score: float        # higher = more similar (normalised to [0, 1])
    chunk: CodeChunk

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, chunk={self.chunk!r})"


# Abstract base

class VectorStore(ABC):
    """
    Interface every vector store backend must implement.

    Usage pattern:
        store = QdrantStore.connect(host="localhost", port=6333)
        store.ensure_collection(dim=provider.dimension)
        ids   = store.upsert(chunks, vectors)
        hits  = store.search(query_vec, top_k=10)
        store.delete_by_file("src/auth.py")
    """

    @abstractmethod
    def ensure_collection(self, dim: int) -> None:
        """
        Create the underlying collection/index if it doesn't exist.
        Safe to call multiple times — must be idempotent.

        Args:
            dim: Embedding dimension (must match the EmbeddingProvider used).
        """

    @abstractmethod
    def upsert(
        self,
        chunks: list[CodeChunk],
        vectors: list[list[float]],
    ) -> list[str]:
        """
        Upsert chunks and their embedding vectors.

        Returns the vector IDs assigned to each chunk (same order as input).
        Upserts are idempotent — re-indexing the same chunk produces the same ID.
        """

    @abstractmethod
    def delete_by_ids(self, vector_ids: list[str]) -> None:
        """Delete specific points/documents by their IDs."""

    @abstractmethod
    def delete_by_file(self, file_path: str) -> int:
        """
        Delete all vectors belonging to a specific file.
        Returns the number of vectors deleted.
        """

    @abstractmethod
    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_language: Optional[str] = None,
        filter_file: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Dense vector search with optional metadata filters.

        Args:
            query_vector:    Embedded query (same dimension as stored vectors).
            top_k:           Maximum results to return.
            filter_language: Restrict to a specific language (e.g. "python").
            filter_file:     Restrict to a specific file path.

        Returns:
            List of SearchResult, sorted by score descending.
        """

    @abstractmethod
    def count(self) -> int:
        """Return total number of indexed chunk vectors."""


# Shared serialisation helpers

def chunk_vector_id(chunk: CodeChunk) -> str:
    """
    Deterministic UUID derived from (file_path, name, content_hash).

    Properties:
    - Same chunk body in the same file → same ID (idempotent upsert).
    - Function moved to a different file → new ID (correct: new location).
    - Copy-pasted function in two files → different IDs (correct: different locations).
    """
    key = f"{chunk.file_path}::{chunk.name}::{chunk.content_hash}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))


def chunk_to_payload(chunk: CodeChunk) -> dict[str, Any]:
    """Serialise a CodeChunk to a flat dict suitable for any vector store payload."""
    return {
        "file_path":    chunk.file_path,
        "language":     chunk.language,
        "chunk_type":   chunk.chunk_type.value,
        "name":         chunk.name,
        "content":      chunk.content,
        "content_hash": chunk.content_hash,
        "start_line":   chunk.start_line,
        "end_line":     chunk.end_line,
        "decorators":   chunk.decorators,
        "docstring":    chunk.docstring,
        "calls":        chunk.calls,
        "parent":       chunk.parent,
    }


def payload_to_chunk(payload: dict[str, Any], vector_id: str) -> CodeChunk:
    """Reconstruct a CodeChunk from a stored payload dict."""
    return CodeChunk(
        file_path    = payload.get("file_path", ""),
        language     = payload.get("language", "text"),
        chunk_type   = ChunkType(payload.get("chunk_type", "text_block")),
        name         = payload.get("name"),
        content      = payload.get("content", ""),
        content_hash = payload.get("content_hash", ""),
        start_line   = payload.get("start_line", 0),
        end_line     = payload.get("end_line", 0),
        decorators   = payload.get("decorators", []),
        docstring    = payload.get("docstring"),
        calls        = payload.get("calls", []),
        parent       = payload.get("parent"),
        vector_id    = vector_id,
    )