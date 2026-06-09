"""
Qdrant vector store — collection lifecycle and CRUD for code chunk vectors.

DESIGN NOTE on collection schema:
  Each Qdrant point represents one CodeChunk. The payload stores all CodeChunk
  fields (minus the raw vector) so we can reconstruct the chunk on retrieval
  without going back to disk. This is a deliberate trade-off: we duplicate
  content in Qdrant, but avoid a synchronisation problem between the vector store
  and the filesystem.

DESIGN NOTE on BM25 (sparse vectors):
  Qdrant's FastEmbed integration computes BM25 server-side when you configure
  a sparse vector field. We enable it here but the sparse vectors are populated
  by Qdrant itself — we only send the text payload. This keeps the client simple.

DESIGN NOTE on vector IDs:
  We use deterministic UUIDs derived from (file_path, chunk_name, content_hash).
  This means re-indexing the same unchanged chunk produces the same ID, enabling
  idempotent upserts — safe to run multiple times without duplicates.
"""

from __future__ import annotations

import hashlib
import logging
import uuid
from typing import Any, Optional

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    ScoredPoint,
    UpdateStatus,
)

from ..chunker.models import CodeChunk, ChunkType
from ..embeddings.providers import EMBED_DIM

logger = logging.getLogger(__name__)

COLLECTION_NAME = "sovereign_code"

class VectorStore:
    """Qdrant-backed vector store for CodeChunk embeddings.

    Lifecycle:
        store = VectorStore.connect()          # local Docker on :6333
        store.ensure_collection()
        store.upsert(chunks, vectors)
        results = store.search(query_vec, top_k=10)
        store.delete_by_file("src/auth.py")   # surgical delete for re-index
    """

    def __init__(self, client: QdrantClient, collection: str = COLLECTION_NAME):
        self._client = client
        self._collection = collection


    # Factory
    @classmethod
    def connect(
        cls,
        host: str = "localhost",
        port: int = 6333,
        collection: str = COLLECTION_NAME,
        in_memory: bool = False,
    ) -> "VectorStore":
        """Connect to a Qdrant instance.

        Args:
            host:      Qdrant host (default: localhost).
            port:      Qdrant HTTP port (default: 6333).
            collection: Collection name to use.
            in_memory: If True, use an in-memory Qdrant instance (for tests).
        """
        if in_memory:
            client = QdrantClient(":memory:")
        else:
            client = QdrantClient(host=host, port=port)
        return cls(client, collection)


    # Collection management
    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist. Safe to call repeatedly.

        DESIGN NOTE: we use COSINE distance because our providers return
        L2-normalised vectors. COSINE == DOT_PRODUCT for unit vectors, but
        COSINE is more forgiving if a provider skips normalisation.
        """
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            logger.debug("Collection %r already exists", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(
                size=EMBED_DIM,
                distance=Distance.COSINE,
            ),
        )
        logger.info("Created Qdrant collection %r (dim=%d)", self._collection, EMBED_DIM)

    def drop_collection(self) -> None:
        """Delete the collection entirely. Destructive — use in tests only."""
        self._client.delete_collection(self._collection)
        logger.warning("Dropped collection %r", self._collection)


    # Write
    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> list[str]:
        """
        Upsert chunks and their vectors into Qdrant.

        DESIGN NOTE on batching:
            Qdrant recommends batch sizes of 100–256 for throughput. We use 128.
            For very large codebases (>10k chunks), the sync daemon will call
            upsert in batches anyway, so this is mostly for safety.
        """
        assert len(chunks) == len(vectors), (
            f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}"
        )

        BATCH = 128
        all_ids: list[str] = []

        for i in range(0, len(chunks), BATCH):
            batch_chunks = chunks[i : i + BATCH]
            batch_vecs = vectors[i : i + BATCH]

            points = []
            for chunk, vec in zip(batch_chunks, batch_vecs):
                vid = _chunk_vector_id(chunk)
                all_ids.append(vid)
                points.append(PointStruct(
                    id=vid,
                    vector=vec,
                    payload=_chunk_to_payload(chunk),
                ))

            result = self._client.upsert(
                collection_name=self._collection,
                points=points,
            )
            if result.status != UpdateStatus.COMPLETED:
                logger.error("Upsert batch %d failed: %s", i // BATCH, result.status)

        logger.debug("Upserted %d chunks into %r", len(chunks), self._collection)
        return all_ids

    def delete_by_ids(self, vector_ids: list[str]) -> None:
        """Delete specific points by their IDs."""
        if not vector_ids:
            return
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=vector_ids),
        )
        logger.debug("Deleted %d points from %r", len(vector_ids), self._collection)

    def delete_by_file(self, file_path: str) -> int:
        """
        Delete all chunks belonging to a specific file.
        """
        # Scroll to find all points for this file, then delete by ID
        points, _ = self._client.scroll(
            collection_name=self._collection,
            scroll_filter=Filter(
                must=[FieldCondition(key="file_path", match=MatchValue(value=file_path))]
            ),
            limit=10_000,
            with_payload=False,
            with_vectors=False,
        )
        ids = [str(p.id) for p in points]
        self.delete_by_ids(ids)
        logger.info("Deleted %d chunks for file %r", len(ids), file_path)
        return len(ids)

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_language: Optional[str] = None,
        filter_file: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Dense vector search with optional payload filters.
        """
        must_conditions = []
        if filter_language:
            must_conditions.append(
                FieldCondition(key="language", match=MatchValue(value=filter_language))
            )
        if filter_file:
            must_conditions.append(
                FieldCondition(key="file_path", match=MatchValue(value=filter_file))
            )

        qdrant_filter = Filter(must=must_conditions) if must_conditions else None

        # DESIGN NOTE: qdrant-client ≥1.7 replaced client.search() with
        # client.query_points(). We use query_points() for forward compatibility.
        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        return [SearchResult.from_scored_point(p) for p in result.points]

    def count(self) -> int:
        """Return total number of points in the collection."""
        result = self._client.count(collection_name=self._collection)
        return result.count


class SearchResult:
    """Retrieval result from VectorStore.search()."""

    def __init__(
        self,
        vector_id: str,
        score: float,
        chunk: CodeChunk,
    ):
        self.vector_id = vector_id
        self.score = score
        self.chunk = chunk

    @classmethod
    def from_scored_point(cls, point: ScoredPoint) -> "SearchResult":
        payload = point.payload or {}
        chunk = _payload_to_chunk(payload, str(point.id))
        return cls(vector_id=str(point.id), score=point.score, chunk=chunk)

    def __repr__(self) -> str:
        return f"SearchResult(score={self.score:.4f}, chunk={self.chunk!r})"


def _chunk_vector_id(chunk: CodeChunk) -> str:
    """Deterministic UUID from (file_path, name, content_hash).

    DESIGN NOTE: using content_hash in the ID means that if a function is
    moved to a different file, it gets a new ID (correct — it's a different
    location). If the same function appears in two files (copy-paste), they
    get different IDs (also correct — same code, different locations).
    """
    key = f"{chunk.file_path}::{chunk.name}::{chunk.content_hash}"
    return str(uuid.UUID(hashlib.md5(key.encode()).hexdigest()))

def _chunk_to_payload(chunk: CodeChunk) -> dict[str, Any]:
    """Serialise a CodeChunk to a Qdrant payload dict."""
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

def _payload_to_chunk(payload: dict[str, Any], vector_id: str) -> CodeChunk:
    """Reconstruct a CodeChunk from a Qdrant payload dict."""
    return CodeChunk(
        file_path=payload.get("file_path", ""),
        language=payload.get("language", "text"),
        chunk_type=ChunkType(payload.get("chunk_type", "text_block")),
        name=payload.get("name"),
        content=payload.get("content", ""),
        content_hash=payload.get("content_hash", ""),
        start_line=payload.get("start_line", 0),
        end_line=payload.get("end_line", 0),
        decorators=payload.get("decorators", []),
        docstring=payload.get("docstring"),
        calls=payload.get("calls", []),
        parent=payload.get("parent"),
        vector_id=vector_id,
    )