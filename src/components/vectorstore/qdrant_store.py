"""
Qdrant vector store backend.

Install:  pip install "ast-surgeon[qdrant]"
Run locally: docker run -p 6333:6333 qdrant/qdrant

DESIGN NOTE on vector IDs:
  We use deterministic UUIDs so re-indexing an unchanged chunk is a true
  no-op at the Qdrant level — the upsert lands on the same point ID.

DESIGN NOTE on dynamic dimensions:
  The collection is created with the dimension reported by whatever
  EmbeddingProvider the user chose. No hardcoded 1536, no zero-padding.
"""

from __future__ import annotations

import logging
from typing import Optional

from .base import (
    VectorStore, SearchResult,
    chunk_vector_id, chunk_to_payload, payload_to_chunk,
)
from ..chunker.models import CodeChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "ast_surgeon"


class QdrantStore(VectorStore):
    """
    Qdrant-backed vector store.

    Quick start:
        store = QdrantStore.connect()                    # local Docker
        store = QdrantStore.connect(in_memory=True)      # tests / CI
        store = QdrantStore.connect(url="https://...")   # Qdrant Cloud
        store.ensure_collection(dim=provider.dimension)
    """

    def __init__(self, client, collection: str = COLLECTION_NAME):
        self._client = client
        self._collection = collection

    # Factory

    @classmethod
    def connect(
        cls,
        host: str = "localhost",
        port: int = 6333,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection: str = COLLECTION_NAME,
        in_memory: bool = False,
    ) -> "QdrantStore":
        """
        Connect to a Qdrant instance.

        Args:
            host:       Qdrant host for local/self-hosted (default: localhost).
            port:       Qdrant HTTP port (default: 6333).
            url:        Full URL for Qdrant Cloud, e.g. "https://xyz.qdrant.io".
            api_key:    API key for Qdrant Cloud.
            collection: Collection name (default: "ast_surgeon").
            in_memory:  Use an in-memory instance — no server required (tests/CI).
        """
        try:
            from qdrant_client import QdrantClient
        except ImportError:
            raise ImportError(
                "qdrant-client is not installed.\n"
                "Run: pip install 'ast-surgeon[qdrant]'"
            )

        if in_memory:
            client = QdrantClient(":memory:")
        elif url:
            client = QdrantClient(url=url, api_key=api_key)
        else:
            client = QdrantClient(host=host, port=port, api_key=api_key)

        return cls(client, collection)

    # Collection lifecycle

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection in existing:
            logger.debug("Qdrant collection %r already exists", self._collection)
            return

        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection %r (dim=%d)", self._collection, dim)

    def drop_collection(self) -> None:
        """Delete the collection entirely. Destructive — use in tests only."""
        self._client.delete_collection(self._collection)
        logger.warning("Dropped Qdrant collection %r", self._collection)

    # Write

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> list[str]:
        from qdrant_client.models import PointStruct, UpdateStatus

        assert len(chunks) == len(vectors), (
            f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}"
        )

        BATCH = 128
        all_ids: list[str] = []

        for i in range(0, len(chunks), BATCH):
            batch_chunks = chunks[i : i + BATCH]
            batch_vecs   = vectors[i : i + BATCH]

            points = []
            for chunk, vec in zip(batch_chunks, batch_vecs):
                vid = chunk_vector_id(chunk)
                all_ids.append(vid)
                points.append(PointStruct(
                    id=vid, vector=vec, payload=chunk_to_payload(chunk)
                ))

            result = self._client.upsert(
                collection_name=self._collection, points=points
            )
            if result.status != UpdateStatus.COMPLETED:
                logger.error("Qdrant upsert batch %d failed: %s", i // BATCH, result.status)

        logger.debug("Upserted %d chunks into Qdrant %r", len(chunks), self._collection)
        return all_ids

    def delete_by_ids(self, vector_ids: list[str]) -> None:
        if not vector_ids:
            return
        from qdrant_client.models import PointIdsList
        self._client.delete(
            collection_name=self._collection,
            points_selector=PointIdsList(points=vector_ids),
        )
        logger.debug("Deleted %d points from Qdrant %r", len(vector_ids), self._collection)

    def delete_by_file(self, file_path: str) -> int:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

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
        logger.info("Deleted %d Qdrant chunks for file %r", len(ids), file_path)
        return len(ids)

    # Search

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_language: Optional[str] = None,
        filter_file: Optional[str] = None,
    ) -> list[SearchResult]:
        from qdrant_client.models import Filter, FieldCondition, MatchValue

        must = []
        if filter_language:
            must.append(FieldCondition(key="language", match=MatchValue(value=filter_language)))
        if filter_file:
            must.append(FieldCondition(key="file_path", match=MatchValue(value=filter_file)))

        result = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            limit=top_k,
            query_filter=Filter(must=must) if must else None,
            with_payload=True,
        )
        return [
            SearchResult(
                vector_id=str(p.id),
                score=p.score,
                chunk=payload_to_chunk(p.payload or {}, str(p.id)),
            )
            for p in result.points
        ]

    def count(self) -> int:
        return self._client.count(collection_name=self._collection).count