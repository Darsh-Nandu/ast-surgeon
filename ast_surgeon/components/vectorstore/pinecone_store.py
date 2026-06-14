"""
Pinecone vector store backend.

Pinecone is the leading managed vector database — no infrastructure to run,
scales to billions of vectors, and has a generous free tier.

Install:  pip install "ast-surgeon[pinecone]"

Usage:
    store = PineconeStore.connect(
        api_key="...",          # or set PINECONE_API_KEY
        index_name="my-index",  # must already exist in your Pinecone console
    )
    store.ensure_collection(dim=provider.dimension)

DESIGN NOTE on index creation:
    Pinecone indexes must be created via the console or the control-plane API
    before first use (they take ~60s to initialise). We don't auto-create them
    here because index creation requires specifying cloud/region and a pod type,
    which are account-specific decisions. `ensure_collection` validates the index
    exists and its dimension matches; it raises clearly if either check fails.

DESIGN NOTE on namespaces:
    We default to namespace="" (the default namespace). Pass `namespace` to
    PineconeStore.connect() to isolate different projects in one index.

DESIGN NOTE on filtering:
    Pinecone supports metadata filtering via MongoDB-style dicts. We translate
    filter_language / filter_file into that format.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

from .base import (
    VectorStore, SearchResult,
    chunk_vector_id, chunk_to_payload, payload_to_chunk,
)
from ..chunker.models import CodeChunk

logger = logging.getLogger(__name__)


class PineconeStore(VectorStore):
    """
    Pinecone-backed vector store.

    Quick start:
        store = PineconeStore.connect(
            api_key="pc-...",         # or env PINECONE_API_KEY
            index_name="ast-surgeon", # pre-created in your Pinecone console
        )
        store.ensure_collection(dim=provider.dimension)
    """

    UPSERT_BATCH = 100   # Pinecone recommends ≤100 vectors per upsert

    def __init__(self, index, namespace: str = ""):
        self._index = index
        self._namespace = namespace

    # Factory

    @classmethod
    def connect(
        cls,
        index_name: str,
        api_key: Optional[str] = None,
        namespace: str = "",
    ) -> "PineconeStore":
        """
        Connect to an existing Pinecone index.

        Args:
            index_name: Name of a pre-created Pinecone index.
            api_key:    Pinecone API key (falls back to PINECONE_API_KEY env var).
            namespace:  Pinecone namespace for data isolation (default: "").
        """
        try:
            from pinecone import Pinecone
        except ImportError:
            raise ImportError(
                "pinecone is not installed.\n"
                "Run: pip install 'ast-surgeon[pinecone]'"
            )

        key = api_key or os.environ.get("PINECONE_API_KEY", "")
        if not key:
            raise ValueError(
                "Pinecone API key not provided. "
                "Pass api_key= or set the PINECONE_API_KEY environment variable."
            )

        pc = Pinecone(api_key=key)
        index = pc.Index(index_name)
        logger.info("Connected to Pinecone index %r (namespace=%r)", index_name, namespace)
        return cls(index, namespace)

    # Collection lifecycle

    def ensure_collection(self, dim: int) -> None:
        """
        Validate that the Pinecone index exists and has the right dimension.

        Does NOT create the index — Pinecone indexes must be created upfront
        (console or control-plane API). Raises clearly if there's a mismatch.
        """
        try:
            stats = self._index.describe_index_stats()
        except Exception as exc:
            raise RuntimeError(
                f"Cannot connect to Pinecone index: {exc}\n"
                "Make sure the index exists and your API key is correct."
            ) from exc

        # Pinecone reports dimension in index stats (v3 client)
        index_dim = getattr(stats, "dimension", None)
        if index_dim and index_dim != dim:
            raise ValueError(
                f"Pinecone index dimension ({index_dim}) does not match "
                f"embedding provider dimension ({dim}). "
                "Create a new index with the correct dimension or switch providers."
            )

        logger.info(
            "Pinecone index ready (dim=%d, vectors=%d)",
            dim, stats.total_vector_count if hasattr(stats, "total_vector_count") else "?",
        )

    # Write

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> list[str]:
        assert len(chunks) == len(vectors), (
            f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}"
        )

        all_ids: list[str] = []

        for i in range(0, len(chunks), self.UPSERT_BATCH):
            batch_chunks = chunks[i : i + self.UPSERT_BATCH]
            batch_vecs   = vectors[i : i + self.UPSERT_BATCH]

            records = []
            for chunk, vec in zip(batch_chunks, batch_vecs):
                vid = chunk_vector_id(chunk)
                all_ids.append(vid)
                records.append({
                    "id":     vid,
                    "values": vec,
                    "metadata": _sanitise_metadata(chunk_to_payload(chunk)),
                })

            self._index.upsert(vectors=records, namespace=self._namespace)

        logger.debug("Upserted %d chunks into Pinecone", len(chunks))
        return all_ids

    def delete_by_ids(self, vector_ids: list[str]) -> None:
        if not vector_ids:
            return
        # Pinecone delete accepts up to 1000 IDs per call
        for i in range(0, len(vector_ids), 1000):
            self._index.delete(
                ids=vector_ids[i : i + 1000],
                namespace=self._namespace,
            )
        logger.debug("Deleted %d Pinecone vectors", len(vector_ids))

    def delete_by_file(self, file_path: str) -> int:
        """
        Delete all vectors for a file using metadata filtering.

        DESIGN NOTE: Pinecone's delete-by-filter requires a paid plan (P1+).
        On free/starter plans, this falls back to a fetch-then-delete pattern
        which is slower but works everywhere.
        """
        try:
            # Preferred: delete-by-metadata (paid plans)
            self._index.delete(
                filter={"file_path": {"$eq": file_path}},
                namespace=self._namespace,
                delete_all=False,
            )
            logger.info("Deleted Pinecone vectors for file %r (filter method)", file_path)
            # Pinecone doesn't return a count for filter deletes
            return -1
        except Exception:
            # Fallback: query to find IDs, then delete
            return self._delete_by_file_fallback(file_path)

    def _delete_by_file_fallback(self, file_path: str) -> int:
        """Fetch IDs by metadata, then delete. Works on all Pinecone plans."""
        results = self._index.query(
            vector=[0.0] * 1,   # dummy — we only want metadata
            filter={"file_path": {"$eq": file_path}},
            top_k=10_000,
            include_values=False,
            include_metadata=False,
            namespace=self._namespace,
        )
        ids = [m["id"] for m in results.get("matches", [])]
        if ids:
            self.delete_by_ids(ids)
        logger.info("Deleted %d Pinecone chunks for file %r (fallback)", len(ids), file_path)
        return len(ids)

    # Search

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_language: Optional[str] = None,
        filter_file: Optional[str] = None,
    ) -> list[SearchResult]:
        pinecone_filter = _build_filter(filter_language, filter_file)

        response = self._index.query(
            vector=query_vector,
            top_k=top_k,
            filter=pinecone_filter,
            include_metadata=True,
            namespace=self._namespace,
        )

        results = []
        for match in response.get("matches", []):
            chunk = payload_to_chunk(match.get("metadata", {}), match["id"])
            results.append(SearchResult(
                vector_id=match["id"],
                score=float(match.get("score", 0.0)),
                chunk=chunk,
            ))
        return results

    def count(self) -> int:
        stats = self._index.describe_index_stats()
        ns_stats = stats.namespaces.get(self._namespace or "", None)
        if ns_stats:
            return ns_stats.vector_count
        return stats.total_vector_count or 0


# Helpers

def _sanitise_metadata(payload: dict) -> dict:
    """Pinecone metadata values must be str / int / float / bool / list[str]."""
    out = {}
    for k, v in payload.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, list):
            out[k] = [str(x) for x in v]
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _build_filter(
    filter_language: Optional[str],
    filter_file: Optional[str],
) -> Optional[dict]:
    conditions = {}
    if filter_language:
        conditions["language"] = {"$eq": filter_language}
    if filter_file:
        conditions["file_path"] = {"$eq": filter_file}
    return conditions if conditions else None