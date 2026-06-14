"""
ChromaDB vector store backend.

ChromaDB is the most popular local-first vector store — no Docker required,
data lives in a local folder, and it's pip-installable in seconds.

Install:  pip install "ast-surgeon[chroma]"

Modes:
    ChromaStore.local(path="./my_db")             # persistent on disk (default)
    ChromaStore.in_memory()                        # ephemeral, great for tests
    ChromaStore.http(host="...", port=8000)        # remote ChromaDB server

DESIGN NOTE on scoring:
    ChromaDB returns cosine *distance* in [0, 2] (0 = identical).
    We convert to similarity via  score = 1 - distance / 2
    so scores land in [0, 1] consistent with Qdrant and Pinecone.

DESIGN NOTE on metadata filtering:
    ChromaDB uses a `where` dict with MongoDB-style operators.
    We translate filter_language / filter_file into that format.
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


class ChromaStore(VectorStore):
    """
    ChromaDB-backed vector store.

    Quick start:
        store = ChromaStore.local()                   # ./chroma_db folder
        store = ChromaStore.local(path="/tmp/my_db")  # custom path
        store = ChromaStore.in_memory()               # no disk, great for CI
        store.ensure_collection(dim=provider.dimension)
    """

    def __init__(self, client, collection_name: str = COLLECTION_NAME):
        self._client = client
        self._collection_name = collection_name
        self._collection = None   # set by ensure_collection()

    # Factories

    @classmethod
    def local(
        cls,
        path: str = "./chroma_db",
        collection: str = COLLECTION_NAME,
    ) -> "ChromaStore":
        """Persistent on-disk ChromaDB. No server needed."""
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed.\n"
                "Run: pip install 'ast-surgeon[chroma]'"
            )
        client = chromadb.PersistentClient(path=path)
        return cls(client, collection)

    @classmethod
    def in_memory(cls, collection: str = COLLECTION_NAME) -> "ChromaStore":
        """
        Ephemeral in-memory ChromaDB. Perfect for tests and small projects.

        Each call gets its own isolated database — chromadb's EphemeralClient
        shares a default database across instances in the same process, which
        would otherwise leak collections between unrelated stores/tests.
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed.\n"
                "Run: pip install 'ast-surgeon[chroma]'"
            )
        import uuid
        client = chromadb.EphemeralClient()
        db_name = f"ast-surgeon-{uuid.uuid4().hex}"
        client._admin_client.create_database(name=db_name, tenant=client.tenant)
        client.set_database(db_name)
        return cls(client, collection)

    @classmethod
    def http(
        cls,
        host: str = "localhost",
        port: int = 8000,
        collection: str = COLLECTION_NAME,
    ) -> "ChromaStore":
        """Connect to a remote ChromaDB server."""
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb is not installed.\n"
                "Run: pip install 'ast-surgeon[chroma]'"
            )
        client = chromadb.HttpClient(host=host, port=port)
        return cls(client, collection)

    # Collection lifecycle

    def ensure_collection(self, dim: int) -> None:
        """
        Get or create the collection with cosine similarity.

        ChromaDB handles its own HNSW index — we just set the space to cosine.
        The `dim` argument is stored for reference but ChromaDB infers it
        automatically from the first upsert.
        """
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"hnsw:space": "cosine", "dim": dim},
        )
        logger.info(
            "ChromaDB collection %r ready (dim=%d)", self._collection_name, dim
        )

    def _coll(self):
        if self._collection is None:
            raise RuntimeError(
                "Call ensure_collection(dim) before using the store."
            )
        return self._collection

    def drop_collection(self) -> None:
        """Delete the collection. Destructive — use in tests only."""
        self._client.delete_collection(self._collection_name)
        self._collection = None
        logger.warning("Dropped ChromaDB collection %r", self._collection_name)

    # Write

    def upsert(self, chunks: list[CodeChunk], vectors: list[list[float]]) -> list[str]:
        assert len(chunks) == len(vectors), (
            f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}"
        )

        ids       = [chunk_vector_id(c) for c in chunks]
        payloads  = [chunk_to_payload(c) for c in chunks]

        # ChromaDB requires all metadata values to be str/int/float/bool
        sanitised = [_sanitise_metadata(p) for p in payloads]

        # ChromaDB upsert in one call (it handles internal batching)
        self._coll().upsert(ids=ids, embeddings=vectors, metadatas=sanitised)
        logger.debug(
            "Upserted %d chunks into ChromaDB %r", len(chunks), self._collection_name
        )
        return ids

    def delete_by_ids(self, vector_ids: list[str]) -> None:
        if not vector_ids:
            return
        self._coll().delete(ids=vector_ids)
        logger.debug(
            "Deleted %d items from ChromaDB %r", len(vector_ids), self._collection_name
        )

    def delete_by_file(self, file_path: str) -> int:
        # Fetch IDs matching the file, then delete
        results = self._coll().get(
            where={"file_path": {"$eq": file_path}},
            include=[],   # IDs only
        )
        ids = results.get("ids", [])
        if ids:
            self._coll().delete(ids=ids)
        logger.info(
            "Deleted %d ChromaDB chunks for file %r", len(ids), file_path
        )
        return len(ids)

    # Search

    def search(
        self,
        query_vector: list[float],
        top_k: int = 10,
        filter_language: Optional[str] = None,
        filter_file: Optional[str] = None,
    ) -> list[SearchResult]:
        where = _build_where(filter_language, filter_file)

        results = self._coll().query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where if where else None,
            include=["metadatas", "distances"],
        )

        hits = []
        ids       = results["ids"][0]
        distances = results["distances"][0]
        metadatas = results["metadatas"][0]

        for vid, dist, meta in zip(ids, distances, metadatas):
            # Cosine distance ∈ [0,1] in ChromaDB → similarity = 1 - dist
            score = max(0.0, 1.0 - dist)
            chunk = payload_to_chunk(_desanitise_metadata(meta), vid)
            hits.append(SearchResult(vector_id=vid, score=score, chunk=chunk))

        return hits

    def count(self) -> int:
        return self._coll().count()


# Helpers

# Payload fields that are list[str] in CodeChunk and get flattened to a
# comma-joined string for ChromaDB storage. Must be kept in sync with
# CodeChunk's list-typed fields (decorators, calls).
_LIST_FIELDS = ("decorators", "calls")


def _sanitise_metadata(payload: dict) -> dict:
    """
    ChromaDB only accepts str / int / float / bool metadata values.
    Convert lists to comma-joined strings, None to empty string.

    See _desanitise_metadata() for the inverse operation applied on read.
    """
    out = {}
    for k, v in payload.items():
        if v is None:
            out[k] = ""
        elif isinstance(v, list):
            out[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = str(v)
    return out


def _desanitise_metadata(meta: dict) -> dict:
    """
    Inverse of _sanitise_metadata() for fields known to be list[str] on
    CodeChunk (decorators, calls). Splits comma-joined strings back into
    lists; empty strings become empty lists.
    """
    out = dict(meta)
    for field_name in _LIST_FIELDS:
        raw = out.get(field_name, "")
        if isinstance(raw, str):
            out[field_name] = [s.strip() for s in raw.split(",") if s.strip()]
    # docstring was stored as "" for None - restore None for round-trip fidelity
    if out.get("docstring") == "":
        out["docstring"] = None
    if out.get("parent") == "":
        out["parent"] = None
    if out.get("name") == "":
        out["name"] = None
    return out


def _build_where(
    filter_language: Optional[str],
    filter_file: Optional[str],
) -> Optional[dict]:
    """Build a ChromaDB `where` clause from optional filters."""
    conditions = []
    if filter_language:
        conditions.append({"language": {"$eq": filter_language}})
    if filter_file:
        conditions.append({"file_path": {"$eq": filter_file}})

    if not conditions:
        return None
    if len(conditions) == 1:
        return conditions[0]
    return {"$and": conditions}