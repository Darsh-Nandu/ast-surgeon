"""
Vector store package.

Supports Qdrant, ChromaDB, and Pinecone out of the box.
Swap backends with a single argument — your chunking + sync logic stays identical.

Quick start:
    from ast_surgeon.components.vectorstore import get_store

    store = get_store("chroma")                          # local, no Docker
    store = get_store("qdrant")                          # local Docker
    store = get_store("pinecone", index_name="my-index") # cloud
    store.ensure_collection(dim=provider.dimension)
"""

from .base import VectorStore, SearchResult, chunk_vector_id, chunk_to_payload, payload_to_chunk
from .qdrant_store import QdrantStore
from .chroma_store import ChromaStore
from .pinecone_store import PineconeStore

__all__ = [
    # Abstract base + result type
    "VectorStore", "SearchResult",
    # Helpers
    "chunk_vector_id", "chunk_to_payload", "payload_to_chunk",
    # Backends
    "QdrantStore", "ChromaStore", "PineconeStore",
    # Factory
    "get_store",
]


def get_store(
    backend: str = "chroma",
    **kwargs,
) -> VectorStore:
    """
    Convenience factory — build a VectorStore by name.

    Args:
        backend: "qdrant" | "chroma" | "pinecone"
        **kwargs: Passed directly to the backend's connect/local factory.

    Examples:
        get_store("chroma")                              # local persistent
        get_store("chroma", path="/data/my_db")
        get_store("chroma", in_memory=True)              # for tests
        get_store("qdrant", host="localhost", port=6333)
        get_store("qdrant", url="https://xyz.qdrant.io", api_key="...")
        get_store("pinecone", index_name="ast-surgeon", api_key="pc-...")
    """
    backend = backend.lower()

    if backend == "qdrant":
        return QdrantStore.connect(**kwargs)

    if backend == "chroma":
        if kwargs.pop("in_memory", False):
            return ChromaStore.in_memory(**kwargs)
        return ChromaStore.local(**kwargs)

    if backend in ("pinecone", "pine"):
        return PineconeStore.connect(**kwargs)

    raise ValueError(
        f"Unknown vector store backend: {backend!r}. "
        "Choose from: 'qdrant', 'chroma', 'pinecone'."
    )