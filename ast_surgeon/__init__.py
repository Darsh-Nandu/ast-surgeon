"""
ast-surgeon - AST-based code chunking, embedding, and semantic search.

Quick start:
    from ast_surgeon import Indexer

    indexer = Indexer.create("/path/to/project")   # chroma + auto-detected embeddings
    indexer.index_project()
    hits = indexer.search("how does authentication work?")
    for hit in hits:
        print(hit.score, hit.chunk.qualified_name())

Live-updating index (re-embeds on file save):
    from ast_surgeon import Indexer, FileWatcher

    indexer = Indexer.create("/path/to/project")
    indexer.index_project()

    with FileWatcher(indexer, "/path/to/project"):
        ...  # index stays in sync while this block runs

Chunking only (no embeddings/vector store needed):
    from ast_surgeon import chunk_file

    chunks = chunk_file("auth.py", source_code)

Choosing providers and stores explicitly:
    from ast_surgeon import Indexer, get_provider, get_store

    indexer = Indexer.create(
        "/path/to/project",
        store_type="qdrant",
        embedding_provider="cohere",
    )
"""

from __future__ import annotations

from .components.chunker import (
    ChunkType,
    CodeChunk,
    ChunkRecord,
    ChunkManifest,
    chunk_file,
    chunk_python,
    chunk_js,
    chunk_text,
    supported_extensions,
)
from .components.embeddings import (
    EmbeddingProvider,
    EmbeddingError,
    EmbeddingPipeline,
    EmbedStats,
    get_provider,
    list_providers,
)
from .components.vectorstore import (
    VectorStore,
    SearchResult,
    ChromaStore,
    QdrantStore,
    PineconeStore,
    get_store,
)
from .components.sync import (
    ManifestStore,
    Indexer,
    IndexResult,
    FileDiff,
    FileWatcher,
)

__version__ = "0.4.0"

__all__ = [
    "__version__",
    # Chunking
    "ChunkType",
    "CodeChunk",
    "ChunkRecord",
    "ChunkManifest",
    "chunk_file",
    "chunk_python",
    "chunk_js",
    "chunk_text",
    "supported_extensions",
    # Embeddings
    "EmbeddingProvider",
    "EmbeddingError",
    "EmbeddingPipeline",
    "EmbedStats",
    "get_provider",
    "list_providers",
    # Vector stores
    "VectorStore",
    "SearchResult",
    "ChromaStore",
    "QdrantStore",
    "PineconeStore",
    "get_store",
    # Sync / indexing
    "ManifestStore",
    "Indexer",
    "IndexResult",
    "FileDiff",
    "FileWatcher",
]