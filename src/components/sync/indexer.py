"""
Indexer - full project indexing and surgical per-file re-indexing.

WHY two entry points (index_project vs reindex_file):
  index_project() is run once on first use. It walks every file, chunks,
  embeds, and upserts. It's resumable - files already in the manifest with
  matching hashes are skipped, so re-running after a crash is safe.

  reindex_file() is called by the FileWatcher on every save. It targets
  sub-500ms latency by:
    1. Only re-embedding changed/new chunks (diff against manifest)
    2. Only deleting vectors for removed chunks (not the whole file)
    3. Leaving unchanged chunks completely untouched

WHY store-agnostic:
  The Indexer now accepts any VectorStore implementation - Qdrant, ChromaDB,
  Pinecone, or your own. Swap backends without touching any sync logic.

WHY the ignore list is configurable:
  Every project has different generated dirs. The default covers the most
  common cases; users add project-specific patterns (e.g. "*.min.js", "dist/")
  via ignore_dirs in Indexer.create().
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ..chunker.dispatcher import chunk_file, supported_extensions
from ..chunker.models import ChunkManifest, ChunkRecord, CodeChunk
from ..embeddings.pipeline import EmbeddingPipeline
from ..embeddings.providers import get_provider
from ..vectorstore.base import VectorStore
from ..vectorstore import get_store
from .manifest import ManifestStore

logger = logging.getLogger(__name__)

DEFAULT_IGNORE: frozenset[str] = frozenset({
    ".git", ".ast-surgeon", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build",
    ".next", ".nuxt", "coverage", ".pytest_cache", ".mypy_cache",
    ".tox", ".eggs", "*.egg-info",
})


# Result types

@dataclass
class IndexResult:
    """Summary returned by index_project() or reindex_file()."""
    files_scanned:   int = 0
    files_changed:   int = 0
    chunks_added:    int = 0
    chunks_deleted:  int = 0
    chunks_unchanged: int = 0
    embed_errors:    list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"IndexResult(files={self.files_scanned}, changed={self.files_changed}, "
            f"+{self.chunks_added}/-{self.chunks_deleted} chunks, "
            f"{self.chunks_unchanged} unchanged, {self.elapsed_seconds:.2f}s)"
        )


@dataclass
class FileDiff:
    """
    What changed in a single file relative to the manifest.

    We diff by content_hash (not name), so:
    - Changed body      → chunk goes into to_add (correct: needs re-embedding)
    - Renamed function  → treated as delete old + add new (acceptable: name
                          is part of the embedded text anyway)
    - Unchanged body    → chunk goes into unchanged (never re-embedded)
    """
    to_add:    list[CodeChunk]   # new or changed - need embedding
    to_delete: list[str]         # stale vector_ids - need deletion
    unchanged: list[CodeChunk]   # same hash as manifest - skip


# Core Indexer

class Indexer:
    """
    Orchestrates: chunk → embed → vector store upsert + manifest sync.

    The simplest possible usage:
        indexer = Indexer.create("/path/to/project")   # chroma + local embeddings
        result  = indexer.index_project()
        result  = indexer.reindex_file("src/auth.py")  # surgical update on save

    With specific backends:
        indexer = Indexer.create(
            "/path/to/project",
            store_type="qdrant",
            embedding_provider="cohere",
        )

    BYO store or pipeline:
        store    = QdrantStore.connect(url="https://...")
        provider = get_provider("voyage")
        pipeline = EmbeddingPipeline(provider=provider)
        indexer  = Indexer(project_root, store=store, pipeline=pipeline)
    """

    def __init__(
        self,
        project_root: str | Path,
        store: VectorStore,
        pipeline: EmbeddingPipeline,
        manifest_store: ManifestStore,
        ignore_dirs: Optional[set[str]] = None,
    ):
        self._root     = Path(project_root).resolve()
        self._store    = store
        self._pipeline = pipeline
        self._ms       = manifest_store
        self._ignore   = frozenset(ignore_dirs or set()) | DEFAULT_IGNORE
        self._manifest: ChunkManifest = {}

    # Factory

    @classmethod
    def create(
        cls,
        project_root: str | Path,
        *,
        # Vector store
        store: Optional[VectorStore] = None,
        store_type: str = "chroma",
        store_kwargs: Optional[dict] = None,
        # Embeddings
        pipeline: Optional[EmbeddingPipeline] = None,
        embedding_provider: Optional[str] = None,
        embedding_kwargs: Optional[dict] = None,
        # Misc
        ignore_dirs: Optional[set[str]] = None,
        on_progress: Optional[Callable[[int, int], None]] = None,
    ) -> "Indexer":
        """
        Convenience factory - builds and wires all dependencies.

        Args:
            project_root:       Root directory of the codebase to index.
            store:              Pre-built VectorStore (overrides store_type).
            store_type:         "chroma" | "qdrant" | "pinecone" (default: chroma).
            store_kwargs:       Extra kwargs forwarded to the store factory.
            pipeline:           Pre-built EmbeddingPipeline (overrides provider args).
            embedding_provider: Provider name - see get_provider() for options.
            embedding_kwargs:   Extra kwargs forwarded to get_provider().
            ignore_dirs:        Extra directory names to skip during project walk.
            on_progress:        Callback(done, total) for progress reporting.

        Examples:
            # Zero config - chroma on disk + auto-detected embeddings
            Indexer.create("/my/project")

            # Specific provider + Qdrant
            Indexer.create("/my/project", store_type="qdrant", embedding_provider="openai")

            # Bring your own store
            Indexer.create("/my/project", store=QdrantStore.connect(url="https://..."))
        """
        # Build or accept the vector store
        if store is None:
            store = get_store(store_type, **(store_kwargs or {}))

        # Build or accept the pipeline
        if pipeline is None:
            provider = get_provider(embedding_provider, **(embedding_kwargs or {}))
            pipeline = EmbeddingPipeline(
                provider=provider,
                on_progress=on_progress,
            )

        # Initialise the collection with the provider's actual dimension
        store.ensure_collection(dim=pipeline.provider.dimension)

        manifest_store = ManifestStore(project_root)
        combined_ignore = (ignore_dirs or set()) | set(DEFAULT_IGNORE)

        return cls(
            project_root=project_root,
            store=store,
            pipeline=pipeline,
            manifest_store=manifest_store,
            ignore_dirs=combined_ignore,
        )

    # Public API

    def load_manifest(self) -> None:
        """Load the manifest from disk into memory. Called automatically by index_project."""
        self._manifest = self._ms.load()
        stats = self._ms.stats(self._manifest)
        logger.info(
            "Manifest loaded: %d files, %d chunks", stats["files"], stats["chunks"]
        )

    def index_project(
        self,
        progress_cb: Optional[Callable[[int, int], None]] = None,
    ) -> IndexResult:
        """
        Walk the entire project and index all supported files.

        Resumable - files already indexed with matching hashes are skipped.
        Safe to re-run after a crash.

        Args:
            progress_cb: Optional callback(files_done, files_total).

        Returns:
            IndexResult with full statistics.
        """
        t0 = time.monotonic()
        self.load_manifest()

        all_files = list(self._walk_project())
        result = IndexResult(files_scanned=len(all_files))

        logger.info("Indexing %d files in %s", len(all_files), self._root)

        for i, file_path in enumerate(all_files):
            rel_path = str(file_path.relative_to(self._root))
            try:
                fr = self._index_one_file(file_path, rel_path)
                result.files_changed    += fr.files_changed
                result.chunks_added     += fr.chunks_added
                result.chunks_deleted   += fr.chunks_deleted
                result.chunks_unchanged += fr.chunks_unchanged
                result.embed_errors.extend(fr.embed_errors)
            except Exception as exc:
                logger.error("Failed to index %s: %s", rel_path, exc)
                result.embed_errors.append(f"{rel_path}: {exc}")

            if progress_cb:
                progress_cb(i + 1, len(all_files))

        result.elapsed_seconds = time.monotonic() - t0
        logger.info("Index complete: %s", result)
        return result

    def reindex_file(self, file_path: str | Path) -> IndexResult:
        """
        Surgically re-index a single file.

        Only re-embeds chunks whose content hash changed. Unchanged chunks
        are never touched. Target latency: <500ms for a typical source file.

        Args:
            file_path: Absolute or project-relative path to the changed file.

        Returns:
            IndexResult for this single file.
        """
        t0 = time.monotonic()
        abs_path = (
            Path(file_path) if Path(file_path).is_absolute()
            else self._root / file_path
        )
        rel_path = str(abs_path.relative_to(self._root))

        result = IndexResult(files_scanned=1)
        try:
            fr = self._index_one_file(abs_path, rel_path)
            result.files_changed    = fr.files_changed
            result.chunks_added     = fr.chunks_added
            result.chunks_deleted   = fr.chunks_deleted
            result.chunks_unchanged = fr.chunks_unchanged
            result.embed_errors     = fr.embed_errors
        except Exception as exc:
            logger.error("reindex_file failed for %s: %s", rel_path, exc)
            result.embed_errors.append(str(exc))

        result.elapsed_seconds = time.monotonic() - t0
        logger.debug("Reindex %s: %s", rel_path, result)
        return result

    def remove_file(self, file_path: str | Path) -> int:
        """
        Remove all vectors and manifest records for a deleted file.

        Returns the number of vectors deleted.
        """
        abs_path = (
            Path(file_path) if Path(file_path).is_absolute()
            else self._root / file_path
        )
        rel_path = str(abs_path.relative_to(self._root))

        deleted = self._store.delete_by_file(rel_path)
        self._ms.remove_file(self._manifest, rel_path)
        logger.info("Removed file %s (%d vectors deleted)", rel_path, deleted)
        return deleted

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_language: Optional[str] = None,
        filter_file: Optional[str] = None,
    ):
        """
        Embed a query and search the vector store.

        Args:
            query:           Natural-language or code search query.
            top_k:           Max results to return.
            filter_language: Restrict to a specific language ("python", "typescript" …).
            filter_file:     Restrict to a specific file path.

        Returns:
            List of SearchResult sorted by relevance.
        """
        vec = self._pipeline.embed_query(query)
        return self._store.search(
            vec, top_k=top_k,
            filter_language=filter_language,
            filter_file=filter_file,
        )

    # Internal

    def _index_one_file(self, abs_path: Path, rel_path: str) -> IndexResult:
        result = IndexResult(files_scanned=1)

        if not abs_path.exists():
            self.remove_file(abs_path)
            return result

        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Cannot read %s: %s", rel_path, exc)
            return result

        new_chunks = chunk_file(rel_path, source)

        # If the file is now empty/unsupported, clean up any stale records
        if not new_chunks:
            old_records = self._manifest.get(rel_path, [])
            if old_records:
                self._store.delete_by_ids([r.vector_id for r in old_records])
                self._ms.remove_file(self._manifest, rel_path)
            return result

        diff = self._diff(rel_path, new_chunks)
        result.chunks_unchanged = len(diff.unchanged)

        if diff.to_delete:
            self._store.delete_by_ids(diff.to_delete)
            result.chunks_deleted = len(diff.to_delete)

        if diff.to_add:
            vectors, stats = self._pipeline.run(diff.to_add)
            result.embed_errors.extend(stats.failed_chunks)
            new_ids = self._store.upsert(diff.to_add, vectors)
            result.chunks_added = len(diff.to_add)
            for chunk, vid in zip(diff.to_add, new_ids):
                chunk.vector_id = vid

        all_current  = diff.to_add + diff.unchanged
        new_records  = self._build_records(rel_path, all_current)
        self._ms.update_file(self._manifest, rel_path, new_records)

        if diff.to_add or diff.to_delete:
            result.files_changed = 1
            logger.info(
                "Indexed %s: +%d/-%d chunks (%d unchanged)",
                rel_path, len(diff.to_add), len(diff.to_delete), len(diff.unchanged),
            )
        return result

    def _diff(self, rel_path: str, new_chunks: list[CodeChunk]) -> FileDiff:
        old_records = self._manifest.get(rel_path, [])
        old_hash_to_vid: dict[str, str] = {r.content_hash: r.vector_id for r in old_records}
        new_hashes: set[str] = {c.content_hash for c in new_chunks}

        to_add: list[CodeChunk] = []
        unchanged: list[CodeChunk] = []

        for chunk in new_chunks:
            if chunk.content_hash in old_hash_to_vid:
                chunk.vector_id = old_hash_to_vid[chunk.content_hash]
                unchanged.append(chunk)
            else:
                to_add.append(chunk)

        to_delete = [
            r.vector_id
            for r in old_records
            if r.content_hash not in new_hashes
        ]
        return FileDiff(to_add=to_add, to_delete=to_delete, unchanged=unchanged)

    def _build_records(
        self, rel_path: str, chunks: list[CodeChunk]
    ) -> list[ChunkRecord]:
        records = []
        for chunk in chunks:
            if chunk.vector_id is None:
                logger.warning(
                    "Chunk %s has no vector_id - skipping manifest record",
                    chunk.qualified_name(),
                )
                continue
            records.append(ChunkRecord(
                name         = chunk.name,
                content_hash = chunk.content_hash,
                vector_id    = chunk.vector_id,
                chunk_type   = chunk.chunk_type,
                start_line   = chunk.start_line,
                end_line     = chunk.end_line,
            ))
        return records

    def _walk_project(self):
        """Yield all indexable file paths under the project root."""
        supported = set(supported_extensions())
        for path in self._root.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in supported:
                continue
            if any(part in self._ignore for part in path.parts):
                continue
            yield path