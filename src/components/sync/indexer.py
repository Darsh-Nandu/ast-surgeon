"""
Indexer — full project indexing and surgical per-file re-indexing.

WHY two separate entry points (index_project vs reindex_file):
  index_project() is run once at `sovereign init`. It walks every file,
  chunks, embeds, upserts, and writes the manifest. It's designed to be
  resumable: files already in the manifest with matching hashes are skipped,
  so you can re-run it safely after a crash.

  reindex_file() is called by the file watcher on every save. It must be
  fast — ideally under 500ms for a typical source file. It achieves this by:
    1. Only re-embedding changed/new chunks (diff against manifest)
    2. Only deleting vectors for removed chunks (not the whole file)
    3. Never touching unchanged chunks at all

WHY we compute the diff here (not in the watcher):
  The watcher knows THAT a file changed. The indexer knows WHAT changed.
  Keeping diff logic here means the watcher stays a thin event forwarder
  and the indexer can be tested independently.

DESIGN NOTE on ignore patterns:
  We skip .git/, __pycache__/, node_modules/, and binary files by default.
  The ignore list is configurable via the Indexer constructor so projects
  can add their own patterns (e.g. "*.min.js", "dist/").
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..chunker.dispatcher import chunk_file, supported_extensions
from ..chunker.models import ChunkManifest, ChunkRecord, CodeChunk
from ..embeddings.pipeline import EmbeddingPipeline
from ..embeddings.providers import get_provider
from ..vectorstore.qdrant_store import VectorStore
from .manifest import ManifestStore

logger = logging.getLogger(__name__)

# Default directory/file patterns to skip during project walk
DEFAULT_IGNORE = {
    ".git", ".sovereign", "__pycache__", "node_modules",
    ".venv", "venv", "env", ".env", "dist", "build",
    ".next", ".nuxt", "coverage", ".pytest_cache", ".mypy_cache",
}


# Result types

@dataclass
class IndexResult:
    """Summary of one index_project() or reindex_file() call."""
    files_scanned: int = 0
    files_changed: int = 0
    chunks_added: int = 0
    chunks_deleted: int = 0
    chunks_unchanged: int = 0
    embed_errors: list[str] = field(default_factory=list)
    elapsed_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"IndexResult("
            f"files={self.files_scanned}, changed={self.files_changed}, "
            f"+{self.chunks_added}/-{self.chunks_deleted} chunks, "
            f"{self.elapsed_seconds:.2f}s)"
        )


@dataclass
class FileDiff:
    """What changed in a single file relative to the manifest.

    DESIGN NOTE: we diff by content_hash. A function that is renamed but
    unchanged in body gets treated as (deleted old name, added new name).
    This is correct — the vector for the old name should be removed.
    """
    to_add: list[CodeChunk]        # new or changed chunks - need embedding
    to_delete: list[str]           # vector_ids of stale chunks - need deletion
    unchanged: list[CodeChunk]     # same hash as manifest - skip



# Core Indexer

class Indexer:
    """Orchestrates chunking → embedding → Qdrant upsert + manifest sync.

    Usage:
        indexer = Indexer.create(project_root="/path/to/project")
        result = indexer.index_project()         # full initial index
        result = indexer.reindex_file("src/auth.py")  # on file save
    """

    def __init__(
        self,
        project_root: str | Path,
        store: VectorStore,
        pipeline: EmbeddingPipeline,
        manifest_store: ManifestStore,
        ignore_dirs: Optional[set[str]] = None,
    ):
        self._root = Path(project_root).resolve()
        self._store = store
        self._pipeline = pipeline
        self._ms = manifest_store
        self._ignore = ignore_dirs or DEFAULT_IGNORE
        self._manifest: ChunkManifest = {}


    # Factory
    @classmethod
    def create(
        cls,
        project_root: str | Path,
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        embedding_provider: Optional[str] = None,
        ignore_dirs: Optional[set[str]] = None,
    ) -> "Indexer":
        """Convenience factory — wires up all dependencies.

        Args:
            project_root:       Root directory of the project to index.
            qdrant_host/port:   Qdrant connection details.
            embedding_provider: "voyage"|"openai"|"local"|None (auto-detect).
            ignore_dirs:        Additional directory names to skip.
        """
        store = VectorStore.connect(host=qdrant_host, port=qdrant_port)
        store.ensure_collection()

        provider = get_provider(embedding_provider)
        pipeline = EmbeddingPipeline(provider=provider)
        manifest_store = ManifestStore(project_root)

        combined_ignore = DEFAULT_IGNORE | (ignore_dirs or set())

        return cls(
            project_root=project_root,
            store=store,
            pipeline=pipeline,
            manifest_store=manifest_store,
            ignore_dirs=combined_ignore,
        )


    # Public API
    def load_manifest(self) -> None:
        """Load the manifest from disk into memory. Call before indexing."""
        self._manifest = self._ms.load()
        stats = self._ms.stats(self._manifest)
        logger.info(
            "Manifest loaded: %d files, %d chunks",
            stats["files"], stats["chunks"]
        )

    def index_project(self, progress_cb=None) -> IndexResult:
        """
        Walk the entire project and index all supported files.
        """
        t0 = time.monotonic()
        self.load_manifest()

        # Collect all indexable files first so we can report total
        all_files = list(self._walk_project())
        result = IndexResult(files_scanned=len(all_files))

        logger.info("Starting full index of %d files in %s", len(all_files), self._root)

        for i, file_path in enumerate(all_files):
            rel_path = str(file_path.relative_to(self._root))
            try:
                file_result = self._index_one_file(file_path, rel_path)
                result.files_changed += file_result.files_changed
                result.chunks_added += file_result.chunks_added
                result.chunks_deleted += file_result.chunks_deleted
                result.chunks_unchanged += file_result.chunks_unchanged
                result.embed_errors.extend(file_result.embed_errors)
            except Exception as exc:
                logger.error("Failed to index %s: %s", rel_path, exc)
                result.embed_errors.append(f"{rel_path}: {exc}")

            if progress_cb:
                progress_cb(i + 1, len(all_files))

        result.elapsed_seconds = time.monotonic() - t0
        logger.info("Full index complete: %s", result)
        return result

    def reindex_file(self, file_path: str | Path) -> IndexResult:
        """
        Re-index a single file surgically.
        """
        t0 = time.monotonic()
        abs_path = self._root / file_path if not Path(file_path).is_absolute() else Path(file_path)
        rel_path = str(abs_path.relative_to(self._root))

        result = IndexResult(files_scanned=1)

        try:
            file_result = self._index_one_file(abs_path, rel_path)
            result.files_changed = file_result.files_changed
            result.chunks_added = file_result.chunks_added
            result.chunks_deleted = file_result.chunks_deleted
            result.chunks_unchanged = file_result.chunks_unchanged
            result.embed_errors = file_result.embed_errors
        except Exception as exc:
            logger.error("reindex_file failed for %s: %s", rel_path, exc)
            result.embed_errors.append(str(exc))

        result.elapsed_seconds = time.monotonic() - t0
        logger.debug("Reindex %s: %s", rel_path, result)
        return result

    def remove_file(self, file_path: str | Path) -> int:
        """
        Remove all vectors and manifest records for a deleted file.
        """
        abs_path = self._root / file_path if not Path(file_path).is_absolute() else Path(file_path)
        rel_path = str(abs_path.relative_to(self._root))

        deleted = self._store.delete_by_file(rel_path)
        self._ms.remove_file(self._manifest, rel_path)
        logger.info("Removed file %s: %d vectors deleted", rel_path, deleted)
        return deleted
    

    def _index_one_file(self, abs_path: Path, rel_path: str) -> IndexResult:
        """Index a single file — core logic shared by full and incremental index."""
        result = IndexResult(files_scanned=1)

        if not abs_path.exists():
            # File was deleted — clean up
            self.remove_file(abs_path)
            return result

        try:
            source = abs_path.read_text(encoding="utf-8", errors="replace")
        except Exception as exc:
            logger.warning("Cannot read %s: %s", rel_path, exc)
            return result

        # Chunk the file
        new_chunks = chunk_file(rel_path, source)

        if not new_chunks:
            # File has no indexable content (e.g. empty __init__.py)
            # Clean up any stale records from a previous version
            old_records = self._manifest.get(rel_path, [])
            if old_records:
                stale_ids = [r.vector_id for r in old_records]
                self._store.delete_by_ids(stale_ids)
                self._ms.remove_file(self._manifest, rel_path)
            return result

        # Diff against manifest
        diff = self._diff(rel_path, new_chunks)
        result.chunks_unchanged = len(diff.unchanged)

        # Delete stale vectors
        if diff.to_delete:
            self._store.delete_by_ids(diff.to_delete)
            result.chunks_deleted = len(diff.to_delete)

        # Embed and upsert new/changed chunks
        if diff.to_add:
            vectors, stats = self._pipeline.run(diff.to_add)
            result.embed_errors.extend(stats.failed_chunks)

            new_ids = self._store.upsert(diff.to_add, vectors)
            result.chunks_added = len(diff.to_add)

            # Assign vector_ids back onto chunks so we can build records
            for chunk, vid in zip(diff.to_add, new_ids):
                chunk.vector_id = vid

        # Build updated manifest records from ALL current chunks
        # (unchanged chunks keep their existing vector_id from manifest)
        all_current = diff.to_add + diff.unchanged
        new_records = self._build_records(rel_path, all_current)
        self._ms.update_file(self._manifest, rel_path, new_records)

        if diff.to_add or diff.to_delete:
            result.files_changed = 1
            logger.info(
                "Indexed %s: +%d/-%d chunks (%d unchanged)",
                rel_path, len(diff.to_add), len(diff.to_delete), len(diff.unchanged)
            )

        return result

    def _diff(self, rel_path: str, new_chunks: list[CodeChunk]) -> FileDiff:
        """Compute what changed by comparing new chunks against the manifest.

        DESIGN NOTE: we key by content_hash, not by name. This means:
        - A function body that changes → re-embedded (correct)
        - A function that's renamed but body unchanged → re-embedded (acceptable
          — the name is in the text that gets embedded anyway)
        - An unchanged function → skipped (the core surgical update property)
        """
        old_records = self._manifest.get(rel_path, [])

        # Build lookup: hash → vector_id for existing records
        old_hash_to_vid: dict[str, str] = {
            r.content_hash: r.vector_id for r in old_records
        }
        # Track which old hashes we've seen in the new chunks
        new_hashes: set[str] = {c.content_hash for c in new_chunks}

        to_add: list[CodeChunk] = []
        unchanged: list[CodeChunk] = []

        for chunk in new_chunks:
            if chunk.content_hash in old_hash_to_vid:
                # Restore the existing vector_id so we can build records
                chunk.vector_id = old_hash_to_vid[chunk.content_hash]
                unchanged.append(chunk)
            else:
                to_add.append(chunk)

        # Stale: old records whose hash no longer appears in new chunks
        to_delete: list[str] = [
            r.vector_id
            for r in old_records
            if r.content_hash not in new_hashes
        ]

        return FileDiff(to_add=to_add, to_delete=to_delete, unchanged=unchanged)

    def _build_records(
        self, rel_path: str, chunks: list[CodeChunk]
    ) -> list[ChunkRecord]:
        """Build ChunkRecord list from chunks that already have vector_ids set."""
        records = []
        for chunk in chunks:
            if chunk.vector_id is None:
                logger.warning(
                    "Chunk %s has no vector_id — skipping manifest record",
                    chunk.qualified_name()
                )
                continue
            records.append(ChunkRecord(
                name=chunk.name,
                content_hash=chunk.content_hash,
                vector_id=chunk.vector_id,
                chunk_type=chunk.chunk_type,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
            ))
        return records

    def _walk_project(self):
        """Yield all indexable file paths under the project root."""
        supported = set(supported_extensions())
        for path in self._root.rglob("*"):
            if path.is_file() and path.suffix.lower() in supported:
                # Skip ignored directories anywhere in the path
                if not any(part in self._ignore for part in path.parts):
                    yield path