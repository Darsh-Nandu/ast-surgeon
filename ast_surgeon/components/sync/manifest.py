"""
Manifest persistence — atomic JSON read/write of the ChunkManifest.

WHY this exists as a separate module:
  The manifest is the single source of truth for "what is already indexed."
  Every other sync operation (diff, re-embed, delete stale) reads from it.
  Keeping I/O isolated here means the Indexer never touches disk directly —
  making it trivial to swap JSON for SQLite later without touching business logic.

WHY atomic writes (write-then-rename):
  If the process dies mid-write we end up with a complete old manifest rather
  than a corrupt partial one. Python's os.replace() is atomic on POSIX and
  near-atomic on Windows (same drive).

Manifest file location:
  <project_root>/.ast-surgeon/manifest.json
  Hidden dot-folder so it doesn't clutter the project tree.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

from ..chunker.models import ChunkManifest, ChunkRecord, ChunkType

logger = logging.getLogger(__name__)

AST_SURGEON_DIR = ".ast-surgeon"
MANIFEST_FILE = "manifest.json"


class ManifestStore:
    """Persists and loads the ChunkManifest for a project.

    Usage:
        ms = ManifestStore(project_root="/path/to/project")
        manifest = ms.load()                    # {} on first run
        ms.save(manifest)                       # atomic write
        ms.update_file(manifest, path, records) # update one file in place + save
        ms.remove_file(manifest, path)          # file deleted → remove + save
    """

    def __init__(self, project_root: str | Path, manifest_path: str | Path | None = None):
        self._root = Path(project_root).resolve()
        self._ast_surgeon_dir = self._root / AST_SURGEON_DIR    
        if manifest_path is not None:
            # Allow callers to override the manifest location (e.g. per-session path)
            self._manifest_path = Path(manifest_path)
        else:
            self._manifest_path = self._ast_surgeon_dir / MANIFEST_FILE


    # Public API
    @property
    def manifest_path(self) -> Path:
        return self._manifest_path

    def load(self) -> ChunkManifest:
        """Load the manifest from disk. Returns empty dict on first run.

        DESIGN NOTE: we never raise on missing file — first run is valid.
        We DO raise on corrupt JSON so the caller knows something is wrong
        rather than silently re-indexing everything.
        """
        if not self._manifest_path.exists():
            logger.debug("No manifest found at %s — starting fresh", self._manifest_path)
            return {}

        try:
            raw = self._manifest_path.read_text(encoding="utf-8")
            data = json.loads(raw)
            return self._deserialise(data)
        except json.JSONDecodeError as exc:
            logger.error("Corrupt manifest at %s: %s — returning empty", self._manifest_path, exc)
            return {}

    def save(self, manifest: ChunkManifest) -> None:
        """Atomically write the manifest to disk.

        Creates .ast-surgeon/ if it doesn't exist.
        """
        self._ast_surgeon_dir.mkdir(parents=True, exist_ok=True)
        data = self._serialise(manifest)
        serialised = json.dumps(data, indent=2)

        # Atomic write: temp file in same directory, then rename
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=self._ast_surgeon_dir, suffix=".tmp", prefix="manifest_"
        )
        try:
            with os.fdopen(tmp_fd, "w", encoding="utf-8") as f:
                f.write(serialised)
            os.replace(tmp_path, self._manifest_path)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.debug(
            "Manifest saved: %d files, %d total chunks",
            len(manifest),
            sum(len(v) for v in manifest.values()),
        )

    def update_file(
        self,
        manifest: ChunkManifest,
        file_path: str,
        records: list[ChunkRecord],
    ) -> None:
        """Replace the records for one file and save.

        Args:
            manifest:  The in-memory manifest (mutated in place).
            file_path: Relative path string used as manifest key.
            records:   New list of ChunkRecords for this file.
        """
        if records:
            manifest[file_path] = records
        else:
            # File now has no indexable chunks — remove key entirely
            manifest.pop(file_path, None)
        self.save(manifest)

    def remove_file(self, manifest: ChunkManifest, file_path: str) -> None:
        """Remove a file's records from the manifest and save.

        Called when a file is deleted from disk.
        """
        if file_path in manifest:
            del manifest[file_path]
            self.save(manifest)
            logger.debug("Removed %s from manifest", file_path)

    def stats(self, manifest: ChunkManifest) -> dict:
        """Return a summary dict for logging / CLI display."""
        total_chunks = sum(len(v) for v in manifest.values())
        return {
            "files": len(manifest),
            "chunks": total_chunks,
            "manifest_path": str(self._manifest_path),
        }

    @staticmethod
    def _serialise(manifest: ChunkManifest) -> dict:
        """Convert ChunkManifest → plain JSON-serialisable dict."""
        out = {}
        for file_path, records in manifest.items():
            out[file_path] = [
                {
                    "name": r.name,
                    "content_hash": r.content_hash,
                    "vector_id": r.vector_id,
                    "chunk_type": r.chunk_type.value,
                    "start_line": r.start_line,
                    "end_line": r.end_line,
                }
                for r in records
            ]
        return out

    @staticmethod
    def _deserialise(data: dict) -> ChunkManifest:
        """Convert plain JSON dict → ChunkManifest."""
        manifest: ChunkManifest = {}
        for file_path, records_raw in data.items():
            manifest[file_path] = [
                ChunkRecord(
                    name=r["name"],
                    content_hash=r["content_hash"],
                    vector_id=r["vector_id"],
                    chunk_type=ChunkType(r["chunk_type"]),
                    start_line=r["start_line"],
                    end_line=r["end_line"],
                )
                for r in records_raw
            ]
        return manifest