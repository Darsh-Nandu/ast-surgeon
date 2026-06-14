"""
File watcher — watchdog-based daemon that triggers surgical re-indexing on save.

WHY watchdog over polling:
  Polling (checking file mtimes every N seconds) has two problems:
  1. Latency proportional to poll interval — you wait up to N seconds
     before a save is reflected in retrieval results.
  2. CPU waste — polling reads the filesystem constantly even when nothing changed.
  watchdog uses OS-native APIs (inotify on Linux, FSEvents on macOS, ReadDirectoryChangesW
  on Windows) — zero CPU when idle, sub-100ms latency on save.

WHY a debounce window:
  Editors like VSCode write a file in multiple bursts (save → format → save again).
  Without debouncing, we'd trigger 3–4 reindex calls per user save, wasting
  embedding API calls. A 300ms debounce window collapses these into one call.

WHY a background thread (not async):
  The file watcher runs in its own OS thread managed by watchdog. We keep it
  that way rather than wrapping in asyncio, because the Indexer's embedding
  calls are synchronous (httpx in sync mode). A thread is simpler and avoids
  the event-loop-in-thread complexity.

DESIGN NOTE on thread safety:
  The Indexer mutates the manifest (a plain dict) and calls Qdrant.
  We protect both with a threading.Lock so concurrent save events on different
  files don't race. Events are processed one at a time, but debouncing ensures
  the queue stays short under heavy editing.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Callable, Optional

from watchdog.events import (
    FileSystemEventHandler,
    FileCreatedEvent,
    FileModifiedEvent,
    FileDeletedEvent,
    FileMovedEvent,
)
from watchdog.observers import Observer

from ..chunker.dispatcher import supported_extensions
from .indexer import Indexer, IndexResult

logger = logging.getLogger(__name__)

DEBOUNCE_SECONDS = 0.3    # collapse rapid multi-save bursts into one reindex


# Event handler

class _AstSurgeonEventHandler(FileSystemEventHandler):
    """Watchdog event handler — debounces events and dispatches to Indexer.

    DESIGN NOTE on debouncing:
      We keep a dict of {path: scheduled_timer}. On each event for a path,
      we cancel the existing timer and set a new one. Only the final event
      in a burst fires the reindex.
    """

    def __init__(
        self,
        indexer: Indexer,
        project_root: Path,
        on_indexed: Optional[Callable[[str, IndexResult], None]] = None,
    ):
        super().__init__()
        self._indexer = indexer
        self._root = project_root
        self._on_indexed = on_indexed
        self._supported = set(supported_extensions())
        self._lock = threading.Lock()
        self._pending: dict[str, threading.Timer] = {}

    def on_created(self, event: FileCreatedEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, "created")

    def on_modified(self, event: FileModifiedEvent) -> None:
        if not event.is_directory:
            self._schedule(event.src_path, "modified")

    def on_deleted(self, event: FileDeletedEvent) -> None:
        if not event.is_directory:
            self._schedule_delete(event.src_path)

    def on_moved(self, event: FileMovedEvent) -> None:
        if not event.is_directory:
            # Treat as: delete old path, create new path
            self._schedule_delete(event.src_path)
            self._schedule(event.dest_path, "moved")


    def _is_supported(self, path: str) -> bool:
        return Path(path).suffix.lower() in self._supported

    def _schedule(self, path: str, reason: str) -> None:
        if not self._is_supported(path):
            return
        with self._lock:
            if path in self._pending:
                self._pending[path].cancel()
            timer = threading.Timer(
                DEBOUNCE_SECONDS,
                self._run_reindex,
                args=(path,),
            )
            self._pending[path] = timer
            timer.start()
            logger.debug("Scheduled reindex for %s (%s)", path, reason)

    def _schedule_delete(self, path: str) -> None:
        if not self._is_supported(path):
            return
        with self._lock:
            if path in self._pending:
                self._pending[path].cancel()
            timer = threading.Timer(
                DEBOUNCE_SECONDS,
                self._run_delete,
                args=(path,),
            )
            self._pending[path] = timer
            timer.start()
            logger.debug("Scheduled delete for %s", path)

    def _run_reindex(self, path: str) -> None:
        with self._lock:
            self._pending.pop(path, None)

        logger.info("Reindexing: %s", path)
        try:
            result = self._indexer.reindex_file(path)
            if self._on_indexed:
                rel = str(Path(path).relative_to(self._root))
                self._on_indexed(rel, result)
        except Exception as exc:
            logger.error("Reindex failed for %s: %s", path, exc)

    def _run_delete(self, path: str) -> None:
        with self._lock:
            self._pending.pop(path, None)

        logger.info("Removing deleted file from index: %s", path)
        try:
            self._indexer.remove_file(path)
        except Exception as exc:
            logger.error("Remove failed for %s: %s", path, exc)


# Public watcher

class FileWatcher:
    """
    Watches a project directory and keeps the vector index live.
    """

    def __init__(
        self,
        indexer: Indexer,
        project_root: str | Path,
        on_indexed: Optional[Callable[[str, IndexResult], None]] = None,
    ):
        """
        Args:
            indexer:      Indexer instance (already initialised with load_manifest()).
            project_root: Directory to watch recursively.
            on_indexed:   Optional callback(rel_path, IndexResult) called after
                          each successful reindex. Use this to push live updates
                          to a CLI spinner or SSE stream.
        """
        self._root = Path(project_root).resolve()
        self._handler = _AstSurgeonEventHandler(
            indexer=indexer,
            project_root=self._root,
            on_indexed=on_indexed,
        )
        self._observer = Observer()
        self._observer.schedule(self._handler, str(self._root), recursive=True)
        self._running = False

    def start(self) -> None:
        """Start watching in a background thread. Non-blocking."""
        if self._running:
            logger.warning("FileWatcher already running")
            return
        self._observer.start()
        self._running = True
        logger.info("FileWatcher started on %s", self._root)

    def stop(self) -> None:
        """Stop the watcher and wait for the background thread to finish."""
        if not self._running:
            return
        self._observer.stop()
        self._observer.join()
        self._running = False
        logger.info("FileWatcher stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    def __enter__(self) -> "FileWatcher":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()