"""
Unified file chunker — routes source files to the right AST or prose chunker.

DESIGN NOTE on language detection:
  We detect by file extension, not by content sniffing. Content sniffing is
  fragile and slow. Extension-based detection is wrong ~0.1% of the time
  (e.g. a `.py` file that's actually a shell script), which is acceptable.
"""

from __future__ import annotations

from pathlib import Path

from .models import CodeChunk
from .python_chunker import chunk_python
from .js_chunker import chunk_js
from .text_chunker import chunk_text

# Extension → (chunker_fn, language_string)
_EXTENSION_MAP: dict[str, tuple] = {
    ".py":   (chunk_python, "python"),
    ".js":   (chunk_js,     "javascript"),
    ".jsx":  (chunk_js,     "javascript"),
    ".ts":   (chunk_js,     "typescript"),
    ".tsx":  (chunk_js,     "typescript"),
    ".md":   (chunk_text,   "markdown"),
    ".txt":  (chunk_text,   "text"),
    ".rst":  (chunk_text,   "text"),
}

def chunk_file(file_path: str, source: str) -> list[CodeChunk]:
    """
    Parse a source file and return its CodeChunks.
    """
    ext = Path(file_path).suffix.lower()
    entry = _EXTENSION_MAP.get(ext)

    if entry is None:
        # Unknown extension — treat as plain text so we don't silently drop it
        return chunk_text(file_path, source, language="text")

    chunker_fn, language = entry

    if chunker_fn is chunk_python:
        return chunker_fn(file_path, source)
    else:
        return chunker_fn(file_path, source, language)

def supported_extensions() -> list[str]:
    """Return the list of file extensions this chunker handles natively."""
    return list(_EXTENSION_MAP.keys())