"""
Sentence-aware text chunker for non-code files.

DESIGN NOTE on overlap:
  We add a 1-sentence overlap between consecutive chunks. This prevents a
  relevant concept that spans a paragraph boundary from being silently split
  across two chunks that both score low in retrieval.
"""

from __future__ import annotations

import re
from .models import ChunkType, CodeChunk

# Approximate character budget per chunk (not tokens - we avoid a tokenizer
# dependency here; 1500 chars ≈ 400 tokens for typical prose).
_MAX_CHARS = 1500
_MIN_CHARS = 200      # don't emit tiny trailing fragments


def chunk_text(file_path: str, source: str, language: str = "text") -> list[CodeChunk]:
    """
    Split a prose document into overlapping paragraph-level chunks.e.
    """
    paragraphs = _split_paragraphs(source)
    if not paragraphs:
        return []

    # Merge paragraphs into budget-respecting windows
    windows = _merge_into_windows(paragraphs, _MAX_CHARS)

    chunks = []
    for i, (text, start_line, end_line) in enumerate(windows):
        if len(text.strip()) < 10:
            continue
        chunks.append(CodeChunk(
            file_path=file_path,
            language=language,
            chunk_type=ChunkType.TEXT_BLOCK,
            name=f"block_{i}",
            content=text,
            content_hash=CodeChunk.hash_content(text),
            start_line=start_line,
            end_line=end_line,
            decorators=[],
            docstring=None,
            calls=[],
            parent=None,
        ))
    return chunks

def _split_paragraphs(source: str) -> list[tuple[str, int, int]]:
    """
    Split source into (text, start_line, end_line) paragraph tuples.

    A paragraph boundary is one or more blank lines. Fenced code blocks
    in markdown are kept as a single paragraph unit.
    """
    lines = source.splitlines()
    paragraphs: list[tuple[str, int, int]] = []

    current_lines: list[str] = []
    current_start = 1
    in_code_fence = False

    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_code_fence = not in_code_fence

        if not stripped and not in_code_fence:
            if current_lines:
                para_text = "\n".join(current_lines).strip()
                if para_text:
                    paragraphs.append((para_text, current_start, lineno - 1))
                current_lines = []
                current_start = lineno + 1
        else:
            if not current_lines:
                current_start = lineno
            current_lines.append(line)

    # Flush remaining
    if current_lines:
        para_text = "\n".join(current_lines).strip()
        if para_text:
            paragraphs.append((para_text, current_start, len(lines)))

    return paragraphs

def _merge_into_windows(
    paragraphs: list[tuple[str, int, int]],
    max_chars: int,
) -> list[tuple[str, int, int]]:
    """
    Merge consecutive paragraphs into windows respecting max_chars.
    """
    if not paragraphs:
        return []

    windows: list[tuple[str, int, int]] = []
    buffer: list[str] = []
    buf_chars = 0
    buf_start = paragraphs[0][1]
    buf_end = paragraphs[0][2]

    for text, start, end in paragraphs:
        para_chars = len(text)

        # If adding this paragraph exceeds budget, flush first
        if buf_chars + para_chars > max_chars and buffer:
            windows.append(("\n\n".join(buffer), buf_start, buf_end))
            # Overlap: keep last paragraph for context continuity
            buffer = [buffer[-1]]
            buf_chars = len(buffer[0])
            buf_start = start

        buffer.append(text)
        buf_chars += para_chars
        buf_end = end

    if buffer:
        windows.append(("\n\n".join(buffer), buf_start, buf_end))

    return windows