"""
AST-based chunker for Python source files.
"""

from __future__ import annotations

import re
from typing import Optional

import tree_sitter_python as tspython
import tree_sitter as ts

from models import ChunkType, CodeChunk

_PY_LANGUAGE = ts.language(tspython.language())
_PARSER = ts.Parser(_PY_LANGUAGE)

# Public API

def chunk_python(file_path: str, source: str) -> list[CodeChunk]:
    """Parse a Python source file and return one CodeChunk per logical unit."""
    source_bytes = source.encode("utf-8")
    tree = _PARSER.parse(source_bytes)
    root = tree.root_node

    chunks: list[CodeChunk] = []

    # 1. Module docstring


class _ChunkBuilder:
    """Thin builder used internally so we can populate fields incrementally."""
    def __init__(self, **kwargs):
        self._data = kwargs

    def _replace_file(self, file_path: str) -> CodeChunk:
        self._data["file_path"] = file_path
        return CodeChunk(**self._data)

def _text(node: ts.Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

def _node_lines(node: ts.Node) -> tuple[int, int]:
    """Return 1-indexed (start_line, end_line)."""
    return node.start_point[0] + 1, node.end_point[0] + 1


def _extract_module_docstring(root: ts.Node, source_bytes: bytes) -> Optional[_ChunkBuilder]:
    """Return a chunk for the module-level docstring, or None."""
    for child in root.children:
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr and expr.type in ("string", "concatenated_string"):
                content = _text(child, source_bytes)
                start, end = _node_lines(child)
                return _ChunkBuilder(
                    file_path="",   # filled by caller
                    language="python",
                    chunk_type=ChunkType.MODULE_DOCSTRING,
                    name=None,
                    content=content,
                    content_hash=CodeChunk.hash_content(content),
                    start_line=start,
                    end_line=end,
                    decorators=[],
                    docstring=content.strip().strip('"""').strip("'''").strip(),
                    calls=[],
                    parent=None,
                )

        if child.type not in ("comment", "newline", "module_docstring"):
            break

    return None