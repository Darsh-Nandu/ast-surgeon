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
    """Parse a Python source file and return one CodeChunk per logical unit.
 
    Logical units (in order of appearance):
      - Module-level docstring (if present)
      - Top-level functions
      - Top-level classes (as a whole chunk)
      - Methods within each class
    """
    source_bytes = source.encode("utf-8")
    tree = _PARSER.parse(source_bytes)
    root = tree.root_node
 
    chunks: list[CodeChunk] = []
 
    # 1. Module docstring
    mod_doc = _extract_module_docstring(root, source_bytes)
    if mod_doc:
        chunks.append(mod_doc._replace_file(file_path))
 
    # 2. Walk top-level children
    for node in root.children:
        if node.type == "function_definition":
            chunk = _extract_function(node, source_bytes, file_path, parent=None)
            chunks.append(chunk)
        elif node.type == "class_definition":
            class_chunks = _extract_class(node, source_bytes, file_path)
            chunks.extend(class_chunks)
        elif node.type == "decorated_definition":
            inner = _get_decorated_inner(node)
            if inner and inner.type == "function_definition":
                chunk = _extract_function(
                    inner, source_bytes, file_path,
                    parent=None,
                    decorator_node=node,
                )
                chunks.append(chunk)
            elif inner and inner.type == "class_definition":
                class_chunks = _extract_class(
                    inner, source_bytes, file_path, decorator_node=node
                )
                chunks.extend(class_chunks)
 
    chunks.sort(key=lambda c: c.start_line)
    return chunks

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

def _get_decorated_inner(node: ts.Node) -> Optional[ts.Node]:
    """Extract the function_definition or class_definition inside a decorated_definition."""
    for child in node.children:
        if child.type in ("function_definition", "class_definition"):
            return child
    return None

def _extract_decorators(decorator_node: Optional[ts.Node], source_bytes: bytes) -> list[str]:
    """Return list of decorator source strings (e.g. ['@property', '@staticmethod'])."""
    if decorator_node is None:
        return []
    decorators = []
    for child in decorator_node.children:
        if child.type == "decorator":
            decorators.append(_text(child, source_bytes).strip())
    return decorators

def _extract_docstring_from_body(body_node: ts.Node, source_bytes: bytes) -> Optional[str]:
    """Extract the first string literal from a function/class body as docstring."""
    for child in body_node.children:
        if child.type == "expression_statement":
            expr = child.children[0] if child.children else None
            if expr and expr.type in ("string", "concatenated_string"):
                raw = _text(expr, source_bytes)
                # Strip quotes
                cleaned = re.sub(r'^[rRuU]*[\'\"]{3}|[\'\"]{3}$', '', raw)
                cleaned = re.sub(r"^[rRuU]*[\'\"]{1}|[\'\"]{1}$", '', cleaned)
                return cleaned.strip()
        break  # docstring must be first statement in body
    return None

def _extract_calls(body_node: ts.Node, source_bytes: bytes) -> list[str]:
    """Walk the body AST and collect all function call names.
 
    DESIGN NOTE: we only collect the *callee name* (or attribute access like
    'self.method'), not the full call expression. This is sufficient for the
    dependency graph and avoids embedding large call-argument strings.
    """
    calls: list[str] = []
    _walk_calls(body_node, source_bytes, calls)
    # Deduplicate while preserving order
    seen = set()
    unique = []
    for c in calls:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

def _walk_calls(node: ts.Node, source_bytes: bytes, out: list[str]) -> None:
    if node.type == "call":
        func_node = node.child_by_field_name("function")
        if func_node:
            name = _text(func_node, source_bytes)
            # Keep only simple names and attribute access (not complex expressions)
            if re.match(r'^[\w.]+$', name):
                out.append(name)
    for child in node.children:
        _walk_calls(child, source_bytes, out)

def _extract_function(
    node: ts.Node,
    source_bytes: bytes,
    file_path: str,
    parent: Optional[str],
    decorator_node: Optional[ts.Node] = None,
) -> CodeChunk:
    """Build a CodeChunk from a function_definition node."""
    name_node = node.child_by_field_name("name")
    func_name = _text(name_node, source_bytes) if name_node else "<anonymous>"
    qualified = f"{parent}.{func_name}" if parent else func_name
 
    # Use the outer decorated_definition if present so content includes decorators
    source_node = decorator_node if decorator_node else node
    content = _text(source_node, source_bytes)
    start, end = _node_lines(source_node)
 
    body = node.child_by_field_name("body")
    docstring = _extract_docstring_from_body(body, source_bytes) if body else None
    calls = _extract_calls(body, source_bytes) if body else []
    decorators = _extract_decorators(decorator_node, source_bytes)
 
    chunk_type = ChunkType.METHOD if parent else ChunkType.FUNCTION
 
    return CodeChunk(
        file_path=file_path,
        language="python",
        chunk_type=chunk_type,
        name=qualified,
        content=content,
        content_hash=CodeChunk.hash_content(content),
        start_line=start,
        end_line=end,
        decorators=decorators,
        docstring=docstring,
        calls=calls,
        parent=parent,
    )

def _extract_class(
    node: ts.Node,
    source_bytes: bytes,
    file_path: str,
    decorator_node: Optional[ts.Node] = None,
) -> list[CodeChunk]:
    """Build CodeChunks for a class and all its methods.
 
    Returns [class_chunk, method_chunk, method_chunk, ...].
    """
    chunks: list[CodeChunk] = []
 
    name_node = node.child_by_field_name("name")
    class_name = _text(name_node, source_bytes) if name_node else "<anonymous_class>"
 
    source_node = decorator_node if decorator_node else node
    content = _text(source_node, source_bytes)
    start, end = _node_lines(source_node)
 
    body = node.child_by_field_name("body")
    class_docstring = _extract_docstring_from_body(body, source_bytes) if body else None
    decorators = _extract_decorators(decorator_node, source_bytes)
 
    # Class-level chunk (the whole class)
    chunks.append(CodeChunk(
        file_path=file_path,
        language="python",
        chunk_type=ChunkType.CLASS,
        name=class_name,
        content=content,
        content_hash=CodeChunk.hash_content(content),
        start_line=start,
        end_line=end,
        decorators=decorators,
        docstring=class_docstring,
        calls=[],
        parent=None,
    ))
 
    # Method chunks
    if body:
        for child in body.children:
            if child.type == "function_definition":
                method_chunk = _extract_function(
                    child, source_bytes, file_path, parent=class_name
                )
                chunks.append(method_chunk)
            elif child.type == "decorated_definition":
                inner = _get_decorated_inner(child)
                if inner and inner.type == "function_definition":
                    method_chunk = _extract_function(
                        inner, source_bytes, file_path,
                        parent=class_name,
                        decorator_node=child,
                    )
                    chunks.append(method_chunk)
 
    return chunks