"""
AST-based chunker for JavaScript and TypeScript source files.

DESIGN NOTE on TypeScript:
  TypeScript interfaces and type aliases are NOT executable — they vanish at runtime.
  We still chunk them because they're retrieval-valuable: "what fields does UserDTO have?"
  is a real query. They get chunk_type=FUNCTION with name like "type:UserDTO".
"""

from __future__ import annotations

import re
from typing import Optional

import tree_sitter as ts
import tree_sitter_javascript as tsjs

from .models import ChunkType, CodeChunk


# Language + parser singletons

_JS_LANGUAGE = ts.Language(tsjs.language())
_JS_PARSER = ts.Parser(_JS_LANGUAGE)


# TypeScript grammar - optional; falls back to JS parser if not installed
try:
    import tree_sitter_typescript as tsts
    _TS_LANGUAGE = ts.Language(tsts.language_typescript())
    _TS_PARSER = ts.Parser(_TS_LANGUAGE)
    _TS_AVAILABLE = True
except Exception:
    _TS_LANGUAGE = _JS_LANGUAGE
    _TS_PARSER = _JS_PARSER
    _TS_AVAILABLE = False


def _get_parser(language: str) -> ts.Parser:
    if language == "typescript" and _TS_AVAILABLE:
        return _TS_PARSER
    return _JS_PARSER


# Public API

def chunk_js(file_path: str, source: str, language: str = "javascript") -> list[CodeChunk]:
    """
    Parse a JS/TS source file and return one CodeChunk per logical unit.
    """
    parser = _get_parser(language)
    source_bytes = source.encode("utf-8")
    tree = parser.parse(source_bytes)
    root = tree.root_node

    chunks: list[CodeChunk] = []

    for node in root.children:
        extracted = _dispatch_node(node, source_bytes, file_path, language, parent=None)
        chunks.extend(extracted)

    chunks.sort(key=lambda c: c.start_line)
    return chunks

# Node types that wrap a real declaration (export default, export named)
_EXPORT_WRAPPERS = {
    "export_statement",
    "export_default_declaration",
}

_FUNCTION_TYPES = {
    "function_declaration",
    "generator_function_declaration",
}

_CLASS_TYPES = {
    "class_declaration",
    "class",
}


def _dispatch_node(
    node: ts.Node,
    source_bytes: bytes,
    file_path: str,
    language: str,
    parent: Optional[str],
) -> list[CodeChunk]:
    """Route a top-level or class-body node to the right extractor."""

    if node.type in _FUNCTION_TYPES:
        return [_extract_function_decl(node, source_bytes, file_path, language, parent)]

    if node.type in _CLASS_TYPES:
        return _extract_class(node, source_bytes, file_path, language)

    if node.type == "lexical_declaration":
        return _try_extract_arrow(node, source_bytes, file_path, language, parent)

    if node.type in _EXPORT_WRAPPERS:
        return _extract_export(node, source_bytes, file_path, language)

    if node.type in ("interface_declaration", "type_alias_declaration"):
        return [_extract_ts_type(node, source_bytes, file_path, language)]

    return []

def _extract_function_decl(
    node: ts.Node,
    source_bytes: bytes,
    file_path: str,
    language: str,
    parent: Optional[str],
) -> CodeChunk:
    name_node = node.child_by_field_name("name")
    func_name = _text(name_node, source_bytes) if name_node else "<anonymous>"
    qualified = f"{parent}.{func_name}" if parent else func_name

    content = _text(node, source_bytes)
    start, end = _node_lines(node)
    body = node.child_by_field_name("body")
    calls = _extract_calls(body, source_bytes) if body else []
    docstring = _extract_jsdoc(node, source_bytes)

    return CodeChunk(
        file_path=file_path,
        language=language,
        chunk_type=ChunkType.METHOD if parent else ChunkType.FUNCTION,
        name=qualified,
        content=content,
        content_hash=CodeChunk.hash_content(content),
        start_line=start,
        end_line=end,
        decorators=[],
        docstring=docstring,
        calls=calls,
        parent=parent,
    )

def _try_extract_arrow(
    node: ts.Node,
    source_bytes: bytes,
    file_path: str,
    language: str,
    parent: Optional[str],
) -> list[CodeChunk]:
    """Extract const/let arrow functions from lexical_declaration nodes."""
    chunks = []
    for declarator in node.children:
        if declarator.type != "variable_declarator":
            continue
        value = declarator.child_by_field_name("value")
        if value is None or value.type not in ("arrow_function", "function"):
            continue
        name_node = declarator.child_by_field_name("name")
        func_name = _text(name_node, source_bytes) if name_node else "<anonymous>"
        qualified = f"{parent}.{func_name}" if parent else func_name

        content = _text(node, source_bytes)   # include the `const` keyword
        start, end = _node_lines(node)
        body = value.child_by_field_name("body")
        calls = _extract_calls(body, source_bytes) if body else []

        chunks.append(CodeChunk(
            file_path=file_path,
            language=language,
            chunk_type=ChunkType.METHOD if parent else ChunkType.FUNCTION,
            name=qualified,
            content=content,
            content_hash=CodeChunk.hash_content(content),
            start_line=start,
            end_line=end,
            decorators=[],
            docstring=_extract_jsdoc(node, source_bytes),
            calls=calls,
            parent=parent,
        ))
    return chunks

def _extract_class(
    node: ts.Node,
    source_bytes: bytes,
    file_path: str,
    language: str,
) -> list[CodeChunk]:
    chunks: list[CodeChunk] = []

    name_node = node.child_by_field_name("name")
    class_name = _text(name_node, source_bytes) if name_node else "<anonymous_class>"
    content = _text(node, source_bytes)
    start, end = _node_lines(node)

    chunks.append(CodeChunk(
        file_path=file_path,
        language=language,
        chunk_type=ChunkType.CLASS,
        name=class_name,
        content=content,
        content_hash=CodeChunk.hash_content(content),
        start_line=start,
        end_line=end,
        decorators=[],
        docstring=_extract_jsdoc(node, source_bytes),
        calls=[],
        parent=None,
    ))

    body = node.child_by_field_name("body")
    if body:
        for child in body.children:
            if child.type == "method_definition":
                method_name_node = child.child_by_field_name("name")
                method_name = _text(method_name_node, source_bytes) if method_name_node else "<method>"
                qualified = f"{class_name}.{method_name}"
                method_content = _text(child, source_bytes)
                ms, me = _node_lines(child)
                mbody = child.child_by_field_name("body")
                mcalls = _extract_calls(mbody, source_bytes) if mbody else []
                chunks.append(CodeChunk(
                    file_path=file_path,
                    language=language,
                    chunk_type=ChunkType.METHOD,
                    name=qualified,
                    content=method_content,
                    content_hash=CodeChunk.hash_content(method_content),
                    start_line=ms,
                    end_line=me,
                    decorators=[],
                    docstring=_extract_jsdoc(child, source_bytes),
                    calls=mcalls,
                    parent=class_name,
                ))

    return chunks

def _extract_export(
    node: ts.Node,
    source_bytes: bytes,
    file_path: str,
    language: str,
) -> list[CodeChunk]:
    """Unwrap export statements and delegate to inner declaration."""
    for child in node.children:
        if child.type in _FUNCTION_TYPES:
            return [_extract_function_decl(child, source_bytes, file_path, language, parent=None)]
        if child.type in _CLASS_TYPES:
            return _extract_class(child, source_bytes, file_path, language)
        if child.type == "lexical_declaration":
            return _try_extract_arrow(child, source_bytes, file_path, language, parent=None)
        if child.type in ("function", "arrow_function"):
            # export default () => { ... }  — anonymous
            content = _text(node, source_bytes)
            s, e = _node_lines(node)
            body = child.child_by_field_name("body")
            calls = _extract_calls(body, source_bytes) if body else []
            return [CodeChunk(
                file_path=file_path,
                language=language,
                chunk_type=ChunkType.FUNCTION,
                name="default",
                content=content,
                content_hash=CodeChunk.hash_content(content),
                start_line=s,
                end_line=e,
                decorators=[],
                docstring=None,
                calls=calls,
                parent=None,
            )]
    return []

def _extract_ts_type(
    node: ts.Node,
    source_bytes: bytes,
    file_path: str,
    language: str,
) -> CodeChunk:
    name_node = node.child_by_field_name("name")
    raw_name = _text(name_node, source_bytes) if name_node else "<type>"
    qualified = f"type:{raw_name}"
    content = _text(node, source_bytes)
    start, end = _node_lines(node)
    return CodeChunk(
        file_path=file_path,
        language=language,
        chunk_type=ChunkType.FUNCTION,  # treated as a logical unit
        name=qualified,
        content=content,
        content_hash=CodeChunk.hash_content(content),
        start_line=start,
        end_line=end,
        decorators=[],
        docstring=_extract_jsdoc(node, source_bytes),
        calls=[],
        parent=None,
    )

def _text(node: ts.Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte:node.end_byte].decode("utf-8")

def _node_lines(node: ts.Node) -> tuple[int, int]:
    return node.start_point[0] + 1, node.end_point[0] + 1

def _extract_calls(node: ts.Node, source_bytes: bytes) -> list[str]:
    calls: list[str] = []
    _walk_calls(node, source_bytes, calls)
    seen, unique = set(), []
    for c in calls:
        if c not in seen:
            seen.add(c)
            unique.append(c)
    return unique

def _walk_calls(node: ts.Node, source_bytes: bytes, out: list[str]) -> None:
    if node.type == "call_expression":
        func_node = node.child_by_field_name("function")
        if func_node:
            name = _text(func_node, source_bytes)
            if re.match(r'^[\w.]+$', name):
                out.append(name)
    for child in node.children:
        _walk_calls(child, source_bytes, out)

def _extract_jsdoc(node: ts.Node, source_bytes: bytes) -> Optional[str]:
    """Look for a JSDoc comment (/** ... */) immediately preceding the node."""
    # tree-sitter attaches comments as siblings; look at prev_named_sibling
    prev = node.prev_named_sibling
    if prev and prev.type == "comment":
        text = _text(prev, source_bytes)
        if text.startswith("/**"):
            return text.strip("/** \n").strip("*/").strip()
    return None