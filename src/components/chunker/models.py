"""
Data models for AST-based code chunking.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class ChunkType(str, Enum):
    FUNCTION = "function"
    CLASS = "class"
    METHOD = "method"
    MODULE_DOCSTRING = "module_docstring"
    TEXT_BLOCK = "text_block"


@dataclass
class CodeChunk:

    # Identity
    file_path: str
    language: str 
    chunk_type: ChunkType

    # What it is
    name: Optional[str]
    content: str
    content_hash: str

    # Location
    start_line: int
    end_line: int

    # Semantic metadata
    decorators: list[str] = field(default_factory=list)
    docstring: Optional[str] = None
    calls: list[str] = field(default_factory=list)
    parent: Optional[str] = None

    # Set by vector embedding pipeline
    vector_id: Optional[str] = None


    @classmethod
    def hash_content(cls, content: str) -> str:
        return hashlib.sha256(content.strip().encode()).hexdigest()

    def qualified_name(self) -> str:
        name_part = self.name or f"<{self.chunk_type.value}>"
        return f"{self.file_path}::{name_part}"

    def __repr__(self) -> str:
        return (
            f"CodeChunk({self.chunk_type.value} {self.qualified_name()!r} "
            f"L{self.start_line}–{self.end_line})"
        )


@dataclass
class ChunkRecord:
    """ Lightweight Record """
    name: Optional[str]
    content_hash: str
    vector_id: str
    chunk_type: ChunkType
    start_line: int
    end_line: int


ChunkManifest = dict[str, list[ChunkRecord]]