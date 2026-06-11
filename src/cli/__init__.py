"""
sovereign.cli — command-line interface for Sovereign-Code.

Exports:
    app          — Typer CLI application (used by pyproject.toml entry point)
    ChatSession  — interactive REPL session (used programmatically in tests)

Entry point (pyproject.toml):
    [project.scripts]
    sovereign = "src.cli.main:app"
"""

from .main import app
from .session import ChatSession

__all__ = ["app", "ChatSession"]