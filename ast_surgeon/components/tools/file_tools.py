"""
File system tools — the agent's hands.

WHY these are structured as Tool objects (not plain functions):
  Phase 4's deep agent loop needs to reason about available tools, pick one,
  call it, observe the result, and decide the next step. That requires a
  uniform interface: every tool has a name, description, parameter schema,
  and a run() method that returns a ToolResult.

  This is the same pattern Claude Code, Devin, and SWE-agent use internally.
  The agent loop (Phase 4) will never call OS functions directly — it always
  goes through tools, which means we can intercept, log, sandbox, and replay
  every action the agent takes.

WHY explicit ToolResult with is_error:
  The agent loop needs to know whether to retry, back off, or surface the
  error to the user. A bare exception would blow up the loop. A ToolResult
  with is_error=True lets the loop reason: "this action failed, here is why,
  what should I do next?" — just like a human reading an error message.

DESIGN NOTE on sandboxing:
  All path operations resolve against a project_root and refuse to escape it
  (no ../../etc/passwd). This is enforced in _safe_path(). The run_command
  tool also has a configurable allowlist/timeout so the agent can't run
  arbitrary destructive commands without explicit permission.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# ToolResult - uniform return type for every tool

@dataclass
class ToolResult:
    """What a tool returns to the agent loop.

    The loop reads `is_error` to decide whether to retry or continue.
    `content` is the text/data shown to the LLM as the observation.
    `metadata` carries structured data (line counts, exit codes, etc.)
    that the loop can use without parsing `content`.
    """
    content: str
    is_error: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        prefix = "ERROR: " if self.is_error else ""
        return f"{prefix}{self.content}"


# Base tool

class Tool:
    """Abstract base for all agent tools."""

    name: str = ""
    description: str = ""

    def __init__(self, project_root: str | Path):
        self._root = Path(project_root).resolve()

    def run(self, **kwargs) -> ToolResult:
        raise NotImplementedError

    def _safe_path(self, path: str) -> Path:
        """Resolve path relative to project root and reject escape attempts.

        DESIGN NOTE: we always resolve to an absolute path first, then check
        that it's inside the root. This catches symlinks and .. traversal.
        """
        resolved = (self._root / path).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise PermissionError(
                f"Path {path!r} escapes project root {self._root}"
            )
        return resolved

    @property
    def schema(self) -> dict:
        """JSON schema for this tool's parameters. Used by the agent loop."""
        raise NotImplementedError


# ReadFile

class ReadFileTool(Tool):
    """Read a file and return its contents, optionally with line numbers.

    WHY line numbers:
      The agent frequently needs to reference specific lines ("edit line 42").
      Returning numbered output makes those references unambiguous without
      requiring a separate 'get line number' tool call.
    """

    name = "read_file"
    description = (
        "Read the contents of a file. Returns the file content with optional "
        "line numbers. Use start_line/end_line to read a specific range."
    )

    def run(
        self,
        path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None,
        show_line_numbers: bool = True,
    ) -> ToolResult:
        try:
            safe = self._safe_path(path)
            if not safe.exists():
                return ToolResult(f"File not found: {path}", is_error=True)
            if not safe.is_file():
                return ToolResult(f"Not a file: {path}", is_error=True)

            content = safe.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            total = len(lines)

            # Apply range
            s = (start_line - 1) if start_line else 0
            e = end_line if end_line else total
            s = max(0, min(s, total))
            e = max(s, min(e, total))
            selected = lines[s:e]

            if show_line_numbers:
                width = len(str(e))
                numbered = "\n".join(
                    f"{s + i + 1:>{width}} │ {line}"
                    for i, line in enumerate(selected)
                )
                output = numbered
            else:
                output = "\n".join(selected)

            return ToolResult(
                content=output,
                metadata={
                    "path": path,
                    "total_lines": total,
                    "shown_lines": len(selected),
                    "start_line": s + 1,
                    "end_line": e,
                },
            )
        except PermissionError as e:
            return ToolResult(str(e), is_error=True)
        except Exception as e:
            return ToolResult(f"Error reading {path}: {e}", is_error=True)

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "path": {"type": "string", "description": "File path relative to project root"},
                "start_line": {"type": "integer", "description": "First line to read (1-indexed)", "optional": True},
                "end_line": {"type": "integer", "description": "Last line to read (inclusive)", "optional": True},
                "show_line_numbers": {"type": "boolean", "description": "Prefix each line with its number", "optional": True},
            },
        }


# WriteFile

class WriteFileTool(Tool):
    """Write content to a file. Creates parent directories if needed.

    WHY we track a write log:
      The agent session needs to know what it changed so we can show a diff
      summary at the end, support undo, and feed the file watcher.
    """

    name = "write_file"
    description = (
        "Write content to a file. Creates the file and any parent directories "
        "if they don't exist. Overwrites existing content."
    )

    def __init__(self, project_root: str | Path):
        super().__init__(project_root)
        self._write_log: list[dict] = []   # audit trail for the session

    def run(self, path: str, content: str) -> ToolResult:
        try:
            safe = self._safe_path(path)
            safe.parent.mkdir(parents=True, exist_ok=True)

            existed = safe.exists()
            old_content = safe.read_text(encoding="utf-8") if existed else None

            safe.write_text(content, encoding="utf-8")

            self._write_log.append({
                "path": path,
                "timestamp": time.time(),
                "created": not existed,
                "lines": len(content.splitlines()),
            })

            action = "Created" if not existed else "Updated"
            return ToolResult(
                content=f"{action} {path} ({len(content.splitlines())} lines)",
                metadata={
                    "path": path,
                    "created": not existed,
                    "lines": len(content.splitlines()),
                    "bytes": len(content.encode()),
                },
            )
        except PermissionError as e:
            return ToolResult(str(e), is_error=True)
        except Exception as e:
            return ToolResult(f"Error writing {path}: {e}", is_error=True)

    @property
    def write_log(self) -> list[dict]:
        return list(self._write_log)

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "path": {"type": "string", "description": "File path relative to project root"},
                "content": {"type": "string", "description": "Full content to write"},
            },
        }


# EditFile - surgical line-range replacement (preferred over full rewrite)

class EditFileTool(Tool):
    """Replace a specific line range in a file.

    WHY this exists alongside WriteFile:
      Full rewrites are risky — one bad LLM generation overwrites everything.
      Surgical edits are safer: replace lines 42-48 with new content, leave
      the rest untouched. This is how Claude Code and Cursor apply diffs.
    """

    name = "edit_file"
    description = (
        "Replace a range of lines in an existing file. "
        "Use this for targeted edits instead of rewriting the whole file."
    )

    def run(
        self,
        path: str,
        start_line: int,
        end_line: int,
        new_content: str,
    ) -> ToolResult:
        try:
            safe = self._safe_path(path)
            if not safe.exists():
                return ToolResult(f"File not found: {path}", is_error=True)

            lines = safe.read_text(encoding="utf-8").splitlines(keepends=True)
            total = len(lines)

            if start_line < 1 or end_line > total or start_line > end_line:
                return ToolResult(
                    f"Invalid range {start_line}-{end_line} for file with {total} lines",
                    is_error=True,
                )

            new_lines = new_content.splitlines(keepends=True)
            # Ensure last line has newline
            if new_lines and not new_lines[-1].endswith("\n"):
                new_lines[-1] += "\n"

            replaced = lines[: start_line - 1] + new_lines + lines[end_line:]
            safe.write_text("".join(replaced), encoding="utf-8")

            return ToolResult(
                content=(
                    f"Edited {path}: replaced lines {start_line}-{end_line} "
                    f"with {len(new_lines)} line(s)"
                ),
                metadata={
                    "path": path,
                    "start_line": start_line,
                    "end_line": end_line,
                    "lines_replaced": end_line - start_line + 1,
                    "lines_inserted": len(new_lines),
                },
            )
        except PermissionError as e:
            return ToolResult(str(e), is_error=True)
        except Exception as e:
            return ToolResult(f"Error editing {path}: {e}", is_error=True)

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "path": {"type": "string"},
                "start_line": {"type": "integer", "description": "First line to replace (1-indexed)"},
                "end_line": {"type": "integer", "description": "Last line to replace (inclusive)"},
                "new_content": {"type": "string", "description": "Replacement content"},
            },
        }


# ListDir

class ListDirTool(Tool):
    """List directory contents as a tree."""

    name = "list_dir"
    description = (
        "List the contents of a directory. Returns a tree view with file sizes. "
        "Use depth to control how many levels to show."
    )

    IGNORE = {".git", "__pycache__", ".sovereign", "node_modules", ".venv", "venv"}

    def run(self, path: str = ".", depth: int = 2) -> ToolResult:
        try:
            safe = self._safe_path(path)
            if not safe.exists():
                return ToolResult(f"Path not found: {path}", is_error=True)
            if not safe.is_dir():
                return ToolResult(f"Not a directory: {path}", is_error=True)

            lines = [f"{path}/"]
            self._tree(safe, lines, prefix="", depth=depth, current=0)

            return ToolResult(
                content="\n".join(lines),
                metadata={"path": path, "depth": depth},
            )
        except PermissionError as e:
            return ToolResult(str(e), is_error=True)
        except Exception as e:
            return ToolResult(f"Error listing {path}: {e}", is_error=True)

    def _tree(self, directory: Path, out: list, prefix: str, depth: int, current: int):
        if current >= depth:
            return
        try:
            entries = sorted(directory.iterdir(), key=lambda p: (p.is_file(), p.name))
        except PermissionError:
            return

        entries = [e for e in entries if e.name not in self.IGNORE]
        for i, entry in enumerate(entries):
            is_last = i == len(entries) - 1
            connector = "└── " if is_last else "├── "
            if entry.is_dir():
                out.append(f"{prefix}{connector}{entry.name}/")
                extension = "    " if is_last else "│   "
                self._tree(entry, out, prefix + extension, depth, current + 1)
            else:
                size = entry.stat().st_size
                size_str = self._fmt_size(size)
                out.append(f"{prefix}{connector}{entry.name} ({size_str})")

    @staticmethod
    def _fmt_size(size: int) -> str:
        for unit in ("B", "KB", "MB", "GB"):
            if size < 1024:
                return f"{size:.0f}{unit}"
            size /= 1024
        return f"{size:.1f}GB"

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "path": {"type": "string", "description": "Directory path (default: project root)", "optional": True},
                "depth": {"type": "integer", "description": "Tree depth (default: 2)", "optional": True},
            },
        }


# SearchFiles - grep-style text search across the project

class SearchFilesTool(Tool):
    """Search for a pattern across all project files."""

    name = "search_files"
    description = (
        "Search for a text pattern (regex) across all files in the project. "
        "Returns matching lines with file path and line number."
    )

    def run(
        self,
        pattern: str,
        path: str = ".",
        file_glob: str = "*",
        max_results: int = 50,
    ) -> ToolResult:
        import re
        try:
            safe = self._safe_path(path)
            try:
                regex = re.compile(pattern)
            except re.error as e:
                return ToolResult(f"Invalid regex pattern: {e}", is_error=True)

            results = []
            searched = 0
            for file_path in safe.rglob(file_glob):
                if not file_path.is_file():
                    continue
                if any(p in file_path.parts for p in ListDirTool.IGNORE):
                    continue
                try:
                    lines = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
                    searched += 1
                    for lineno, line in enumerate(lines, 1):
                        if regex.search(line):
                            rel = str(file_path.relative_to(self._root))
                            results.append(f"{rel}:{lineno}: {line.rstrip()}")
                            if len(results) >= max_results:
                                break
                except Exception:
                    continue
                if len(results) >= max_results:
                    break

            if not results:
                return ToolResult(
                    f"No matches for {pattern!r} in {searched} files",
                    metadata={"matches": 0, "files_searched": searched},
                )

            truncated = len(results) >= max_results
            content = "\n".join(results)
            if truncated:
                content += f"\n... (truncated at {max_results} results)"

            return ToolResult(
                content=content,
                metadata={
                    "matches": len(results),
                    "files_searched": searched,
                    "truncated": truncated,
                },
            )
        except PermissionError as e:
            return ToolResult(str(e), is_error=True)
        except Exception as e:
            return ToolResult(f"Search error: {e}", is_error=True)

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "pattern": {"type": "string", "description": "Regex pattern to search for"},
                "path": {"type": "string", "description": "Directory to search (default: project root)", "optional": True},
                "file_glob": {"type": "string", "description": "File pattern e.g. '*.py' (default: all files)", "optional": True},
                "max_results": {"type": "integer", "description": "Max results to return (default: 50)", "optional": True},
            },
        }


# RunCommand - execute shell commands with timeout + allowlist

class RunCommandTool(Tool):
    """Execute a shell command in the project directory.

    WHY an allowlist:
      The agent must not run arbitrary destructive commands (rm -rf, curl | sh).
      We default to allowing only safe read/build/test commands. The user can
      extend the allowlist at init time for their specific project needs.

    DESIGN NOTE on timeout:
      30 seconds is the default. Long-running processes (dev servers) should
      be started differently — this tool is for one-shot commands only.
    """

    name = "run_command"
    description = (
        "Run a shell command in the project directory. "
        "Returns stdout, stderr, and exit code."
    )

    DEFAULT_ALLOWED_PREFIXES = (
        "python", "pip", "pytest", "ruff", "mypy", "black", "isort",
        "npm", "npx", "node", "tsc",
        "git status", "git diff", "git log", "git show",
        "ls", "cat", "echo", "find", "grep", "head", "tail", "wc",
        "cargo", "go test", "go build",
    )

    def __init__(
        self,
        project_root: str | Path,
        allowed_prefixes: Optional[tuple] = None,
        timeout: int = 30,
    ):
        super().__init__(project_root)
        self._allowed = allowed_prefixes or self.DEFAULT_ALLOWED_PREFIXES
        self._timeout = timeout

    def run(self, command: str) -> ToolResult:
        if not self._is_allowed(command):
            return ToolResult(
                f"Command not in allowlist: {command!r}\n"
                f"Allowed prefixes: {', '.join(self._allowed)}",
                is_error=True,
            )

        try:
            result = subprocess.run(
                command,
                shell=True,
                cwd=str(self._root),
                capture_output=True,
                text=True,
                timeout=self._timeout,
            )
            output = ""
            if result.stdout:
                output += result.stdout
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}" if result.stdout else result.stderr

            is_error = result.returncode != 0
            return ToolResult(
                content=output.strip() or "(no output)",
                is_error=is_error,
                metadata={
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout_lines": len(result.stdout.splitlines()),
                },
            )
        except subprocess.TimeoutExpired:
            return ToolResult(
                f"Command timed out after {self._timeout}s: {command}",
                is_error=True,
            )
        except Exception as e:
            return ToolResult(f"Error running command: {e}", is_error=True)

    def _is_allowed(self, command: str) -> bool:
        cmd = command.strip().lower()
        return any(cmd.startswith(p.lower()) for p in self._allowed)

    @property
    def schema(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "command": {"type": "string", "description": "Shell command to execute"},
            },
        }


# ToolRegistry - central lookup used by the agent loop

class ToolRegistry:
    """Holds all available tools and dispatches calls by name.

    The agent loop calls registry.run("read_file", path="src/auth.py") and
    gets back a ToolResult. It never imports or instantiates tools directly.
    """

    def __init__(self, project_root: str | Path, allowed_commands: Optional[tuple] = None):
        root = Path(project_root)
        self._tools: dict[str, Tool] = {}

        for tool in [
            ReadFileTool(root),
            WriteFileTool(root),
            EditFileTool(root),
            ListDirTool(root),
            SearchFilesTool(root),
            RunCommandTool(root, allowed_prefixes=allowed_commands),
        ]:
            self._tools[tool.name] = tool

    def run(self, tool_name: str, **kwargs) -> ToolResult:
        if tool_name not in self._tools:
            return ToolResult(
                f"Unknown tool: {tool_name!r}. Available: {list(self._tools)}",
                is_error=True,
            )
        return self._tools[tool_name].run(**kwargs)

    def schemas(self) -> list[dict]:
        """Return all tool schemas — used to build the LLM system prompt."""
        return [t.schema for t in self._tools.values()]

    @property
    def available(self) -> list[str]:
        return list(self._tools.keys())