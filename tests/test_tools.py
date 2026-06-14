"""Unit tests for the file system tools."""

import pytest
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from ast_surgeon.components.tools.file_tools import (
    ReadFileTool, WriteFileTool, EditFileTool,
    ListDirTool, SearchFilesTool, RunCommandTool,
    ToolRegistry,
)


@pytest.fixture
def project(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text(
        "def hello():\n    return 'world'\n\ndef goodbye():\n    return 'bye'\n"
    )
    (tmp_path / "src" / "utils.py").write_text("CONSTANT = 42\n")
    (tmp_path / "README.md").write_text("# Project\n\nA test project.\n")
    return tmp_path


# ReadFileTool

def test_read_file_returns_content(project):
    t = ReadFileTool(project)
    r = t.run(path="src/main.py")
    assert not r.is_error
    assert "hello" in r.content
    assert "world" in r.content


def test_read_file_with_line_numbers(project):
    t = ReadFileTool(project)
    r = t.run(path="src/main.py", show_line_numbers=True)
    assert "1 │" in r.content or "1│" in r.content.replace(" ", "")


def test_read_file_line_range(project):
    t = ReadFileTool(project)
    r = t.run(path="src/main.py", start_line=1, end_line=2)
    assert "hello" in r.content
    assert "goodbye" not in r.content


def test_read_file_missing(project):
    t = ReadFileTool(project)
    r = t.run(path="nonexistent.py")
    assert r.is_error


def test_read_file_escape_blocked(project):
    t = ReadFileTool(project)
    r = t.run(path="../../etc/passwd")
    assert r.is_error


def test_read_file_metadata(project):
    t = ReadFileTool(project)
    r = t.run(path="src/main.py")
    assert r.metadata["total_lines"] == 5


# WriteFileTool

def test_write_creates_file(project):
    t = WriteFileTool(project)
    r = t.run(path="src/new.py", content="x = 1\n")
    assert not r.is_error
    assert (project / "src" / "new.py").read_text() == "x = 1\n"


def test_write_creates_dirs(project):
    t = WriteFileTool(project)
    r = t.run(path="deep/nested/file.py", content="pass\n")
    assert not r.is_error
    assert (project / "deep" / "nested" / "file.py").exists()


def test_write_overwrites_existing(project):
    t = WriteFileTool(project)
    t.run(path="src/utils.py", content="NEW = 99\n")
    assert (project / "src" / "utils.py").read_text() == "NEW = 99\n"


def test_write_log_tracks_operations(project):
    t = WriteFileTool(project)
    t.run(path="a.py", content="a\n")
    t.run(path="b.py", content="b\n")
    assert len(t.write_log) == 2
    assert t.write_log[0]["path"] == "a.py"
    assert t.write_log[0]["created"] is True


def test_write_escape_blocked(project):
    t = WriteFileTool(project)
    r = t.run(path="../evil.py", content="rm -rf /\n")
    assert r.is_error


# EditFileTool

def test_edit_replaces_lines(project):
    t = EditFileTool(project)
    r = t.run(
        path="src/main.py",
        start_line=2,
        end_line=2,
        new_content="    return 'EDITED'\n",
    )
    assert not r.is_error
    content = (project / "src" / "main.py").read_text()
    assert "EDITED" in content
    assert "goodbye" in content  # rest unchanged


def test_edit_invalid_range(project):
    t = EditFileTool(project)
    r = t.run(path="src/main.py", start_line=100, end_line=200, new_content="x\n")
    assert r.is_error


def test_edit_missing_file(project):
    t = EditFileTool(project)
    r = t.run(path="nope.py", start_line=1, end_line=1, new_content="x\n")
    assert r.is_error


# ListDirTool

def test_list_dir_shows_files(project):
    t = ListDirTool(project)
    r = t.run(path=".")
    assert not r.is_error
    assert "src" in r.content
    assert "README.md" in r.content


def test_list_dir_depth(project):
    t = ListDirTool(project)
    r_depth1 = t.run(path=".", depth=1)
    r_depth2 = t.run(path=".", depth=2)
    # depth=2 should show files inside src/
    assert "main.py" in r_depth2.content
    assert "main.py" not in r_depth1.content


def test_list_dir_missing(project):
    t = ListDirTool(project)
    r = t.run(path="nonexistent/")
    assert r.is_error


# SearchFilesTool

def test_search_finds_pattern(project):
    t = SearchFilesTool(project)
    r = t.run(pattern="def hello")
    assert not r.is_error
    assert "main.py" in r.content
    assert "hello" in r.content


def test_search_with_file_glob(project):
    t = SearchFilesTool(project)
    r = t.run(pattern="def ", file_glob="*.py")
    assert not r.is_error
    assert "main.py" in r.content


def test_search_no_results(project):
    t = SearchFilesTool(project)
    r = t.run(pattern="ZZZNOMATCH_XYZ")
    assert not r.is_error
    assert r.metadata["matches"] == 0


def test_search_invalid_regex(project):
    t = SearchFilesTool(project)
    r = t.run(pattern="[[[invalid")
    assert r.is_error


def test_search_returns_line_numbers(project):
    t = SearchFilesTool(project)
    r = t.run(pattern="hello")
    # Format: "path:lineno: content"
    assert ":1:" in r.content or ":2:" in r.content


# RunCommandTool

def test_run_allowed_command(project):
    t = RunCommandTool(project)
    r = t.run(command="echo hello")
    assert not r.is_error
    assert "hello" in r.content


def test_run_command_captures_exit_code(project):
    t = RunCommandTool(project)
    r = t.run(command="python -c 'import sys; sys.exit(1)'")
    assert r.is_error
    assert r.metadata["exit_code"] == 1


def test_run_blocked_command(project):
    t = RunCommandTool(project)
    r = t.run(command="rm -rf /tmp/test")
    assert r.is_error
    assert "allowlist" in r.content


def test_run_timeout(project):
    t = RunCommandTool(project, timeout=1)
    r = t.run(command="python -c 'import time; time.sleep(5)'")
    assert r.is_error
    assert "timed out" in r.content


# ToolRegistry

def test_registry_dispatches_correctly(project):
    registry = ToolRegistry(project)
    r = registry.run("read_file", path="README.md")
    assert not r.is_error
    assert "Project" in r.content


def test_registry_unknown_tool(project):
    registry = ToolRegistry(project)
    r = registry.run("nonexistent_tool")
    assert r.is_error


def test_registry_lists_available(project):
    registry = ToolRegistry(project)
    available = registry.available
    assert "read_file" in available
    assert "write_file" in available
    assert "edit_file" in available
    assert "list_dir" in available
    assert "search_files" in available
    assert "run_command" in available


def test_registry_returns_schemas(project):
    registry = ToolRegistry(project)
    schemas = registry.schemas()
    names = [s["name"] for s in schemas]
    assert "read_file" in names
    assert "write_file" in names


if __name__ == "__main__":
    pytest.main([__file__, "-v"])