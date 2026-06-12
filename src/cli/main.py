"""
CLI entry point - `sovereign` command.

Commands:
  sovereign init    Index the current project, start the file watcher daemon
  sovereign chat    Start an interactive chat session with the agent
  sovereign search  Quick semantic search (no full chat)
  sovereign status  Show index stats for the current project

WHY Typer over argparse/click:
  Typer generates --help automatically from type annotations and docstrings,
  produces clean error messages, and works great with Rich for styled output.
  It's what FastAPI's creator built specifically for CLI tools.

DESIGN NOTE on config resolution:
  All commands look for a .sovereign/config.json in the current working
  directory (or any parent). This is the same pattern as git - you can run
  `sovereign chat` from any subdirectory and it finds the project root.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.prompt import Prompt
from rich.table import Table
from rich.text import Text
from rich import print as rprint

app = typer.Typer(
    name="sovereign",
    help="Sovereign-Code - production-grade coding agent",
    add_completion=False,
)
console = Console()


# Helpers

def _find_project_root(start: Path = Path(".")) -> Optional[Path]:
    """Walk up from start looking for .sovereign/config.json or .git."""
    current = start.resolve()
    for parent in [current, *current.parents]:
        if (parent / ".sovereign").is_dir():
            return parent
        if (parent / ".git").is_dir():
            return parent
    return current  # fallback: use cwd


def _load_config(project_root: Path) -> dict:
    config_path = project_root / ".sovereign" / "config.json"
    if config_path.exists():
        return json.loads(config_path.read_text())
    return {}


def _save_config(project_root: Path, config: dict) -> None:
    config_dir = project_root / ".sovereign"
    config_dir.mkdir(exist_ok=True)
    (config_dir / "config.json").write_text(json.dumps(config, indent=2))


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


def _build_indexer(project_root: Path, config: dict):
    """Wire up Indexer from config. Deferred import to keep CLI startup fast."""
    from src.components.sync.indexer import Indexer

    return Indexer.create(
        project_root=project_root,
        qdrant_host=config.get("qdrant_host", "localhost"),
        qdrant_port=config.get("qdrant_port", 6333),
        embedding_provider=config.get("embedding_provider"),
    )


# sovereign init

@app.command()
def init(
    path: str = typer.Argument(".", help="Project directory to index"),
    provider: Optional[str] = typer.Option(None, "--provider", "-p", help="Embedding provider: voyage|openai|local"),
    qdrant_host: str = typer.Option("localhost", "--qdrant-host"),
    qdrant_port: int = typer.Option(6333, "--qdrant-port"),
    watch: bool = typer.Option(False, "--watch", "-w", help="Keep watching for changes after indexing"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Index a project and optionally start the file watcher daemon.

    Run this once per project. After init, use `sovereign chat` to query the agent.
    """
    _setup_logging(verbose)
    project_root = Path(path).resolve()

    if not project_root.exists():
        console.print(f"[red]Error:[/] Path does not exist: {project_root}")
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Sovereign-Code[/] - Initialising project\n"
        f"[dim]Root:[/] {project_root}",
        border_style="cyan",
    ))

    # Save config
    config = {
        "project_root": str(project_root),
        "qdrant_host": qdrant_host,
        "qdrant_port": qdrant_port,
        "embedding_provider": provider,
    }
    _save_config(project_root, config)

    # Build indexer
    try:
        indexer = _build_indexer(project_root, config)
    except Exception as e:
        console.print(f"[red]Failed to connect to Qdrant at {qdrant_host}:{qdrant_port}[/]")
        console.print(f"[dim]{e}[/]")
        console.print("\n[yellow]Tip:[/] Start Qdrant with Docker:")
        console.print("[dim]  docker run -p 6333:6333 qdrant/qdrant[/]")
        raise typer.Exit(1)

    # Full index with progress bar
    total_files = sum(1 for _ in indexer._walk_project())
    done_files = 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Indexing files...", total=total_files)

        def on_progress(done: int, total: int):
            nonlocal done_files
            done_files = done
            progress.update(task, completed=done)

        result = indexer.index_project(progress_cb=on_progress)

    # Summary
    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[dim]Files scanned[/]", str(result.files_scanned))
    table.add_row("[dim]Files indexed[/]", str(result.files_changed))
    table.add_row("[green]Chunks added[/]", str(result.chunks_added))
    table.add_row("[dim]Time[/]", f"{result.elapsed_seconds:.1f}s")

    if result.embed_errors:
        table.add_row("[red]Errors[/]", str(len(result.embed_errors)))

    console.print(Panel(table, title="[bold]Index Complete[/]", border_style="green"))
    console.print(f"[dim]Manifest saved to {project_root / '.sovereign' / 'manifest.json'}[/]")

    if watch:
        _start_watcher(indexer, project_root)


# sovereign status

@app.command()
def status(
    path: str = typer.Argument(".", help="Project directory"),
):
    """Show indexing stats for the current project."""
    project_root = _find_project_root(Path(path))
    config = _load_config(project_root)

    from src.components.sync.manifest import ManifestStore
    ms = ManifestStore(project_root)
    manifest = ms.load()
    stats = ms.stats(manifest)

    if not manifest:
        console.print("[yellow]No index found.[/] Run [bold]sovereign init[/] first.")
        raise typer.Exit(0)

    # Per-language breakdown
    lang_counts: dict[str, int] = {}
    for records in manifest.values():
        for r in records:
            lang = r.chunk_type.value
            lang_counts[lang] = lang_counts.get(lang, 0) + 1

    table = Table(title="[bold]Sovereign-Code Index Status[/]", border_style="cyan")
    table.add_column("Metric", style="dim")
    table.add_column("Value", style="bold")

    table.add_row("Project root", str(project_root))
    table.add_row("Files indexed", str(stats["files"]))
    table.add_row("Total chunks", str(stats["chunks"]))
    table.add_row("Manifest", stats["manifest_path"])

    console.print(table)

    # Chunk type breakdown
    if lang_counts:
        type_table = Table(title="Chunk Types", border_style="dim", show_header=True)
        type_table.add_column("Type")
        type_table.add_column("Count", justify="right")
        for ctype, count in sorted(lang_counts.items(), key=lambda x: -x[1]):
            type_table.add_row(ctype, str(count))
        console.print(type_table)


# sovereign search

@app.command()
def search(
    query: str = typer.Argument(..., help="Natural language search query"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    language: Optional[str] = typer.Option(None, "--lang", "-l", help="Filter by language"),
    path: str = typer.Argument(".", help="Project directory"),
):
    """Semantic search over the indexed codebase."""
    project_root = _find_project_root(Path(path))
    config = _load_config(project_root)

    from src.components.embeddings.providers import get_provider
    from src.components.embeddings.pipeline import EmbeddingPipeline
    from src.components.vectorstore.qdrant_store import VectorStore

    with console.status("[cyan]Searching...[/]"):
        try:
            store = VectorStore.connect(
                host=config.get("qdrant_host", "localhost"),
                port=config.get("qdrant_port", 6333),
            )
            provider = get_provider(config.get("embedding_provider"))
            pipeline = EmbeddingPipeline(provider=provider)
            query_vec = pipeline.embed_query(query)
            results = store.search(query_vec, top_k=top_k, filter_language=language)
        except Exception as e:
            console.print(f"[red]Search failed:[/] {e}")
            raise typer.Exit(1)

    if not results:
        console.print("[yellow]No results found.[/]")
        raise typer.Exit(0)

    console.print(f"\n[bold]Results for:[/] [cyan]{query}[/]\n")
    for i, r in enumerate(results, 1):
        chunk = r.chunk
        score_color = "green" if r.score > 0.8 else "yellow" if r.score > 0.6 else "red"
        console.print(
            f"[bold]{i}.[/] [{score_color}]{r.score:.3f}[/]  "
            f"[bold cyan]{chunk.name or '<block>'}[/]  "
            f"[dim]{chunk.file_path}:{chunk.start_line}-{chunk.end_line}[/]"
        )
        if chunk.docstring:
            console.print(f"   [dim]{chunk.docstring[:100]}[/]")
        if chunk.calls:
            console.print(f"   [dim]calls: {', '.join(chunk.calls[:5])}[/]")
        console.print()


# sovereign chat

@app.command()
def chat(
    path: str = typer.Argument(".", help="Project directory"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Override LLM model"),
    resume: Optional[str] = typer.Option(None, "--resume", "-r", help="Resume a previous session by ID"),
    check: bool = typer.Option(False, "--check", "-c",
        help="Enable CheckerAgent: runs code after each turn, auto-repairs failures."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start or resume an interactive chat session with the coding agent.

    New session:
        sovereign chat

    With code checking enabled:
        sovereign chat --check

    Resume a previous session:
        sovereign chat --resume <session-id>

    The agent has access to your indexed codebase and can read, write,
    search, and run commands in your project.
    Each session gets its own isolated Qdrant vector collection.
    Toggle checker at runtime with /checker on | /checker off.
    """
    _setup_logging(verbose)
    project_root = _find_project_root(Path(path))
    config = _load_config(project_root)

    if not (project_root / ".sovereign").is_dir():
        console.print("[yellow]Project not initialised.[/] Run [bold]sovereign init[/] first.")
        raise typer.Exit(1)

    from src.cli.session import ChatSession

    console.print(Panel(
        "[bold cyan]Sovereign-Code Agent[/]\n"
        "[dim]Type your question or task. /help for commands, /exit to quit.[/]",
        border_style="cyan",
    ))
    console.print(f"[dim]Project: {project_root}[/]\n")

    session = ChatSession(
        project_root=project_root,
        config=config,
        session_id=resume,
        model=model,
        checker_enabled=check or config.get("checker_enabled", False),
    )
    session.run_repl()


@app.command()
def sessions(
    path: str = typer.Argument(".", help="Project directory"),
    delete: Optional[str] = typer.Option(None, "--delete", "-d", help="Delete a session by ID"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """List all saved sessions for this project.

    Each session has its own conversation history and Qdrant collection.

    Resume a session:
        sovereign chat --resume <session-id>

    Delete a session (also removes its Qdrant collection):
        sovereign sessions --delete <session-id>
    """
    _setup_logging(verbose)
    project_root = _find_project_root(Path(path))
    config = _load_config(project_root)

    from src.components.agent.session_manager import SessionManager
    sm = SessionManager(project_root)

    if delete:
        try:
            sm.delete(
                delete,
                drop_qdrant_collection=True,
                qdrant_host=config.get("qdrant_host", "localhost"),
                qdrant_port=config.get("qdrant_port", 6333),
            )
            console.print(f"[green]Deleted session[/] [cyan]{delete[:8]}[/]")
        except FileNotFoundError:
            console.print(f"[red]Session not found:[/] {delete}")
        return

    all_sessions = sm.list_sessions()

    if not all_sessions:
        console.print("[dim]No saved sessions. Start one with [bold]sovereign chat[/bold].[/]")
        return

    table = Table(
        "Short ID", "Full UUID", "Turns", "Files", "Last active",
        box=None, show_header=True, header_style="bold dim",
    )
    for s in all_sessions:
        table.add_row(
            f"[cyan]{s.session_id[:8]}[/]",
            f"[dim]{s.session_id}[/]",
            str(s.turn_count),
            str(len(s.files_modified)),
            s.age_str(),
        )

    console.print(Panel(
        table,
        title=f"[bold]Sessions ({len(all_sessions)})[/]",
        border_style="dim",
    ))
    console.print(
        "[dim]Resume:[/] [bold]sovereign chat --resume <full-uuid>[/bold]  "
        "  [dim]Delete:[/] [bold]sovereign sessions --delete <full-uuid>[/bold]"
    )


# Watcher helper (used by init --watch)

def _start_watcher(indexer, project_root: Path) -> None:
    from src.components.sync.watcher import FileWatcher

    reindex_count = 0

    def on_indexed(rel_path: str, result) -> None:
        nonlocal reindex_count
        reindex_count += 1
        parts = []
        if result.chunks_added:
            parts.append(f"[green]+{result.chunks_added}[/]")
        if result.chunks_deleted:
            parts.append(f"[red]-{result.chunks_deleted}[/]")
        if result.chunks_unchanged:
            parts.append(f"[dim]={result.chunks_unchanged}[/]")
        summary = " ".join(parts) or "[dim]no changes[/]"
        console.print(f"[dim][watcher][/] {rel_path} {summary}")

    watcher = FileWatcher(indexer, project_root, on_indexed=on_indexed)
    watcher.start()
    console.print(f"[cyan]Watching[/] {project_root} for changes. Press Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        watcher.stop()
        console.print(f"\n[dim]Watcher stopped. {reindex_count} reindex(es) performed.[/]")


# Entry point

@app.command()
def serve(
    path: str = typer.Argument(".", help="Project directory"),
    host: str = typer.Option("0.0.0.0", "--host"),
    port: int = typer.Option(8000, "--port"),
    reload: bool = typer.Option(False, "--reload"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Start the Sovereign-Code API server with SSE streaming.

    Exposes /chat (SSE), /search, /status, /index, /health endpoints.
    """
    _setup_logging(verbose)
    project_root = _find_project_root(Path(path))
    config = _load_config(project_root)

    console.print(Panel(
        f"[bold cyan]Sovereign-Code API[/]\n"
        f"[dim]Project:[/] {project_root}\n"
        f"[dim]Listening on:[/] http://{host}:{port}",
        border_style="cyan",
    ))

    try:
        import uvicorn
        from src.api.server import create_app

        api_app = create_app(
            project_root=str(project_root),
            qdrant_host=config.get("qdrant_host", "localhost"),
            qdrant_port=config.get("qdrant_port", 6333),
            embedding_provider=config.get("embedding_provider"),
        )
        uvicorn.run(api_app, host=host, port=port, reload=reload)
    except ImportError:
        console.print("[red]uvicorn not installed.[/] Run: pip install uvicorn")
        raise typer.Exit(1)


@app.command()
def eval(
    output: str = typer.Option("eval_results.json", "--output", "-o"),
    task: Optional[str] = typer.Option(None, "--task", "-t", help="Run a specific task ID only"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Run the built-in eval harness and report agent quality metrics."""
    _setup_logging(verbose)

    from src.evals.harness import EvalSuite, BUILTIN_TASKS
    import os

    tasks = BUILTIN_TASKS
    if task:
        tasks = [t for t in tasks if t.id == task]
        if not tasks:
            console.print(f"[red]Task {task!r} not found.[/] Available: {[t.id for t in BUILTIN_TASKS]}")
            raise typer.Exit(1)

    console.print(Panel(
        f"[bold cyan]Sovereign-Code Eval Harness[/]\n"
        f"[dim]Running {len(tasks)} task(s)[/]",
        border_style="cyan",
    ))

    suite = EvalSuite(tasks)
    results_list = []

    def on_done(result):
        icon = "[green]✓[/]" if result.passed else "[red]✗[/]"
        console.print(f"  {icon} {result.task_id}  [dim]{result.total_steps} steps · {result.latency_ms:.0f}ms[/]")
        if not result.passed:
            for passed, reason in result.assertion_results:
                if not passed:
                    console.print(f"    [red]↳ {reason}[/]")

    report = suite.run(
        groq_api_key=os.environ.get("GROQ_API_KEY"),
        gemini_api_key=os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY"),
        on_task_done=on_done,
    )

    # Summary table
    table = Table(title="[bold]Eval Results[/]", border_style="cyan")
    table.add_column("Metric"); table.add_column("Value", justify="right")
    table.add_row("Tasks", f"{len(report.results)}")
    table.add_row("Passed", f"[green]{report.passed_count}[/]")
    table.add_row("Success rate", f"[bold]{report.task_success_rate:.1%}[/]")
    table.add_row("Avg steps", f"{report.avg_steps:.1f}")
    table.add_row("Avg latency", f"{report.avg_latency_ms:.0f}ms")
    table.add_row("Tool accuracy", f"{report.tool_accuracy:.1%}")
    table.add_row("Total time", f"{report.total_elapsed_seconds:.1f}s")
    console.print(table)

    report.save(output)
    console.print(f"[dim]Results saved to {output}[/]")


def main():
    app()


if __name__ == "__main__":
    main()