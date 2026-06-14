"""
ast-surgeon CLI - index a project, search it semantically, and keep it live.

Usage:
    ast-surgeon index [PATH]
    ast-surgeon search QUERY [PATH]
    ast-surgeon watch [PATH]

Run `ast-surgeon --help` or `ast-surgeon COMMAND --help` for details.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from ..components.sync import Indexer, FileWatcher, IndexResult
from ..components.embeddings.providers import EmbeddingError, list_providers

app = typer.Typer(
    name="ast-surgeon",
    help="AST-based code chunking, embedding, and semantic search.",
    no_args_is_help=True,
)
console = Console()


def _build_indexer(
    path: Path,
    store_type: str,
    embedding_provider: Optional[str],
) -> Indexer:
    try:
        return Indexer.create(
            path,
            store_type=store_type,
            embedding_provider=embedding_provider,
        )
    except EmbeddingError as exc:
        console.print(f"[red]Embedding provider error:[/red] {exc}")
        console.print(
            f"Available providers: {', '.join(list_providers())}\n"
            "Set the relevant API key env var, or use "
            "[bold]--embedding-provider local[/bold] for an offline model "
            r"(requires: pip install "
            + "\"ast-surgeon\\[local-embed]\""
            + ")."
        )
        raise typer.Exit(code=1)


@app.command()
def index(
    path: Path = typer.Argument(Path("."), help="Project root to index."),
    store: str = typer.Option(
        "chroma", "--store", "-s", help="Vector store backend: chroma | qdrant | pinecone."
    ),
    embedding_provider: Optional[str] = typer.Option(
        None,
        "--embedding-provider",
        "-e",
        help="Embedding provider: voyage | openai | cohere | gemini | mistral | local. "
        "Auto-detected from API keys if not given.",
    ),
) -> None:
    """Index (or re-index) a project for semantic search."""
    path = path.resolve()
    if not path.exists():
        console.print(f"[red]Path does not exist:[/red] {path}")
        raise typer.Exit(code=1)

    logging.basicConfig(level=logging.WARNING)
    indexer = _build_indexer(path, store, embedding_provider)

    console.print(f"Indexing [bold]{path}[/bold] ...")

    with console.status("Scanning and embedding files...") as status:
        def on_progress(done: int, total: int) -> None:
            status.update(f"Indexing files: {done}/{total}")

        result: IndexResult = indexer.index_project(progress_cb=on_progress)

    table = Table(title="Index summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Files scanned", str(result.files_scanned))
    table.add_row("Files changed", str(result.files_changed))
    table.add_row("Chunks added", str(result.chunks_added))
    table.add_row("Chunks deleted", str(result.chunks_deleted))
    table.add_row("Chunks unchanged", str(result.chunks_unchanged))
    table.add_row("Elapsed", f"{result.elapsed_seconds:.2f}s")
    console.print(table)

    if result.embed_errors:
        console.print(
            f"[yellow]{len(result.embed_errors)} chunk(s) failed to embed "
            "and will be retried on the next index run:[/yellow]"
        )
        for err in result.embed_errors[:10]:
            console.print(f"  - {err}")


@app.command()
def search(
    query: str = typer.Argument(..., help="Natural-language or code search query."),
    path: Path = typer.Argument(Path("."), help="Project root previously indexed."),
    top_k: int = typer.Option(10, "--top-k", "-k", help="Number of results to return."),
    language: Optional[str] = typer.Option(
        None, "--language", "-l", help="Restrict to a language (e.g. python, typescript)."
    ),
    file: Optional[str] = typer.Option(
        None, "--file", "-f", help="Restrict to a specific file path."
    ),
    store: str = typer.Option(
        "chroma", "--store", "-s", help="Vector store backend: chroma | qdrant | pinecone."
    ),
    embedding_provider: Optional[str] = typer.Option(
        None, "--embedding-provider", "-e", help="Embedding provider (must match the index)."
    ),
) -> None:
    """Search a previously-indexed project."""
    path = path.resolve()
    if not path.exists():
        console.print(f"[red]Path does not exist:[/red] {path}")
        raise typer.Exit(code=1)

    logging.basicConfig(level=logging.ERROR)
    indexer = _build_indexer(path, store, embedding_provider)
    indexer.load_manifest()

    hits = indexer.search(
        query, top_k=top_k, filter_language=language, filter_file=file
    )

    if not hits:
        console.print("[yellow]No results.[/yellow] Have you run `ast-surgeon index`?")
        raise typer.Exit(code=0)

    for hit in hits:
        chunk = hit.chunk
        header = (
            f"[bold cyan]{chunk.qualified_name()}[/bold cyan]  "
            f"[dim]({chunk.chunk_type.value}, L{chunk.start_line}-{chunk.end_line}, "
            f"score={hit.score:.3f})[/dim]"
        )
        console.print(header)
        snippet = chunk.content.strip().splitlines()
        preview = "\n".join(snippet[:6])
        console.print(f"[grey70]{preview}[/grey70]")
        if len(snippet) > 6:
            console.print("[grey50]...[/grey50]")
        console.print()


@app.command()
def watch(
    path: Path = typer.Argument(Path("."), help="Project root to watch."),
    store: str = typer.Option(
        "chroma", "--store", "-s", help="Vector store backend: chroma | qdrant | pinecone."
    ),
    embedding_provider: Optional[str] = typer.Option(
        None, "--embedding-provider", "-e", help="Embedding provider."
    ),
    skip_initial_index: bool = typer.Option(
        False, "--skip-initial-index", help="Don't run a full index before watching."
    ),
) -> None:
    """Index a project, then keep the index live as files change (Ctrl+C to stop)."""
    path = path.resolve()
    if not path.exists():
        console.print(f"[red]Path does not exist:[/red] {path}")
        raise typer.Exit(code=1)

    logging.basicConfig(level=logging.WARNING)
    indexer = _build_indexer(path, store, embedding_provider)

    if not skip_initial_index:
        console.print(f"Running initial index of [bold]{path}[/bold] ...")
        result = indexer.index_project()
        console.print(f"Initial index: {result}")
    else:
        indexer.load_manifest()

    def on_indexed(rel_path: str, result: IndexResult) -> None:
        if result.chunks_added or result.chunks_deleted:
            console.print(
                f"[green]reindexed[/green] {rel_path}: "
                f"+{result.chunks_added}/-{result.chunks_deleted} chunks "
                f"({result.elapsed_seconds * 1000:.0f}ms)"
            )

    console.print(f"Watching [bold]{path}[/bold] for changes (Ctrl+C to stop)...")
    with FileWatcher(indexer, path, on_indexed=on_indexed):
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            console.print("\nStopped.")


@app.command()
def providers() -> None:
    """List available embedding providers and vector store backends."""
    console.print("[bold]Embedding providers:[/bold] " + ", ".join(list_providers()))
    console.print("[bold]Vector stores:[/bold] chroma, qdrant, pinecone")


def main() -> None:
    app()


if __name__ == "__main__":
    main()