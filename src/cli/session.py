"""
ChatSession — the interactive REPL that connects the user to the agent.

SESSION LIFECYCLE:
  On `sovereign chat`:
    - Creates a new session via SessionManager
    - Prints the session ID (save it to resume later)

  On `sovereign chat --resume <session_id>`:
    - Loads history + session metadata from .sovereign/sessions/<id>/
    - Uses the same per-session Qdrant collection
    - Resumes conversation exactly where it left off

Slash commands:
  /help          show available commands
  /status        show index and session stats
  /session       show current session info
  /sessions      list all sessions for this project
  /search <q>    quick semantic search
  /files         list files modified this session
  /clear         clear conversation history (keeps session)
  /exit          exit (session is saved automatically)
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

console = Console()

SLASH_COMMANDS = {
    "/help":              "Show this help message",
    "/status":            "Show index and session stats",
    "/session":           "Show current session ID and info",
    "/sessions":          "List all saved sessions for this project",
    "/search <query>":    "Quick semantic search",
    "/files":             "Show files modified this session",
    "/checker on|off":    "Toggle CheckerAgent (runs & verifies code after each turn)",
    "/clear":             "Clear conversation history (keeps session alive)",
    "/exit":              "Exit (session saved automatically)",
}


class ChatSession:
    """
    Interactive REPL backed by a persisted AgentLoop session.

    History and the per-session Qdrant collection are both scoped to
    session.session_id so every chat session is fully isolated.
    """

    def __init__(
        self,
        project_root: Path,
        config: dict,
        session_id: Optional[str] = None,
        model: Optional[str] = None,
        checker_enabled: bool = False,
    ):
        self._root = project_root
        self._config = config
        self._model = model
        self._checker_enabled = checker_enabled
        self._session_start = time.time()

        from src.components.agent.loop import AgentLoop

        if session_id:
            self._agent_loop = AgentLoop.for_session(
                session_id=session_id,
                project_root=project_root,
                qdrant_host=config.get("qdrant_host", "localhost"),
                qdrant_port=config.get("qdrant_port", 6333),
                embedding_provider=config.get("embedding_provider"),
                checker_enabled=checker_enabled,
            )
            console.print(
                f"[dim]Resumed session [cyan]{session_id[:8]}[/] "
                f"({self._agent_loop.session.turn_count} turns)[/]"
            )
        else:
            self._agent_loop = AgentLoop.for_new_session(
                project_root=project_root,
                qdrant_host=config.get("qdrant_host", "localhost"),
                qdrant_port=config.get("qdrant_port", 6333),
                embedding_provider=config.get("embedding_provider"),
                checker_enabled=checker_enabled,
            )
            sid = self._agent_loop.session_id
            console.print(
                f"[dim]New session [cyan]{sid[:8]}[/cyan] "
                f"(resume later with [bold]sovereign chat --resume {sid}[/bold])[/]"
            )

        from src.components.tools.file_tools import ToolRegistry
        self._tools = ToolRegistry(project_root)

    # ─── REPL ─────────────────────────────────────────────────────────────────

    def run_repl(self) -> None:
        """Run the interactive chat loop until /exit or Ctrl+C."""
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Session saved.[/]")
                break

            if not user_input:
                continue

            if user_input.startswith("/"):
                should_exit = self._handle_slash(user_input)
                if should_exit:
                    break
                continue

            self._chat_turn(user_input)

    # ─── Slash commands ───────────────────────────────────────────────────────

    def _handle_slash(self, command: str) -> bool:
        parts = command.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/exit":
            self._show_session_summary()
            return True

        elif cmd == "/help":
            table = Table(show_header=False, box=None, padding=(0, 2))
            for name, desc in SLASH_COMMANDS.items():
                table.add_row(f"[cyan]{name}[/]", f"[dim]{desc}[/]")
            console.print(Panel(table, title="[bold]Commands[/]", border_style="dim"))

        elif cmd == "/clear":
            self._agent_loop.clear_history()
            console.print("[dim]Conversation history cleared.[/]")

        elif cmd == "/session":
            self._show_current_session()

        elif cmd == "/sessions":
            self._show_all_sessions()

        elif cmd == "/status":
            self._show_status()

        elif cmd == "/files":
            self._show_modified_files()

        elif cmd == "/checker":
            self._handle_checker_toggle(args)

        elif cmd == "/search":
            if not args:
                console.print("[yellow]Usage:[/] /search <query>")
            else:
                self._quick_search(args)

        else:
            console.print(
                f"[yellow]Unknown command:[/] {cmd}. Type [cyan]/help[/] for options."
            )

        return False

    # ─── Agent turn ───────────────────────────────────────────────────────────

    def _chat_turn(self, user_message: str) -> None:
        console.print()
        with console.status("[dim]Agent working…[/]", spinner="dots"):
            result = self._agent_loop.run(user_message)

        console.print("[bold green]Agent:[/]")
        console.print(Markdown(result.response))
        console.print()
        self._render_agent_actions(result)

    # ─── Display helpers ──────────────────────────────────────────────────────

    def _show_current_session(self) -> None:
        s = self._agent_loop.session
        table = Table(show_header=False, box=None, padding=(0, 1))
        table.add_row("[dim]Session ID[/]",   f"[cyan]{s.session_id}[/]")
        table.add_row("[dim]Collection[/]",   f"[dim]{s.collection_name}[/]")
        table.add_row("[dim]Turns[/]",        str(s.turn_count))
        table.add_row("[dim]Last active[/]",  s.age_str())
        table.add_row("[dim]History msgs[/]", str(len(s.history)))
        table.add_row("[dim]Files modified[/]", str(len(s.files_modified)))
        console.print(
            Panel(table, title="[bold]Current Session[/]", border_style="cyan")
        )
        console.print(
            f"[dim]Resume with:[/] [bold]sovereign chat --resume {s.session_id}[/bold]"
        )

    def _show_all_sessions(self) -> None:
        from src.components.agent.session_manager import SessionManager
        sm = SessionManager(self._root)
        sessions = sm.list_sessions()

        if not sessions:
            console.print("[dim]No saved sessions.[/]")
            return

        table = Table(
            "ID (short)",
            "Full UUID",
            "Turns",
            "Files",
            "Last active",
            box=None,
            show_header=True,
            header_style="bold dim",
        )
        for s in sessions:
            is_current = s.session_id == self._agent_loop.session_id
            sid_display = (
                f"[cyan bold]{s.session_id[:8]}[/] ◄ current"
                if is_current
                else f"[cyan]{s.session_id[:8]}[/]"
            )
            table.add_row(
                sid_display,
                f"[dim]{s.session_id}[/]",
                str(s.turn_count),
                str(len(s.files_modified)),
                s.age_str(),
            )

        console.print(
            Panel(table, title=f"[bold]Sessions ({len(sessions)})[/]", border_style="dim")
        )

    def _handle_checker_toggle(self, args: str) -> None:
        """Handle /checker on | /checker off | /checker (show status)."""
        arg = args.strip().lower()
        if arg == "on":
            self._agent_loop.enable_checker()
            console.print(
                "[green]✓ CheckerAgent ENABLED[/] — code will be run and verified after each turn."
            )
        elif arg == "off":
            self._agent_loop.disable_checker()
            console.print(
                "[yellow]CheckerAgent DISABLED[/] — code will not be executed automatically."
            )
        else:
            status = "[green]ON[/]" if self._agent_loop.checker_enabled else "[yellow]OFF[/]"
            console.print(f"CheckerAgent is currently {status}.")
            console.print("[dim]Use /checker on or /checker off to toggle.[/]")

    def _quick_search(self, query: str) -> None:
        store = self._agent_loop._store
        pipeline = self._agent_loop._pipeline
        if not store or not pipeline:
            console.print("[yellow]Vector search not available.[/]")
            return
        try:
            query_vec = pipeline.embed_query(query)
            results = store.search(query_vec, top_k=5)
            if not results:
                console.print("[dim]No results.[/]")
                return
            for r in results:
                c = r.chunk
                console.print(
                    f"[green]{r.score:.3f}[/]  [cyan]{c.name}[/]  "
                    f"[dim]{c.file_path}:{c.start_line}[/]"
                )
        except Exception as e:
            console.print(f"[red]Search error:[/] {e}")

    def _show_status(self) -> None:
        from src.components.sync.manifest import ManifestStore
        ms = ManifestStore(self._root)
        manifest = ms.load()
        stats = ms.stats(manifest) if manifest else {"files": 0, "chunks": 0}

        s = self._agent_loop.session

        table = Table(show_header=False, box=None)
        table.add_row("[dim]Project[/]",         str(self._root))
        table.add_row("[dim]Session ID[/]",       f"[cyan]{s.session_id[:8]}[/]")
        table.add_row("[dim]Collection[/]",       f"[dim]{s.collection_name}[/]")
        table.add_row("[dim]Files indexed[/]",    str(stats.get("files", 0)))
        table.add_row("[dim]Chunks[/]",           str(stats.get("chunks", 0)))
        table.add_row("[dim]Session turns[/]",    str(s.turn_count))
        table.add_row("[dim]Session duration[/]", f"{(time.time() - self._session_start):.0f}s")
        checker_status = "[green]ON[/]" if self._agent_loop.checker_enabled else "[dim]off[/]"
        table.add_row("[dim]CheckerAgent[/]", checker_status)
        console.print(Panel(table, title="Status", border_style="dim"))

    def _show_modified_files(self) -> None:
        files = self._agent_loop.session.files_modified
        if not files:
            console.print("[dim]No files modified this session.[/]")
            return
        for f in files:
            console.print(f"  [cyan]{f}[/]")

    def _render_agent_actions(self, result) -> None:
        parts = []
        if result.files_modified:
            parts.append(
                f"[dim]📝 {len(result.files_modified)} file(s) modified:[/] "
                + ", ".join(f"[cyan]{f}[/]" for f in result.files_modified[:4])
            )
        if result.commands_run:
            parts.append(
                f"[dim]⚡ Ran:[/] [dim]{result.commands_run[-1][:60]}[/]"
            )
        if result.total_steps:
            mode_icon = "⚡" if result.mode.value == "direct" else "🔀"
            parts.append(
                f"[dim]{mode_icon} {result.mode.value} · "
                f"{result.total_steps} step(s) · "
                f"{result.total_latency_ms:.0f}ms[/]"
            )
        if result.sleep_mode:
            parts.append("[yellow]⚠ Pipeline entered sleep mode[/]")
        if result.check_result is not None:
            cr = result.check_result
            if cr.cached:
                parts.append(f"[dim]✓ Check: cached {'PASS' if cr.passed else 'FAIL'}[/]")
            elif cr.passed:
                parts.append(f"[green]✓ Check: PASSED[/] [dim]{cr.command_used[:50]}[/]")
            else:
                parts.append(f"[red]✗ Check: FAILED[/] — {cr.repair_prompt[:80]}")
        if not result.success and result.error:
            parts.append(f"[red]Error: {result.error[:80]}[/]")
        for part in parts:
            console.print(part)
        if parts:
            console.print()

    def _show_session_summary(self) -> None:
        s = self._agent_loop.session
        duration = time.time() - self._session_start
        console.print(Panel(
            f"[dim]Session [cyan]{s.session_id[:8]}[/cyan] saved · "
            f"{s.turn_count} turns · {duration:.0f}s[/]\n"
            f"[dim]Resume:[/] [bold]sovereign chat --resume {s.session_id}[/bold]",
            border_style="dim",
        ))