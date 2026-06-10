"""
ChatSession — the interactive REPL that connects the user to the agent.

WHY this is a separate module from main.py:
  main.py owns CLI argument parsing and startup. ChatSession owns the
  conversation loop, message history, and session-level state (which files
  were modified, tool call history, etc.). Separating them lets us test
  the session logic without a TTY.

DESIGN NOTE on message history:
  We maintain a full message history list that gets sent to the LLM on every
  turn. This is the standard multi-turn approach — no summarisation yet.
  Phase 4's deep agent will plug in here and replace the simple completion
  call with a full planning + tool-use loop.

Slash commands (handled locally, never sent to LLM):
  /help     — show available commands
  /status   — show index stats
  /search   — quick semantic search
  /files    — list recent file modifications
  /clear    — clear conversation history
  /exit     — exit the session
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

console = Console()

SLASH_COMMANDS = {
    "/help":   "Show this help message",
    "/status": "Show index and session stats",
    "/search <query>": "Quick semantic search",
    "/files":  "Show files modified this session",
    "/clear":  "Clear conversation history",
    "/exit":   "Exit the session",
}


class ChatSession:
    """Manages one interactive chat session with the agent.

    Currently wraps a simple LLM completion call with tool context injected
    into the system prompt. Phase 4 will replace _get_response() with the
    full deep agent loop (planner → tool dispatcher → observer → synthesiser).
    """

    def __init__(
        self,
        project_root: Path,
        config: dict,
        model: Optional[str] = None,
    ):
        self._root = project_root
        self._config = config
        self._model = model
        self._history: list[dict] = []
        self._session_start = time.time()
        self._modified_files: list[str] = []

        # Tool registry — the agent's hands
        from src.components.tools.file_tools import ToolRegistry
        self._tools = ToolRegistry(project_root)

        # Vector store + pipeline for retrieval
        self._store = None
        self._pipeline = None
        self._init_retrieval()

    def _init_retrieval(self) -> None:
        """Initialise vector search (non-fatal if Qdrant not running)."""
        try:
            from src.components.vectorstore.qdrant_store import VectorStore
            from src.components.embeddings.providers import get_provider
            from src.components.embeddings.pipeline import EmbeddingPipeline

            self._store = VectorStore.connect(
                host=self._config.get("qdrant_host", "localhost"),
                port=self._config.get("qdrant_port", 6333),
            )
            provider = get_provider(self._config.get("embedding_provider"))
            self._pipeline = EmbeddingPipeline(provider=provider)
        except Exception as e:
            console.print(f"[yellow]Warning:[/] Vector search unavailable ({e})")


    # REPL

    def run_repl(self) -> None:
        """Run the interactive chat loop until /exit or Ctrl+C."""
        while True:
            try:
                user_input = console.input("[bold cyan]You:[/] ").strip()
            except (KeyboardInterrupt, EOFError):
                console.print("\n[dim]Session ended.[/]")
                break

            if not user_input:
                continue

            # Slash command handling
            if user_input.startswith("/"):
                should_exit = self._handle_slash(user_input)
                if should_exit:
                    break
                continue

            # Regular message → agent
            self._chat_turn(user_input)


    # Slash commands

    def _handle_slash(self, command: str) -> bool:
        """Handle a slash command. Returns True if session should exit."""
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
            self._history.clear()
            console.print("[dim]Conversation history cleared.[/]")

        elif cmd == "/status":
            self._show_status()

        elif cmd == "/files":
            self._show_modified_files()

        elif cmd == "/search":
            if not args:
                console.print("[yellow]Usage:[/] /search <query>")
            else:
                self._quick_search(args)

        else:
            console.print(f"[yellow]Unknown command:[/] {cmd}. Type [cyan]/help[/] for options.")

        return False

    # Agent turn

    def _chat_turn(self, user_message: str) -> None:
        """Process one user message: retrieve context → call LLM → display response."""

        # 1. Retrieve relevant context from vector store
        context_chunks = self._retrieve_context(user_message)

        # 2. Build system prompt with tool schemas + retrieved context
        system_prompt = self._build_system_prompt(context_chunks)

        # 3. Add user message to history
        self._history.append({"role": "user", "content": user_message})

        # 4. Call LLM (Phase 4 will replace this with the deep agent loop)
        console.print()
        with console.status("[dim]Agent thinking...[/]", spinner="dots"):
            response_text = self._get_response(system_prompt)

        # 5. Parse response for tool calls and execute them
        final_response = self._process_response(response_text)

        # 6. Display and record
        console.print("[bold green]Agent:[/]")
        console.print(Markdown(final_response))
        console.print()

        self._history.append({"role": "assistant", "content": final_response})

    def _retrieve_context(self, query: str, top_k: int = 6) -> list:
        """Retrieve relevant code chunks for the query."""
        if not self._store or not self._pipeline:
            return []
        try:
            query_vec = self._pipeline.embed_query(query)
            return self._store.search(query_vec, top_k=top_k)
        except Exception:
            return []

    def _build_system_prompt(self, context_chunks: list) -> str:
        """Build the system prompt injected with retrieved context and tool schemas."""
        project_name = self._root.name

        # Context section
        context_section = ""
        if context_chunks:
            context_section = "\n\n## Relevant Code Context\n"
            for r in context_chunks:
                chunk = r.chunk
                context_section += (
                    f"\n### {chunk.name or 'block'} "
                    f"({chunk.file_path}:{chunk.start_line}-{chunk.end_line})\n"
                    f"```{chunk.language}\n{chunk.content}\n```\n"
                )

        # Tool schemas
        tools_section = "\n\n## Available Tools\n"
        for schema in self._tools.schemas():
            tools_section += f"- **{schema['name']}**: {schema['description']}\n"
            for param, spec in schema.get("parameters", {}).items():
                if not spec.get("optional"):
                    tools_section += f"  - `{param}` ({spec['type']}): {spec.get('description','')}\n"

        return f"""You are Sovereign-Code, a production-grade coding agent working on the **{project_name}** project.

You have deep knowledge of the codebase through semantic search. You can read files, write code, run tests, and search the project.

## Project Root
{self._root}
{context_section}{tools_section}

## Instructions
- Answer concisely and accurately
- When you need to read a file, say so — the user can invoke tools explicitly
- When suggesting code changes, show diffs or the new file content clearly
- Reference specific file paths and line numbers when discussing code
- If you're unsure about something in the codebase, say so rather than guessing

**Phase 4 note:** Full autonomous tool execution is coming in the next phase. For now, suggest tool calls and the user can execute them.
"""

    def _get_response(self, system_prompt: str) -> str:
        """Call the LLM. Phase 4 replaces this with the deep agent loop."""
        import os

        # Try Groq first, then Gemini, then fallback message
        groq_key = os.environ.get("GROQ_API_KEY")
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")

        messages = [{"role": "user", "content": m["content"]}
                    if m["role"] == "user" else m
                    for m in self._history]

        if groq_key:
            return self._call_groq(system_prompt, messages, groq_key)
        elif gemini_key:
            return self._call_gemini(system_prompt, messages, gemini_key)
        else:
            return (
                "⚠️  No LLM API key found.\n\n"
                "Set one of:\n"
                "- `GROQ_API_KEY` (recommended: llama-3.3-70b-versatile, fast + free)\n"
                "- `GEMINI_API_KEY` (gemini-2.0-flash)\n\n"
                "The retrieval context and tools are ready — just needs a model."
            )

    def _call_groq(self, system_prompt: str, messages: list, api_key: str) -> str:
        import httpx
        model = self._model or "llama-3.3-70b-versatile"
        try:
            resp = httpx.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "system", "content": system_prompt}] + messages,
                    "max_tokens": 4096,
                    "temperature": 0.2,
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            return f"[Groq error: {e}]"

    def _call_gemini(self, system_prompt: str, messages: list, api_key: str) -> str:
        import httpx
        model = self._model or "gemini-2.0-flash"
        # Convert messages to Gemini format
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        try:
            resp = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                params={"key": api_key},
                json={
                    "system_instruction": {"parts": [{"text": system_prompt}]},
                    "contents": contents,
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 4096},
                },
                timeout=30.0,
            )
            resp.raise_for_status()
            return resp.json()["candidates"][0]["content"]["parts"][0]["text"]
        except Exception as e:
            return f"[Gemini error: {e}]"

    def _process_response(self, response: str) -> str:
        """Parse response for tool calls and execute them.

        Phase 4 will handle this fully autonomously in the agent loop.
        For now we just return the response as-is.
        """
        return response


    # Display helpers

    def _quick_search(self, query: str) -> None:
        if not self._store or not self._pipeline:
            console.print("[yellow]Vector search not available.[/]")
            return
        try:
            query_vec = self._pipeline.embed_query(query)
            results = self._store.search(query_vec, top_k=5)
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
        stats = ms.stats(manifest)

        table = Table(show_header=False, box=None)
        table.add_row("[dim]Project[/]", str(self._root))
        table.add_row("[dim]Files indexed[/]", str(stats["files"]))
        table.add_row("[dim]Chunks[/]", str(stats["chunks"]))
        table.add_row("[dim]Session duration[/]", f"{(time.time() - self._session_start):.0f}s")
        table.add_row("[dim]Messages[/]", str(len(self._history)))
        console.print(Panel(table, title="Status", border_style="dim"))

    def _show_modified_files(self) -> None:
        from src.components.tools.file_tools import WriteFileTool
        write_tool = self._tools._tools.get("write_file")
        log = write_tool.write_log if write_tool else []
        if not log:
            console.print("[dim]No files modified this session.[/]")
            return
        for entry in log:
            action = "[green]created[/]" if entry["created"] else "[yellow]updated[/]"
            console.print(f"  {action}  {entry['path']}  [dim]({entry['lines']} lines)[/]")

    def _show_session_summary(self) -> None:
        duration = time.time() - self._session_start
        console.print(Panel(
            f"[dim]Session ended after {duration:.0f}s  ·  {len(self._history)} messages[/]",
            border_style="dim",
        ))