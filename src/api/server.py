"""
FastAPI server - streaming SSE chat + REST endpoints for Sovereign-Code.

WHY Server-Sent Events (SSE) instead of WebSockets:
  SSE is unidirectional (server → client), which is exactly what we need for
  streaming agent responses. It's simpler than WebSockets (no handshake, no
  bidirectional protocol), works over plain HTTP/2, and is natively supported
  by browsers and curl. WebSockets add complexity we don't need here.

WHY streaming matters for the agent:
  The deep agent loop can run for 10-30 seconds on complex tasks. Without
  streaming, the user stares at a blank screen. With SSE, we emit events as
  each step completes:
    {"type": "step", "agent_id": "ag-1", "tool": "read_file", "step": 1}
    {"type": "step", "agent_id": "ag-1", "tool": "write_file", "step": 2}
    {"type": "done", "response": "...", "files_modified": [...]}

  The CLI's `sovereign chat` connects to this server in server mode, and any
  web frontend can consume the same stream.

Endpoints:
  POST /chat          - stream agent response as SSE
  POST /search        - semantic search, returns JSON
  GET  /status        - index stats
  POST /index         - trigger re-index of a file or full project
  GET  /health        - liveness probe

DESIGN NOTE on AgentLoop per-request vs singleton:
  We use ONE AgentLoop per server instance (singleton), not one per request.
  This preserves conversation history across turns (the loop holds history).
  For multi-user scenarios (Phase 6+), we'd add session IDs and a loop pool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncGenerator, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from ..components.agent.loop import AgentLoop
from ..components.agent.models import AgentMode, StepTrace
from ..components.sync.indexer import Indexer
from ..components.sync.manifest import ManifestStore

logger = logging.getLogger(__name__)


# Request / Response models

class ChatRequest(BaseModel):
    message: str
    stream: bool = True


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5
    language: Optional[str] = None


class IndexRequest(BaseModel):
    file_path: Optional[str] = None   # None = full project re-index


class SearchResult(BaseModel):
    name: Optional[str]
    file_path: str
    start_line: int
    end_line: int
    language: str
    score: float
    docstring: Optional[str]


# App factory

def create_app(
    project_root: str,
    qdrant_host: str = "localhost",
    qdrant_port: int = 6333,
    embedding_provider: Optional[str] = None,
) -> FastAPI:
    """Create and configure the FastAPI app.

    Args:
        project_root:        Root directory of the project being served.
        qdrant_host/port:    Qdrant connection details.
        embedding_provider:  "voyage"|"openai"|"local"|None (auto).

    Returns:
        Configured FastAPI app ready for uvicorn.
    """

    # Shared state - initialised in lifespan
    state: dict = {
        "agent_loop": None,
        "indexer": None,
        "project_root": Path(project_root),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        """Startup: wire up AgentLoop and Indexer."""
        logger.info("Starting Sovereign-Code API for %s", project_root)
        try:
            # Create or resume a session for the API server process
            from ..components.memory import SessionManager
            _sovereign_dir = Path(project_root) / ".sovereign"
            _session_mgr = SessionManager(_sovereign_dir, project_root=project_root)
            _api_session, _is_new = _session_mgr.get_or_create_session()
            logger.info(
                "%s API session %s",
                "Created" if _is_new else "Resumed",
                _api_session.session_id,
            )

            state["agent_loop"] = AgentLoop.create(
                project_root=project_root,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                embedding_provider=embedding_provider,
                session=_api_session,
            )
            state["indexer"] = Indexer.create(
                project_root=project_root,
                qdrant_host=qdrant_host,
                qdrant_port=qdrant_port,
                embedding_provider=embedding_provider,
            )
            state["indexer"].load_manifest()
            logger.info("AgentLoop and Indexer ready")
        except Exception as exc:
            logger.error("Startup failed: %s", exc)
        yield
        logger.info("Sovereign-Code API shutting down")

    app = FastAPI(
        title="Sovereign-Code API",
        description="Production-grade coding agent API",
        version="0.5.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )


    # Health

    @app.get("/health")
    async def health():
        """Liveness probe."""
        loop_ready = state["agent_loop"] is not None
        return {
            "status": "ok" if loop_ready else "degraded",
            "agent_loop": loop_ready,
            "project_root": str(state["project_root"]),
        }


    # Status

    @app.get("/status")
    async def status():
        """Return index statistics."""
        root = state["project_root"]
        ms = ManifestStore(root)
        manifest = ms.load()
        stats = ms.stats(manifest)

        # Per-language chunk counts
        lang_counts: dict[str, int] = {}
        for records in manifest.values():
            for r in records:
                lang_counts[r.chunk_type.value] = lang_counts.get(r.chunk_type.value, 0) + 1

        return {
            "project_root": str(root),
            "files_indexed": stats["files"],
            "total_chunks": stats["chunks"],
            "chunk_types": lang_counts,
            "agent_history_turns": len(state["agent_loop"].history) // 2
                if state["agent_loop"] else 0,
        }


    # Chat - SSE streaming

    @app.post("/chat")
    async def chat(req: ChatRequest):
        """Stream agent response as Server-Sent Events.

        Each SSE event is a JSON object:
          {"type": "step",   "data": {agent_id, step, tool, thought, error}}
          {"type": "done",   "data": {response, mode, steps, files_modified, ...}}
          {"type": "error",  "data": {message}}

        If stream=False, waits for completion and returns JSON directly.
        """
        loop: AgentLoop = state["agent_loop"]
        if loop is None:
            raise HTTPException(503, "Agent loop not initialised")

        if not req.stream:
            # Non-streaming: run synchronously and return
            try:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, loop.run, req.message
                )
                return _result_to_dict(result)
            except Exception as exc:
                raise HTTPException(500, str(exc))

        # Streaming SSE
        return StreamingResponse(
            _stream_agent(loop, req.message),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    async def _stream_agent(loop: AgentLoop, message: str) -> AsyncGenerator[str, None]:
        """Run agent in a thread, stream step events as they complete."""

        # We can't hook into the agent loop's internal steps directly without
        # modifying SubAgent - instead we run the loop in an executor and
        # emit a single "thinking" event, then the done event.
        #
        # DESIGN NOTE: for true per-step streaming, SubAgent would need to
        # accept a callback(StepTrace) that we bridge to the SSE stream.
        # That's a clean Phase 6 enhancement - add on_step callback to SubAgent.

        yield _sse("thinking", {"message": f"Working on: {message[:60]}..."})

        try:
            result = await asyncio.get_event_loop().run_in_executor(
                None, loop.run, message
            )

            # Emit each step trace
            for trace in result.all_traces:
                yield _sse("step", {
                    "agent_id": trace.agent_id,
                    "step": trace.step_number,
                    "tool": trace.tool_name,
                    "thought": trace.thought[:120] if trace.thought else "",
                    "error": trace.tool_error,
                    "latency_ms": round(trace.latency_ms, 1),
                })
                await asyncio.sleep(0)  # yield to event loop

            yield _sse("done", _result_to_dict(result))

        except Exception as exc:
            logger.error("Stream agent error: %s", exc)
            yield _sse("error", {"message": str(exc)})


    # Search

    @app.post("/search")
    async def search(req: SearchRequest):
        """Semantic search over the indexed codebase."""
        loop: AgentLoop = state["agent_loop"]
        if not loop or not loop._store or not loop._pipeline:
            raise HTTPException(503, "Vector search not available")

        try:
            query_vec = await asyncio.get_event_loop().run_in_executor(
                None, loop._pipeline.embed_query, req.query
            )
            results = loop._store.search(
                query_vec, top_k=req.top_k, filter_language=req.language
            )
            return {
                "query": req.query,
                "results": [
                    SearchResult(
                        name=r.chunk.name,
                        file_path=r.chunk.file_path,
                        start_line=r.chunk.start_line,
                        end_line=r.chunk.end_line,
                        language=r.chunk.language,
                        score=round(r.score, 4),
                        docstring=r.chunk.docstring,
                    ).model_dump()
                    for r in results
                ],
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))

    # Index

    @app.post("/index")
    async def index(req: IndexRequest):
        """Trigger re-indexing. Runs in background thread."""
        indexer: Indexer = state["indexer"]
        if indexer is None:
            raise HTTPException(503, "Indexer not initialised")

        t0 = time.monotonic()
        try:
            if req.file_path:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, indexer.reindex_file, req.file_path
                )
            else:
                result = await asyncio.get_event_loop().run_in_executor(
                    None, indexer.index_project
                )
            elapsed = time.monotonic() - t0
            return {
                "files_scanned": result.files_scanned,
                "chunks_added": result.chunks_added,
                "chunks_deleted": result.chunks_deleted,
                "chunks_unchanged": result.chunks_unchanged,
                "errors": result.embed_errors,
                "elapsed_seconds": round(elapsed, 2),
            }
        except Exception as exc:
            raise HTTPException(500, str(exc))


    # Helpers

    def _sse(event_type: str, data: dict) -> str:
        """Format a Server-Sent Event string."""
        return f"event: {event_type}\ndata: {json.dumps(data)}\n\n"

    def _result_to_dict(result) -> dict:
        return {
            "response": result.response,
            "mode": result.mode.value,
            "total_steps": result.total_steps,
            "total_latency_ms": round(result.total_latency_ms, 1),
            "files_modified": result.files_modified,
            "commands_run": result.commands_run,
            "success": result.success,
            "error": result.error,
        }

    return app