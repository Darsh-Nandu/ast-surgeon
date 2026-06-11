"""
Memory system for Sovereign-Code — three layers:

  Layer 1: WorkingMemory    — per-SubAgent RAM (task lifetime)
  Layer 2: EpisodicMemory   — per-session disk store (whole session)
  Layer 3: VectorStore      — Qdrant semantic search (session-scoped)

The MemoryCoordinator wires layers 1 and 2 together.
Layer 3 is managed by AgentLoop via VectorStore + EmbeddingPipeline.

Legacy (kept for backward compat):
  SessionStore   — original flat-file session persistence
  SessionManager — original session lifecycle manager
"""
from .working_memory import WorkingMemory
from .episodic_memory import EpisodicMemory
from .coordinator import MemoryCoordinator
from .session_store import SessionStore
from .session_manager import SessionManager

__all__ = [
    "WorkingMemory",
    "EpisodicMemory",
    "MemoryCoordinator",
    "SessionStore",
    "SessionManager",
]