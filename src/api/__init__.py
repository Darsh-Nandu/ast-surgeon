"""API package - FastAPI server with SSE streaming."""
from .server import create_app
__all__ = ["create_app"]