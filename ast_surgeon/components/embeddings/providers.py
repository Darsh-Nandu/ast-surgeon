"""
Embedding provider abstraction + concrete implementations.

Supported providers:
  ┌─────────────┬──────────────────────────────┬───────────┬──────────────────┐
  │ Key         │ Model                        │ Dims      │ Best for         │
  ├─────────────┼──────────────────────────────┼───────────┼──────────────────┤
  │ voyage      │ voyage-code-2                │ 1536      │ Code (best)      │
  │ openai      │ text-embedding-3-large       │ 1536      │ General, popular │
  │ cohere      │ embed-english-v3.0           │ 1024      │ Great recall     │
  │ gemini      │ text-embedding-004           │ 768       │ Google ecosystem │
  │ mistral     │ mistral-embed                │ 1024      │ Fast + cheap     │
  │ local       │ configurable (sentence-xfmr) │ varies    │ Offline / free   │
  └─────────────┴──────────────────────────────┴───────────┴──────────────────┘

All providers implement the same interface. Switching providers is one line:
    provider = get_provider("cohere")   # was "voyage"

DESIGN NOTE on dimensions:
  Each provider exposes a `dimension` property returning its actual output size.
  The VectorStore is initialised with `store.ensure_collection(provider.dimension)`.
  No global EMBED_DIM constant, no zero-padding, no silent shape mismatches.

DESIGN NOTE on asymmetric embeddings:
  Several providers (Voyage, Cohere, Gemini) support separate document vs query
  embeddings. We exploit this everywhere - chunks are embedded as documents,
  search queries as queries. This measurably improves recall on code.

DESIGN NOTE on env vars:
  Provider selection: AST_SURGEON_EMBEDDING_PROVIDER
  API keys follow each provider's conventional name (VOYAGE_API_KEY, etc.)
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional

import httpx

from ..chunker.models import CodeChunk


# Abstract base

class EmbeddingProvider(ABC):
    """Interface every embedding provider must implement."""

    @abstractmethod
    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        """Return one embedding vector per chunk, in the same order."""

    @abstractmethod
    def embed_query(self, query: str) -> list[float]:
        """Embed a search query (may use a different model head than embed_chunks)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Dimension of vectors this provider produces."""


class EmbeddingError(RuntimeError):
    """Raised when an embedding API call fails unrecoverably after retries."""


# Shared text preparation

def _chunk_to_text(chunk: CodeChunk) -> str:
    """
    Prepare a chunk's text for embedding.

    For code chunks, prepends a brief header:
        "python function Calculator.add\n\ndef add(self, a, b): ..."

    This improves retrieval for natural-language queries like "how does add work?"
    because embedding models see the type and name alongside the body.
    Prose chunks (TEXT_BLOCK) are embedded as-is.
    """
    if chunk.chunk_type.value == "text_block":
        return chunk.content

    header = f"{chunk.language} {chunk.chunk_type.value}"
    if chunk.name:
        header += f" {chunk.name}"
    return f"{header}\n\n{chunk.content}"


# Voyage AI

class VoyageProvider(EmbeddingProvider):
    """
    Voyage AI - voyage-code-2 (best code retrieval, 1536 dims).

    Env: VOYAGE_API_KEY
    Docs: https://docs.voyageai.com/
    """

    API_URL    = "https://api.voyageai.com/v1/embeddings"
    MODEL      = "voyage-code-2"
    BATCH_SIZE = 128
    _DIMENSION = 1536

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY", "")
        if not self._api_key:
            raise EmbeddingError("VOYAGE_API_KEY not set")
        self._model = model or self.MODEL
        self._client = httpx.Client(timeout=60.0)

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        return self._embed([_chunk_to_text(c) for c in chunks], input_type="document")

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], input_type="query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            resp = self._client.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={"model": self._model, "input": batch, "input_type": input_type},
            )
            if resp.status_code != 200:
                raise EmbeddingError(f"Voyage API {resp.status_code}: {resp.text[:200]}")
            for item in sorted(resp.json()["data"], key=lambda x: x["index"]):
                out.append(item["embedding"])
        return out


# OpenAI

class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI text-embedding-3-large (1536 dims via server-side truncation).

    Env: OPENAI_API_KEY
    Docs: https://platform.openai.com/docs/guides/embeddings
    """

    API_URL    = "https://api.openai.com/v1/embeddings"
    MODEL      = "text-embedding-3-large"
    BATCH_SIZE = 100
    _DIMENSION = 1536

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise EmbeddingError("OPENAI_API_KEY not set")
        self._model = model or self.MODEL
        self._client = httpx.Client(timeout=60.0)

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        return self._embed([_chunk_to_text(c) for c in chunks])

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            resp = self._client.post(
                self.API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "input": batch,
                    "dimensions": self._DIMENSION,
                    "encoding_format": "float",
                },
            )
            if resp.status_code != 200:
                raise EmbeddingError(f"OpenAI API {resp.status_code}: {resp.text[:200]}")
            for item in sorted(resp.json()["data"], key=lambda x: x["index"]):
                out.append(item["embedding"])
        return out


# Cohere

class CohereProvider(EmbeddingProvider):
    """
    Cohere embed-english-v3.0 (1024 dims, excellent code + text retrieval).

    Env: COHERE_API_KEY
    Docs: https://docs.cohere.com/reference/embed
    """

    API_URL    = "https://api.cohere.ai/v1/embed"
    MODEL      = "embed-english-v3.0"
    BATCH_SIZE = 96   # Cohere max is 96 texts per call
    _DIMENSION = 1024

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or os.environ.get("COHERE_API_KEY", "")
        if not self._api_key:
            raise EmbeddingError("COHERE_API_KEY not set")
        self._model = model or self.MODEL
        self._client = httpx.Client(timeout=60.0)

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        return self._embed([_chunk_to_text(c) for c in chunks], input_type="search_document")

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], input_type="search_query")[0]

    def _embed(self, texts: list[str], input_type: str) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            resp = self._client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self._model,
                    "texts": batch,
                    "input_type": input_type,
                    "embedding_types": ["float"],
                },
            )
            if resp.status_code != 200:
                raise EmbeddingError(f"Cohere API {resp.status_code}: {resp.text[:200]}")
            data = resp.json()
            # v2 response: embeddings.float is a list of lists
            embeddings = data.get("embeddings", {})
            vecs = embeddings.get("float", embeddings) if isinstance(embeddings, dict) else embeddings
            out.extend(vecs)
        return out


# Google Gemini

class GeminiProvider(EmbeddingProvider):
    """
    Google Gemini text-embedding-004 (768 dims, great general + code recall).

    Env: GEMINI_API_KEY  (get one free at https://aistudio.google.com)
    Docs: https://ai.google.dev/api/embeddings
    """

    API_BASE   = "https://generativelanguage.googleapis.com/v1beta"
    MODEL      = "text-embedding-004"
    BATCH_SIZE = 100  # Gemini batchEmbedContents max
    _DIMENSION = 768

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None):
        self._api_key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise EmbeddingError("GEMINI_API_KEY not set")
        self._model = model or self.MODEL
        self._client = httpx.Client(timeout=60.0)

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        return self._embed([_chunk_to_text(c) for c in chunks], task="RETRIEVAL_DOCUMENT")

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query], task="RETRIEVAL_QUERY")[0]

    def _embed(self, texts: list[str], task: str) -> list[list[float]]:
        out = []
        url = f"{self.API_BASE}/models/{self._model}:batchEmbedContents?key={self._api_key}"

        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            requests = [
                {
                    "model": f"models/{self._model}",
                    "content": {"parts": [{"text": t}]},
                    "taskType": task,
                }
                for t in batch
            ]
            resp = self._client.post(url, json={"requests": requests})
            if resp.status_code != 200:
                raise EmbeddingError(f"Gemini API {resp.status_code}: {resp.text[:200]}")
            for item in resp.json().get("embeddings", []):
                out.append(item["values"])
        return out


# Mistral

class MistralProvider(EmbeddingProvider):
    """
    Mistral AI mistral-embed (1024 dims, fast and cost-effective).

    Env: MISTRAL_API_KEY
    Docs: https://docs.mistral.ai/api/#tag/embeddings
    """

    API_URL    = "https://api.mistral.ai/v1/embeddings"
    MODEL      = "mistral-embed"
    BATCH_SIZE = 512  # Mistral handles large batches well
    _DIMENSION = 1024

    def __init__(self, api_key: Optional[str] = None):
        self._api_key = api_key or os.environ.get("MISTRAL_API_KEY", "")
        if not self._api_key:
            raise EmbeddingError("MISTRAL_API_KEY not set")
        self._client = httpx.Client(timeout=60.0)

    @property
    def dimension(self) -> int:
        return self._DIMENSION

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        return self._embed([_chunk_to_text(c) for c in chunks])

    def embed_query(self, query: str) -> list[float]:
        return self._embed([query])[0]

    def _embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            resp = self._client.post(
                self.API_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json={"model": self.MODEL, "input": batch},
            )
            if resp.status_code != 200:
                raise EmbeddingError(f"Mistral API {resp.status_code}: {resp.text[:200]}")
            for item in sorted(resp.json()["data"], key=lambda x: x["index"]):
                out.append(item["embedding"])
        return out


# Local (sentence-transformers)

class LocalProvider(EmbeddingProvider):
    """
    Local sentence-transformers - zero API cost, works fully offline.

    Install: pip install "ast-surgeon[local-embed]"

    Default model: all-MiniLM-L6-v2  (384 dims, fast)
    Better models:
        "all-mpnet-base-v2"               - 768 dims, higher quality
        "BAAI/bge-m3"                     - 1024 dims, multilingual
        "nomic-ai/nomic-embed-text-v1"    - 768 dims, great for code

    DESIGN NOTE on dimensions:
        Each sentence-transformers model has its own dimension.
        We read it from the model at init time - no zero-padding.
        This means switching models requires a full re-index (different dims).
    """

    DEFAULT_MODEL = "all-MiniLM-L6-v2"

    def __init__(self, model_name: Optional[str] = None):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers not installed.\n"
                "Run: pip install 'ast-surgeon[local-embed]'"
            )
        self._model_name = model_name or os.environ.get(
            "AST_SURGEON_LOCAL_MODEL", self.DEFAULT_MODEL
        )
        try:
            self._model = SentenceTransformer(self._model_name)
        except Exception as exc:
            raise EmbeddingError(
                f"Failed to load local embedding model {self._model_name!r}: {exc}"
            ) from exc
        self._dim = self._model.get_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_chunks(self, chunks: list[CodeChunk]) -> list[list[float]]:
        texts = [_chunk_to_text(c) for c in chunks]
        return self._model.encode(texts, normalize_embeddings=True).tolist()

    def embed_query(self, query: str) -> list[float]:
        return self._model.encode([query], normalize_embeddings=True)[0].tolist()


# Factory

_PROVIDERS = {
    "voyage":   VoyageProvider,
    "openai":   OpenAIProvider,
    "cohere":   CohereProvider,
    "gemini":   GeminiProvider,
    "mistral":  MistralProvider,
    "local":    LocalProvider,
}

_AUTO_ORDER = [
    ("voyage",  "VOYAGE_API_KEY"),
    ("openai",  "OPENAI_API_KEY"),
    ("cohere",  "COHERE_API_KEY"),
    ("gemini",  "GEMINI_API_KEY"),
    ("mistral", "MISTRAL_API_KEY"),
]


def get_provider(name: Optional[str] = None, **kwargs) -> EmbeddingProvider:
    """
    Build an EmbeddingProvider by name.

    Resolution order:
      1. `name` argument
      2. AST_SURGEON_EMBEDDING_PROVIDER environment variable
      3. Auto-detect: first provider whose API key env var is set
      4. LocalProvider (offline fallback - always works)

    Args:
        name:     "voyage" | "openai" | "cohere" | "gemini" | "mistral" | "local"
        **kwargs: Passed to the provider constructor (e.g. api_key=, model=).

    Examples:
        get_provider()                      # auto-detect
        get_provider("cohere")              # explicit
        get_provider("local", model_name="BAAI/bge-m3")
        get_provider("openai", model="text-embedding-3-small")
    """
    chosen = name or os.environ.get("AST_SURGEON_EMBEDDING_PROVIDER", "auto")

    if chosen != "auto":
        cls = _PROVIDERS.get(chosen.lower())
        if cls is None:
            raise EmbeddingError(
                f"Unknown provider: {chosen!r}. "
                f"Choose from: {list(_PROVIDERS)}"
            )
        return cls(**kwargs)

    # Auto mode: try each provider in priority order
    for provider_name, env_key in _AUTO_ORDER:
        if os.environ.get(env_key):
            try:
                return _PROVIDERS[provider_name](**kwargs)
            except EmbeddingError:
                continue

    # Final fallback: local model (always available if sentence-transformers installed)
    return LocalProvider(**kwargs)


def list_providers() -> list[str]:
    """Return the names of all supported embedding providers."""
    return list(_PROVIDERS.keys())