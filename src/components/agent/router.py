"""
ModelRouter — maps task types to the right LLM model and API.

WHY rule-based routing instead of another LLM call:
  Routing itself must be fast and cheap — adding a model call to decide which
  model to call defeats the purpose. Rule-based classification on TaskType
  is deterministic, zero-latency, and easy to tune.

  The routing table reflects real tradeoffs:
  - Gemini 2.0 Flash: best at multi-step reasoning, architecture decisions,
    planning. Slightly higher latency but handles complex structured output.
  - Groq llama-3.3-70b: fastest large model available, excellent at code
    generation and editing. Sub-second latency on Groq's LPU hardware.
  - Groq llama-3.1-8b: tiny and extremely fast, good enough for retrieval
    decisions and simple Q&A where we just need a quick judgment.
  - Groq llama-guard-3: safety/guardrail model, classifies content not generates it.

DESIGN NOTE on fallback chain:
  If the preferred model's API key is missing, we fall back gracefully:
  Gemini → Groq 70b → Groq 8b → error.
  This means the agent works even if only one key is set.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Optional

import httpx

from .models import TaskType

logger = logging.getLogger(__name__)


# Model descriptors

@dataclass
class ModelConfig:
    model_id: str
    api: str            # "groq" | "gemini"
    max_tokens: int
    temperature: float
    description: str


# Routing table: TaskType → preferred ModelConfig
ROUTING_TABLE: dict[TaskType, ModelConfig] = {
    TaskType.PLANNING:    ModelConfig("gemini-2.0-flash", "gemini", 4096, 0.2, "planning & reasoning"),
    TaskType.CODE_GEN:    ModelConfig("llama-3.3-70b-versatile", "groq", 8192, 0.2, "code generation"),
    TaskType.CODE_EDIT:   ModelConfig("llama-3.3-70b-versatile", "groq", 8192, 0.15, "code editing"),
    TaskType.CODE_REVIEW: ModelConfig("llama-3.3-70b-versatile", "groq", 4096, 0.1, "code review"),
    TaskType.DEBUG:       ModelConfig("llama-3.3-70b-versatile", "groq", 8192, 0.1, "debugging"),
    TaskType.TEST_WRITE:  ModelConfig("llama-3.3-70b-versatile", "groq", 8192, 0.2, "test writing"),
    TaskType.SEARCH:      ModelConfig("llama-3.1-8b-instant",    "groq", 2048, 0.1, "retrieval decisions"),
    TaskType.RUN:         ModelConfig("llama-3.1-8b-instant",    "groq", 1024, 0.0, "command decisions"),
    TaskType.EXPLAIN:     ModelConfig("llama-3.3-70b-versatile", "groq", 4096, 0.3, "explanations"),
    TaskType.SYNTHESISE:  ModelConfig("gemini-2.0-flash",        "gemini", 8192, 0.2, "synthesis"),
}


# LLM call result

@dataclass
class LLMResponse:
    content: str
    model_used: str
    latency_ms: float
    is_error: bool = False


# Router

class ModelRouter:
    """Routes LLM calls to the right model based on task type.

    Usage:
        router = ModelRouter()
        response = router.call(
            task_type=TaskType.CODE_GEN,
            system_prompt="You are a code generator...",
            messages=[{"role": "user", "content": "Write a login function"}],
        )
    """

    def __init__(
        self,
        groq_api_key: Optional[str] = None,
        gemini_api_key: Optional[str] = None,
    ):
        self._groq_key = groq_api_key or os.environ.get("GROQ_API_KEY", "")
        self._gemini_key = (
            gemini_api_key
            or os.environ.get("GEMINI_API_KEY", "")
            or os.environ.get("GOOGLE_API_KEY", "")
        )
        self._client = httpx.Client(timeout=60.0)

    def call(
        self,
        task_type: TaskType,
        system_prompt: str,
        messages: list[dict],
        override_model: Optional[str] = None,
    ) -> LLMResponse:
        """Route a call to the appropriate model for this task type.

        Args:
            task_type:      Determines which model to use.
            system_prompt:  System/instruction prompt.
            messages:       Conversation history [{role, content}].
            override_model: Force a specific model ID (for testing/CLI flag).

        Returns:
            LLMResponse with content and metadata.
        """
        config = ROUTING_TABLE.get(task_type, ROUTING_TABLE[TaskType.CODE_GEN])

        # Resolve which API to use based on key availability
        api, model_id = self._resolve_api(config, override_model)

        logger.debug(
            "Router: %s → %s (%s)", task_type.value, model_id, api
        )

        t0 = time.monotonic()
        try:
            if api == "groq":
                content = self._call_groq(model_id, system_prompt, messages, config.max_tokens, config.temperature)
            else:
                content = self._call_gemini(model_id, system_prompt, messages, config.max_tokens, config.temperature)

            latency = (time.monotonic() - t0) * 1000
            return LLMResponse(content=content, model_used=model_id, latency_ms=latency)

        except Exception as exc:
            latency = (time.monotonic() - t0) * 1000
            logger.error("Router call failed (%s/%s): %s", api, model_id, exc)
            return LLMResponse(
                content=f"[LLM error: {exc}]",
                model_used=model_id,
                latency_ms=latency,
                is_error=True,
            )

    def available_apis(self) -> list[str]:
        apis = []
        if self._groq_key:
            apis.append("groq")
        if self._gemini_key:
            apis.append("gemini")
        return apis


    # Internal

    def _resolve_api(
        self, config: ModelConfig, override_model: Optional[str]
    ) -> tuple[str, str]:
        """Pick the best available API, falling back gracefully."""
        if override_model:
            # Infer API from model name
            api = "gemini" if "gemini" in override_model.lower() else "groq"
            return api, override_model

        preferred_api = config.api
        preferred_model = config.model_id

        # Check if preferred API key exists
        if preferred_api == "groq" and self._groq_key:
            return "groq", preferred_model
        if preferred_api == "gemini" and self._gemini_key:
            return "gemini", preferred_model

        # Fallback: try the other API
        if self._groq_key:
            return "groq", "llama-3.3-70b-versatile"
        if self._gemini_key:
            return "gemini", "gemini-2.0-flash"

        raise RuntimeError(
            "No LLM API key found. Set GROQ_API_KEY or GEMINI_API_KEY."
        )

    def _call_groq(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system_prompt}] + messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        resp = self._client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {self._groq_key}"},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def _call_gemini(
        self,
        model: str,
        system_prompt: str,
        messages: list[dict],
        max_tokens: int,
        temperature: float,
    ) -> str:
        # Convert to Gemini format
        contents = []
        for m in messages:
            role = "user" if m["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": m["content"]}]})

        # Gemini needs at least one user message
        if not contents:
            contents = [{"role": "user", "parts": [{"text": "proceed"}]}]

        payload = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents,
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens,
            },
        }
        resp = self._client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": self._gemini_key},
            json=payload,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]