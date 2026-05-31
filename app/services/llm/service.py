from __future__ import annotations

import asyncio
from enum import Enum
from functools import lru_cache
from typing import Any

from google import genai
try:  # optional
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

from app.core.config import settings


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"


class LLMService:
    """Simple multi-provider LLM service.

    Use generate() and pass provider-specific keyword arguments via **kwargs.
    """

    def __init__(self) -> None:
        self._openai_client = self._build_openai_client()
        self._gemini_client = self._build_gemini_client()

    @staticmethod
    def _build_openai_client() -> Any | None:
        if not settings.OPENAI_API_KEY:
            return None

        if AsyncOpenAI is None:
            raise ImportError("OpenAI SDK not installed. Install 'openai' to use provider='openai'.")

        client_kwargs: dict[str, Any] = {"api_key": settings.OPENAI_API_KEY}
        if settings.OPENAI_BASE_URL:
            client_kwargs["base_url"] = settings.OPENAI_BASE_URL

        return AsyncOpenAI(**client_kwargs)

    @staticmethod
    def _build_gemini_client() -> Any | None:
        if not settings.GEMINI_API_KEY:
            return None
        # google-genai SDK
        return genai.Client(api_key=settings.GEMINI_API_KEY)

    async def generate(
        self,
        prompt: str,
        provider: str | LLMProvider = LLMProvider.GEMINI,
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        if not prompt or not isinstance(prompt, str):
            raise ValueError("prompt must be a non-empty string")

        provider_name = provider.value if isinstance(provider, LLMProvider) else str(provider).strip().lower()

        if provider_name == LLMProvider.OPENAI.value:
            return await self._generate_openai(prompt=prompt, model=model, **kwargs)

        if provider_name == LLMProvider.GEMINI.value:
            return await self._generate_gemini(prompt=prompt, model=model, **kwargs)

        raise ValueError("Unsupported provider. Use 'openai' or 'gemini'.")

    async def _generate_openai(self, prompt: str, model: str | None = None, **kwargs: Any) -> str:
        if self._openai_client is None:
            raise ValueError("OPENAI_API_KEY is not set")

        target_model = model or settings.OPENAI_MODEL
        response = await self._openai_client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

        if not response.choices:
            return ""

        message = response.choices[0].message
        return (message.content or "").strip()

    async def _generate_gemini(self, prompt: str, model: str | None = None, **kwargs: Any) -> str:
        if self._gemini_client is None:
            raise ValueError("GEMINI_API_KEY is not set")

        target_model = model or settings.GEMINI_MODEL
        client = self._gemini_client

        def _call() -> Any:
            try:
                return client.models.generate_content(model=target_model, contents=prompt, **kwargs)
            except TypeError:
                # If kwargs don't match this SDK version, still try a minimal call.
                return client.models.generate_content(model=target_model, contents=prompt)

        result = await asyncio.to_thread(_call)
        text = getattr(result, "text", None)

        if isinstance(text, str) and text.strip():
            return text.strip()

        return str(result)
    

@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()
