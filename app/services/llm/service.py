from __future__ import annotations

import asyncio
import random
from enum import Enum
from functools import lru_cache
from typing import Any

from click import prompt
from google import genai
try:  # optional
    from openai import AsyncOpenAI  # type: ignore
except Exception:  # pragma: no cover
    AsyncOpenAI = None  # type: ignore

from app.core.config import settings


class LLMProvider(str, Enum):
    OPENAI = "openai"
    GEMINI = "gemini"
    DEEPSEEK = "deepseek"  # NVIDIA DeepSeek (via OpenAI-compatible API)


class LLMService:
    """Simple multi-provider LLM service.

    Use generate() and pass provider-specific keyword arguments via **kwargs.
    """

    def __init__(self) -> None:
        self._openai_client = self._build_openai_client()
        self._gemini_client = self._build_gemini_client()
        self._deepseek_client = self._build_deepseek_client()

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
    
    @staticmethod
    def _build_deepseek_client():
        if not settings.OPENROUTER_API_KEY:
            return None

        if AsyncOpenAI is None:
            raise ImportError("OpenAI SDK not installed")

        return AsyncOpenAI(
            api_key=settings.OPENROUTER_API_KEY,
            base_url="https://openrouter.ai/api/v1",
        )
    
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
        
        if provider_name == LLMProvider.DEEPSEEK.value:
            return await self._generate_deepseek(
                prompt=prompt,
                model=model,
                **kwargs,
            )

        raise ValueError("Unsupported provider. Use 'openai', 'gemini', or 'deepseek'.")
    
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
            
        try:
            result = await asyncio.to_thread(_call)

            text = getattr(result, "text", None)

            if isinstance(text, str) and text.strip():
                return text.strip()

            return str(result)

        except Exception as e:

            error_text = str(e)

            if (
                "RESOURCE_EXHAUSTED" in error_text
                or "429" in error_text
                or "UNAVAILABLE" in error_text
                or "503" in error_text
                or "PERMISSION_DENIED" in error_text   # 🔥 thêm dòng này
                or "NOT_FOUND" in error_text           # 🔥 thêm dòng này
            ):
                return await self._generate_deepseek(
                    prompt=prompt,
                    model=model,
                    **kwargs,
                )

            raise
            
    async def _generate_deepseek(
        self,
        prompt: str,
        model: str | None = None,
        **kwargs,
    ) -> str:

        if self._deepseek_client is None:
            raise ValueError("DEEPSEEK_API_KEY is not set")

        # ✅ đổi model
        target_model = model or "deepseek/deepseek-chat"

        max_retries = 5
        base_delay = 1

        for attempt in range(max_retries):
            try:
                response = await self._deepseek_client.chat.completions.create(
                    model=target_model,
                    messages=[
                        {"role": "system", "content": "You are a helpful financial assistant."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=kwargs.get("temperature", 0.7),
                    top_p=kwargs.get("top_p", 0.95),
                    max_tokens=kwargs.get("max_tokens", 16384),
                    extra_body={
                        "chat_template_kwargs": {
                            "thinking": False  # giống config NVIDIA
                        }
                    }
                )

                return response.choices[0].message.content.strip()

            except Exception as e:
                error_text = str(e)
                print("[DeepSeek ERROR]:", error_text)  # 👈 thêm dòng này

                if "429" in error_text or "Too Many Requests" in error_text:
                    wait_time = base_delay * (2 ** attempt) + random.uniform(0, 1)
                    print(f"[DeepSeek] Rate limit → retry in {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
                    continue

                # ❌ các lỗi này retry là vô nghĩa
                if any(x in error_text for x in ["401", "403", "404", "Unauthorized", "Not Found"]):
                    raise Exception(f"DeepSeek config error: {error_text}")

                raise

        raise Exception("DeepSeek failed after retries")

@lru_cache(maxsize=1)
def get_llm_service() -> LLMService:
    return LLMService()
