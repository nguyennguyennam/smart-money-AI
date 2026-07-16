from __future__ import annotations

try:
    # Pydantic v2
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover
    # Pydantic v1 fallback
    from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "SmartMoney AI"
    VERSION: str = "1.0.0"

    # Models
    OCR_MODEL_NAME: str = "pytesseract"
    # HuggingFace model id/path for ASR (default: Moonshine Small Streaming)
    ASR_MODEL_NAME: str = "base"  # faster-whisper model size: tiny|base|small|medium|large-v2|large-v3
    ASR_DEVICE: str = "cpu"  # cpu | cuda
    ASR_COMPUTE_TYPE: str = "float32"  # float16 | float32 (float16 only when ASR_DEVICE=cuda)
    ASR_TRUST_REMOTE_CODE: bool = False
    ASR_LANGUAGE: str | None = None  # None / "auto" = detect from audio; or "vi", "en", etc.
    ASR_TASK: str | None = "transcribe"

    # Redis (input stream)
    # Example: redis://localhost:6379/0
    REDIS_STREAM_URL: str = "redis://localhost:6379/0"
    REDIS_STREAM_KEY: str = "stream:jobs:0"
    REDIS_CONSUMER_GROUP: str = "smartmoney-ai"
    REDIS_CONSUMER_NAME: str = "smartmoney-ai-1"
    REDIS_BLOCK_MS: int = 5_000
    REDIS_CLAIM_IDLE_MS: int = 60_000
    REDIS_MAX_RETRIES: int = 5
    REDIS_DEAD_LETTER_STREAM_KEY: str = ""

    # Redis (result storage) — can be a separate instance/host.
    # Default points to a separate logical DB for convenience.
    REDIS_RESULT_URL: str = "redis://localhost:6379/1"
    REDIS_RESULT_STREAM_KEY: str = "result_stream"
    RESULT_TTL_SECONDS: int = 180  #  3 mins

    # Download limits
    DOWNLOAD_MAX_BYTES: int = 25 * 1024 * 1024
    DOWNLOAD_TIMEOUT_SECONDS: float = 30.0

    # LLM providers
    OPENAI_API_KEY: str | None = None
    OPENAI_BASE_URL: str | None = None
    OPENAI_MODEL: str = "gpt-5-nano"
    OPENAI_VISION_MODEL: str = "gpt-5-nano"
    OPENAI_TRANSCRIBE_MODEL: str = "gpt-4o-mini-transcribe"

    GEMINI_API_KEY: str | None = None
    GEMINI_MODEL: str = "gemini-2.0-flash"

    NVIDIA_API_KEY: str | None = None
    NVIDIA_MODEL: str = "nvidia-deepseek-v4-pro"

    DEEPSEEK_API_KEY: str | None = None
    OPENROUTER_API_KEY: str | None = None

    class Config:
        env_file = ".env"
        file_encoding = "utf-8"


settings = Settings()
