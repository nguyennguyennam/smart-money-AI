from __future__ import annotations

try:
    # Pydantic v2
    from pydantic_settings import BaseSettings
except Exception:  # pragma: no cover
    # Pydantic v1 fallback
    from pydantic import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "SmartMoney AI"
    VERSION: str = "1.0.0"

    # Models
    OCR_MODEL_NAME: str = "pytesseract"
    # Name/path for faster-whisper model (e.g. "base", "small", or a local path)
    ASR_MODEL_NAME: str = "whisper"
    ASR_DEVICE: str = "cpu"  # cpu | cuda
    ASR_COMPUTE_TYPE: str = "int8"  # int8 | float16 | float32

    # Redis (input stream)
    # Example: redis://localhost:6379/0
    REDIS_STREAM_URL: str = "redis://localhost:6379/0"
    REDIS_STREAM_KEY: str = "stream:jobs:0"
    REDIS_CONSUMER_GROUP: str = "smartmoney-ai"
    REDIS_CONSUMER_NAME: str = "smartmoney-ai-1"
    REDIS_BLOCK_MS: int = 5_000
    REDIS_CLAIM_IDLE_MS: int = 60_000
    REDIS_MAX_RETRIES: int = 10
    REDIS_DEAD_LETTER_STREAM_KEY: str = ""

    # Redis (result storage) — can be a separate instance/host.
    # Default points to a separate logical DB for convenience.
    REDIS_RESULT_URL: str = "redis://localhost:6379/1"
    RESULT_TTL_SECONDS: int = 86_400  # 1 day

    # Download limits
    DOWNLOAD_MAX_BYTES: int = 25 * 1024 * 1024
    DOWNLOAD_TIMEOUT_SECONDS: float = 30.0

    class Config:
        env_file = ".env"
        file_encoding = "utf-8"


settings = Settings()