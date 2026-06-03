from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import UploadFile

from app.models.asr_model import ASRModel
from app.pipeline.asr_pipeline import ASRPipeline
from app.pipeline.asr_pipeline import load_audio_any

from .base import BaseExtractor

"""Extractor for voice files.

Used by:
- FastAPI endpoint: receives an UploadFile
- Stream worker: receives raw bytes downloaded from Cloudinary
"""


class VoiceExtractor(BaseExtractor):
    def __init__(self):
        self.model = ASRModel()
        self.pipeline = ASRPipeline(self.model)

    async def extract(self, file: "UploadFile"):
        content = await file.read()
        return await self.extract_bytes(content, content_type=file.content_type)

    async def extract_raw(self, file: "UploadFile"):
        """Extract text by decoding the audio then feeding it directly to the model.

        This bypasses pipeline preprocessing (noise reduction + volume normalization).
        """
        content = await file.read()
        return await self.extract_bytes_raw(content, content_type=file.content_type)

    async def extract_bytes(self, content: bytes, content_type: str | None = None):
        if not content:
            return {"error": "Empty file", "text": ""}

        result = await asyncio.to_thread(self.pipeline.run, content, content_type)
        return result

    async def extract_bytes_raw(self, content: bytes, content_type: str | None = None):
        if not content:
            return {"error": "Empty file", "text": ""}

        def _run_raw() -> dict[str, str | None]:
            audio_array, _sr = load_audio_any(content, content_type)
            segments = self.model.transcribe(audio_array)
            text = " ".join([seg.text for seg in segments])
            return {"error": None, "text": text}

        return await asyncio.to_thread(_run_raw)
