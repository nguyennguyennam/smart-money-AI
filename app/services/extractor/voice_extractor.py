from __future__ import annotations

import asyncio

from fastapi import UploadFile

from app.models.asr_model import ASRModel
from app.pipeline.asr_pipeline import ASRPipeline

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

    async def extract(self, file: UploadFile):
        content = await file.read()
        return await self.extract_bytes(content, content_type=file.content_type)

    async def extract_bytes(self, content: bytes, content_type: str | None = None):
        if not content:
            return {"error": "Empty file", "text": ""}

        result = await asyncio.to_thread(self.pipeline.run, content, content_type)
        return result