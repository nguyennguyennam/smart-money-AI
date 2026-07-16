from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import UploadFile

from app.pipeline.asr_pipeline import ASRPipeline

from .base import BaseExtractor

"""Extractor for voice files.

Used by:
- FastAPI endpoint: receives an UploadFile
- Stream worker: receives raw bytes downloaded from Cloudinary
"""


class VoiceExtractor(BaseExtractor):
    def __init__(self):
        self.pipeline = ASRPipeline()

    async def extract(self, file: "UploadFile"):
        content = await file.read()
        return await self.extract_bytes(content, content_type=file.content_type)

    async def extract_raw(self, file: "UploadFile"):
        return await self.extract(file)

    async def extract_bytes(self, content: bytes, content_type: str | None = None):
        if not content:
            return {"error": "Empty file", "text": ""}
        return await self.pipeline.run(content, content_type)

    async def extract_bytes_raw(self, content: bytes, content_type: str | None = None):
        return await self.extract_bytes(content, content_type)
