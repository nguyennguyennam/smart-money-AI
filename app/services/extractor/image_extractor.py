from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

from PIL import Image

from .base import BaseExtractor
from app.models.ocr_model import OCRModel
from app.pipeline.ocr_pipeline import OCRPipeline

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import UploadFile


class ImageExtractor(BaseExtractor):

    def __init__(self):
        self.model = OCRModel()
        self.pipeline = OCRPipeline(self.model)

    async def extract(self, file: "UploadFile"):
        content = await file.read()
        return await self.extract_bytes(content)

    async def extract_bytes(self, content: bytes):
        if not content:
            return {"error": "Empty file", "text": ""}

        try:
            img = Image.open(io.BytesIO(content))
        except Exception:
            return {"error": "Invalid image", "text": ""}

        result = await asyncio.to_thread(self.pipeline.run, img)
        return result