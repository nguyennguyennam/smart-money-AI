from __future__ import annotations

import asyncio
import io
from typing import TYPE_CHECKING

from PIL import Image

from .base import BaseExtractor
from ...pipeline.ocr_pipeline import OCRPipeline
from ...services.llm import get_llm_service
from ...services.llm.financial import ocr_classify_extract

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import UploadFile


_PIL_FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
    "GIF": "image/gif",
    "BMP": "image/bmp",
    "TIFF": "image/tiff",
}


def _mime_from_pil_format(fmt: str | None) -> str:
    if not fmt:
        return "image/jpeg"
    return _PIL_FORMAT_TO_MIME.get(fmt.upper(), "image/jpeg")


class ImageExtractor(BaseExtractor):
    """Validates an image then hands it to the OCR LLM call.

    `extract_bytes` returns the legacy `{error, text}` shape so the FastAPI
    `/process` endpoint keeps working. The Redis worker uses
    `analyze_bytes` instead, which returns text + category + type + expense
    from a single gpt-5-nano vision call.
    """

    def __init__(self):
        self.pipeline = OCRPipeline()

    async def extract(self, file: "UploadFile"):
        content = await file.read()
        return await self.extract_bytes(content)

    async def extract_bytes(self, content: bytes):
        result = await self.analyze_bytes(content)
        return {"error": result.get("error"), "text": result.get("text", "")}

    async def analyze_bytes(self, content: bytes) -> dict:
        if not content:
            return {"error": "Empty file", "text": "", "category": "OTHER", "type": "EXPENSE", "expense": 50000}

        try:
            img = Image.open(io.BytesIO(content))
            mime_type = _mime_from_pil_format(img.format)
        except Exception:
            return {"error": "Invalid image", "text": "", "category": "OTHER", "type": "EXPENSE", "expense": 50000}

        guard = await asyncio.to_thread(self.pipeline.run, img)
        if not guard.get("ok"):
            return {
                "error": guard.get("error") or "Image rejected by preprocessing",
                "text": "",
                "category": "OTHER",
                "type": "EXPENSE",
                "expense": 50000,
            }

        llm = get_llm_service()
        return await ocr_classify_extract(llm, content, mime_type)
