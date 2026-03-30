from fastapi import UploadFile
from .base import BaseExtractor
from app.models.ocr_model import OCRModel
from app.pipeline.ocr_pipeline import OCRPipeline
from PIL import Image
import asyncio
import io


class ImageExtractor(BaseExtractor):

    def __init__(self):
        self.model = OCRModel()
        self.pipeline = OCRPipeline(self.model)

    async def extract(self, file: UploadFile):
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