from fastapi import UploadFile
from .base import BaseExtractor
from models.ocr_model import OCRModel
from pipeline.ocr_pipeline import OCRPipeline
from PIL import Image
import asyncio
import io


class ImageExtractor(BaseExtractor):

    def __init__(self):
        self.model = OCRModel()
        self.pipeline = OCRPipeline(self.model)

    async def extract(self, file: UploadFile):
        content = await file.read()  
        img = Image.open(io.BytesIO(content))

        result = await asyncio.to_thread(self.pipeline.run, img)

        return result