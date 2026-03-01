# app/services/context.py

from fastapi import UploadFile
from app.core.enums import InputType
from app.services.extractor.base import BaseExtractor
from app.services.extractor.image_extractor import ImageExtractor


class Context:

    def __init__(self):
        self.image_extractor = ImageExtractor()

    async def select_extractor(self, input_type: InputType, file: UploadFile):

        extractor: BaseExtractor | None = None

        if input_type == InputType.IMAGE:
            extractor = self.image_extractor

        if extractor is None:
            return {"error": "Unsupported input type"}

        return await extractor.extract(file)
