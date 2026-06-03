# app/services/context.py

from fastapi import UploadFile
from core.enums import InputType
from services.extractor.base import BaseExtractor
from services.extractor.image_extractor import ImageExtractor
from services.extractor.voice_extractor import VoiceExtractor


class Context:

    def __init__(self):
        self.image_extractor = ImageExtractor()
        self.voice_extractor = VoiceExtractor()

    async def select_extractor(self, input_type: InputType, file: UploadFile):

        extractor: BaseExtractor | None = None

        if input_type == InputType.IMAGE:
            extractor = self.image_extractor
        
        if input_type == InputType.VOICE:
            extractor = self.voice_extractor

        if extractor is None:
            return {"error": "Unsupported input type"}

        return await extractor.extract(file)
