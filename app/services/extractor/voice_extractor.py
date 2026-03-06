from fastapi import UploadFile
from .base import BaseExtractor

from pipeline import asr_pipeline
from models import asr_model

import asyncio

'''
    Extractor for voice files. It uses ASRPipeline to process the audio and extract text from it.
'''

class VoiceExtractor(BaseExtractor):
    def __init__(self):
        self.model = asr_model.ASRModel()
        self.pipeline = asr_pipeline.ASRPipeline(self.model)

    async def extract(self, file: UploadFile):
        result = await asyncio.to_thread(self.pipeline.run, file.file)

        return result