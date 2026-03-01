from abc import ABC, abstractmethod
from fastapi import UploadFile

class BaseExtractor(ABC):
    @abstractmethod
    async def extract (self, file: UploadFile) -> str:
        pass
    
