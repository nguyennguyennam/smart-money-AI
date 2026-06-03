from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from fastapi import UploadFile

class BaseExtractor(ABC):
    @abstractmethod
    async def extract(self, file: "UploadFile") -> str:
        pass
    
