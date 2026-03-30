from __future__ import annotations

import asyncio


class VoiceExtractor:
    async def extract_bytes(self, content: bytes) -> dict:
        if not content:
            return {"error": "Empty file", "text": ""}

        try:
            # Lazy import so OCR-only deployments don't require a working ASR native stack.
            from app.models.asr_model import get_asr_model

            model = get_asr_model()
            text = await asyncio.to_thread(model.transcribe_bytes, content)
            return {"error": None, "text": text}
        except Exception as e:
            return {"error": str(e), "text": ""}
