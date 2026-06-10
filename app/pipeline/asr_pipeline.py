from __future__ import annotations

from typing import Any

from app.services.llm import get_llm_service


_EXT_FROM_CONTENT_TYPE = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "audio/ogg": "ogg",
    "application/ogg": "ogg",
    "audio/flac": "flac",
    "audio/webm": "webm",
    "audio/aac": "aac",
}


def filename_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return "audio.mp3"
    ct = content_type.split(";")[0].strip().lower()
    ext = _EXT_FROM_CONTENT_TYPE.get(ct, "mp3")
    return f"audio.{ext}"


class ASRPipeline:
    """Thin orchestrator that hands raw audio bytes to the transcribe LLM call.

    No local decoding / noise reduction / volume normalization — the model
    handles raw audio formats supported by OpenAI directly.
    """

    def __init__(self, model=None):
        self.model = model

    async def run(self, voice: Any, content_type: str | None = None) -> dict:
        if voice is None:
            return {"error": "Empty file", "text": ""}

        if not isinstance(voice, (bytes, bytearray, memoryview)):
            return {"error": "Audio payload must be bytes", "text": ""}

        data = bytes(voice)
        if not data:
            return {"error": "Audio file is empty or has no content", "text": ""}

        llm = get_llm_service()
        filename = filename_from_content_type(content_type)

        try:
            text = await llm.transcribe(data, filename=filename)
        except Exception as e:
            return {"error": f"Transcription failed: {e}", "text": ""}

        text = (text or "").strip()
        if not text:
            return {"error": "No speech detected in audio", "text": ""}

        return {"error": None, "text": text}
