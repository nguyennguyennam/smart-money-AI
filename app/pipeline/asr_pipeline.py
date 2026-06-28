from __future__ import annotations

from typing import Any

from app.services.llm import get_llm_service


# Extensions accepted by OpenAI's transcription endpoint. Anything outside this
# set is rejected up front with "Invalid file format".
_OPENAI_SUPPORTED_EXTS = {
    "flac", "m4a", "mp3", "mp4", "mpeg", "mpga", "oga", "ogg", "wav", "webm",
}

_DEFAULT_EXT = "mp3"

# NOTE: every value here must be in _OPENAI_SUPPORTED_EXTS. AAC has no standalone
# OpenAI extension; in practice "audio/aac" payloads are MP4-wrapped, so we label
# them m4a (and byte-sniffing below corrects it when they really are something else).
_EXT_FROM_CONTENT_TYPE = {
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/mp4": "m4a",
    "audio/m4a": "m4a",
    "audio/x-m4a": "m4a",
    "video/mp4": "mp4",
    "audio/ogg": "ogg",
    "application/ogg": "ogg",
    "audio/flac": "flac",
    "audio/x-flac": "flac",
    "audio/webm": "webm",
    "video/webm": "webm",
    "audio/aac": "m4a",
    "audio/x-aac": "m4a",
}


def _ext_from_magic_bytes(data: bytes) -> str | None:
    """Detect the audio container from leading bytes; return an OpenAI ext or None."""
    if len(data) < 12:
        return None

    head = data[:12]

    if head[0:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    if head[0:4] == b"fLaC":
        return "flac"
    if head[0:4] == b"OggS":
        return "ogg"
    if head[0:4] == b"\x1aE\xdf\xa3":  # EBML (webm / mkv)
        return "webm"
    if head[4:8] == b"ftyp":  # ISO-BMFF: mp4 / m4a
        return "m4a"
    if head[0:3] == b"ID3":  # ID3-tagged MP3
        return "mp3"
    if head[0] == 0xFF and head[1] in (0xFB, 0xF3, 0xF2, 0xFA, 0xF4, 0xF5):  # MP3 frame sync
        return "mp3"

    return None


def _ext_from_content_type(content_type: str | None) -> str | None:
    if not content_type:
        return None
    ct = content_type.split(";")[0].strip().lower()
    return _EXT_FROM_CONTENT_TYPE.get(ct)


def filename_from_content_type(content_type: str | None, data: bytes | None = None) -> str:
    """Pick a filename whose extension OpenAI will accept.

    Prefers the actual file signature (magic bytes) over the HTTP content-type
    header, since the header is often wrong or missing for Cloudinary assets.
    Always returns an extension within _OPENAI_SUPPORTED_EXTS.
    """
    ext = None
    if data:
        ext = _ext_from_magic_bytes(data)
    if ext is None:
        ext = _ext_from_content_type(content_type)
    if ext not in _OPENAI_SUPPORTED_EXTS:
        ext = _DEFAULT_EXT
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
        filename = filename_from_content_type(content_type, data)

        try:
            text = await llm.transcribe(data, filename=filename)
        except Exception as e:
            return {"error": f"Transcription failed: {e}", "text": ""}

        text = (text or "").strip()
        if not text:
            return {"error": "No speech detected in audio", "text": ""}

        return {"error": None, "text": text}
