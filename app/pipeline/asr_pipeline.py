from __future__ import annotations

import io
import os
import tempfile
from typing import Any

import numpy as np
from faster_whisper.audio import decode_audio

from app.services.extractor.voice import noise_reduce, volume_normalize


def _suffix_from_content_type(content_type: str | None) -> str:
    if not content_type:
        return ".bin"

    ct = content_type.split(";")[0].strip().lower()
    # Common audio content-types from Cloudinary
    if ct in {"audio/wav", "audio/x-wav", "audio/wave"}:
        return ".wav"
    if ct in {"audio/mpeg", "audio/mp3"}:
        return ".mp3"
    if ct in {"audio/mp4", "audio/m4a", "audio/x-m4a"}:
        return ".m4a"
    if ct in {"audio/ogg", "application/ogg"}:
        return ".ogg"
    if ct in {"audio/flac"}:
        return ".flac"
    return ".bin"


def _load_audio_any(voice: Any, content_type: str | None) -> tuple[np.ndarray, int]:
    """Load audio as mono float array at 16kHz.

    Supports:
    - bytes/bytearray/memoryview (worker)
    - file-like objects with .read() (UploadFile.file)
    - filesystem paths
    """
    # 1) bytes-like => try BytesIO first
    if isinstance(voice, (bytes, bytearray, memoryview)):
        data = bytes(voice)
        try:
            audio = decode_audio(io.BytesIO(data), sampling_rate=16000, split_stereo=False)
            return audio, 16000
        except Exception:
            # Fall back to a temp file (better MP3/M4A compatibility)
            suffix = _suffix_from_content_type(content_type)
            tmp_path: str | None = None
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                    tmp_path = f.name
                    f.write(data)
                audio = decode_audio(tmp_path, sampling_rate=16000, split_stereo=False)
                return audio, 16000
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

    # 2) file-like object: ensure we're at start
    if hasattr(voice, "read"):
        try:
            if hasattr(voice, "seek"):
                voice.seek(0)
        except Exception:
            pass
        audio = decode_audio(voice, sampling_rate=16000, split_stereo=False)
        return audio, 16000

    # 3) path-like (str/Path)
    audio = decode_audio(str(voice), sampling_rate=16000, split_stereo=False)
    return audio, 16000


class ASRPipeline:
    def __init__(self, model):
        self.model = model

    def run(self, voice: Any, content_type: str | None = None):
        audio_array, _sr = _load_audio_any(voice, content_type)

        audio_array = noise_reduce.reduce_noise(audio_array)
        audio_array = volume_normalize.normalize_volume(audio_array, volume=0.1)

        segments = self.model.transcribe(audio_array)
        text = " ".join([seg.text for seg in segments])

        return {"error": None, "text": text}