from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

import numpy as np
import imageio_ffmpeg

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


def load_audio_any(voice: Any, content_type: str | None = None) -> tuple[np.ndarray, int]:
    """Load audio as mono float array at 16kHz.

    Supports:
    - bytes/bytearray/memoryview (worker)
    - file-like objects with .read() (UploadFile.file)
    - filesystem paths
    """
    def _decode_with_ffmpeg(path: str) -> tuple[np.ndarray, int]:
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        # Decode to mono 16kHz float32 stream.
        cmd = [
            ffmpeg,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            path,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "f32le",
            "pipe:1",
        ]
        p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if p.returncode != 0:
            err = p.stderr.decode("utf-8", errors="replace").strip()
            raise ValueError(f"ffmpeg decode failed: {err}")

        audio = np.frombuffer(p.stdout, dtype=np.float32)
        return audio, 16000

    # 1) bytes-like => decode via temp file (robust across formats)
    if isinstance(voice, (bytes, bytearray, memoryview)):
        data = bytes(voice)
        suffix = _suffix_from_content_type(content_type)
        tmp_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
                tmp_path = f.name
                f.write(data)
            return _decode_with_ffmpeg(tmp_path)
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
        # Read all bytes and reuse bytes-path decoder.
        data = voice.read()
        return load_audio_any(data, content_type)

    # 3) path-like (str/Path)
    return _decode_with_ffmpeg(str(voice))


def _load_audio_any(voice: Any, content_type: str | None) -> tuple[np.ndarray, int]:
    # Backward-compatible alias; prefer load_audio_any().
    return load_audio_any(voice, content_type)


class ASRPipeline:
    def __init__(self, model):
        self.model = model

    def run(self, voice: Any, content_type: str | None = None):
        audio_array, _sr = load_audio_any(voice, content_type)

        audio_array = noise_reduce.reduce_noise(audio_array)
        audio_array = volume_normalize.normalize_volume(audio_array, volume=0.1)

        segments = self.model.transcribe(audio_array)
        text = " ".join([seg.text for seg in segments])

        return {"error": None, "text": text}