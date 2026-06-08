"""ASR model wrapper using faster-whisper.

Auto-detects the audio language and transcribes in that language when
ASR_LANGUAGE is not set (or set to "auto"). Set ASR_LANGUAGE to a BCP-47
code (e.g. "vi", "en") to force a specific language.

Contract: `transcribe()` returns an iterable of objects with a `.text` attribute.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from app.core.config import settings


@dataclass(frozen=True)
class ASRSegment:
    text: str


class ASRModel:
    def __init__(self) -> None:
        print("Loading ASR Model (faster-whisper)...")
        try:
            from faster_whisper import WhisperModel
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "Missing ASR dependency 'faster-whisper'. Install with: pip install faster-whisper"
            ) from e

        model_name = (getattr(settings, "ASR_MODEL_NAME", None) or "").strip()
        # ASR_MODEL_NAME should be a Whisper model size; fall back to "base"
        valid_sizes = {"tiny", "base", "small", "medium", "large", "large-v2", "large-v3"}
        if not model_name or model_name not in valid_sizes:
            model_name = "base"

        device = (getattr(settings, "ASR_DEVICE", None) or "cpu").strip()
        compute_type = (getattr(settings, "ASR_COMPUTE_TYPE", None) or "float32").strip()

        lang_cfg = (getattr(settings, "ASR_LANGUAGE", None) or "").strip().lower()
        # None / empty / "auto" → let Whisper detect the language from audio
        self._language: str | None = None if (not lang_cfg or lang_cfg == "auto") else lang_cfg

        self._model = WhisperModel(model_name, device=device, compute_type=compute_type)
        print(
            f"ASR Model loaded (faster-whisper): model={model_name} device={device} "
            f"language={'auto-detect' if self._language is None else self._language}"
        )

    def transcribe(self, audio: object) -> Iterable[ASRSegment]:
        if audio is None:
            return []

        audio_array = np.asarray(audio)
        if audio_array.ndim != 1:
            audio_array = np.squeeze(audio_array)
        if audio_array.ndim != 1:
            raise ValueError("Audio must be a 1-D mono waveform array")

        audio_array = audio_array.astype(np.float32, copy=False)

        segments, _info = self._model.transcribe(
            audio_array,
            language=self._language,  # None = auto-detect
            task="transcribe",        # always transcribe, never translate
            beam_size=5,
        )

        return [ASRSegment(text=seg.text.strip()) for seg in segments if seg.text.strip()]