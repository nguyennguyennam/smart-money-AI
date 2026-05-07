"""ASR model wrapper.

This project uses the official `moonshine-voice` PyPI package (as demonstrated
in `getting_started_with_moonshine_voice.ipynb`). This avoids relying on
Transformers' model mappings, which may not yet include Moonshine streaming
types in older versions.

Contract: `transcribe()` returns an iterable of objects with a `.text` attribute
so existing pipeline code can join segment texts.
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
      print("Loading ASR Model (Moonshine Voice)...")

      try:
         import moonshine_voice
      except Exception as e:  # pragma: no cover
         raise ImportError(
            "Missing ASR dependency 'moonshine-voice'. Install with: pip install -r app/libs.txt"
         ) from e

      # Primary control is via ASR_LANGUAGE (e.g., 'vi', 'en').
      # For backwards compatibility, if ASR_MODEL_NAME looks like a language code,
      # we accept it as an override.
      lang = (getattr(settings, "ASR_LANGUAGE", None) or "").strip()
      model_name = (getattr(settings, "ASR_MODEL_NAME", None) or "").strip()
      if (not lang) and model_name and ("/" not in model_name) and ("\\" not in model_name) and (len(model_name) <= 10):
         lang = model_name
      if not lang:
         lang = "en"

      self._moonshine_voice = moonshine_voice
      self._language = lang
      self._model_path, self._model_arch = moonshine_voice.get_model_for_language(lang)
      self._transcriber = moonshine_voice.Transcriber(
         model_path=self._model_path,
         model_arch=self._model_arch,
      )

      print(f"ASR Model loaded (moonshine-voice): lang={lang}")

   def transcribe(self, audio: object) -> Iterable[ASRSegment]:
      if audio is None:
         return []

      audio_array = np.asarray(audio)
      if audio_array.ndim != 1:
         audio_array = np.squeeze(audio_array)
      if audio_array.ndim != 1:
         raise ValueError("Audio must be a 1-D mono waveform array")

      audio_array = audio_array.astype(np.float32, copy=False)

      transcript = self._transcriber.transcribe_without_streaming(audio_array)

      lines = getattr(transcript, "lines", None)
      if isinstance(lines, list):
         segments: list[ASRSegment] = []
         for line in lines:
            text = str(getattr(line, "text", "") or "").strip()
            if text:
               segments.append(ASRSegment(text=text))
         return segments

      text = str(getattr(transcript, "text", transcript) or "").strip()
      if not text:
         return []
      return [ASRSegment(text=text)]

   
