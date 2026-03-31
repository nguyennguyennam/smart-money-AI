"""ASR model wrapper.

This project previously used `faster-whisper` directly. To use PhoWhisper
(Vietnamese fine-tuned Whisper), we load it from HuggingFace using
`transformers`.

Contract: `transcribe()` returns an iterable of objects with a `.text` attribute
so existing pipeline code can join segment texts.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np

from app.core.config import settings


@dataclass(frozen=True)
class ASRSegment:
   text: str


class ASRModel:
   def __init__(self) -> None:
      print("Loading ASR Model (PhoWhisper)...")

      try:
         import torch
         from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
      except Exception as e:  # pragma: no cover
         raise ImportError(
            "Missing ASR dependencies. Install with: pip install -r app/libs.txt "
            "(requires 'transformers' and 'torch')."
         ) from e

      model_id = (settings.ASR_MODEL_NAME or "vinai/PhoWhisper-small").strip()

      want_device = (settings.ASR_DEVICE or "cpu").strip().lower()
      if want_device == "cuda" and torch.cuda.is_available():
         self.device = torch.device("cuda")
      else:
         self.device = torch.device("cpu")

      compute_type = (settings.ASR_COMPUTE_TYPE or "float32").strip().lower()
      if self.device.type == "cuda" and compute_type in {"float16", "fp16"}:
         torch_dtype = torch.float16
      else:
         torch_dtype = torch.float32

      self._input_dtype = torch_dtype if self.device.type == "cuda" else torch.float32

      self.processor = AutoProcessor.from_pretrained(model_id)
      self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
         model_id,
         torch_dtype=torch_dtype,
      )
      self.model.to(self.device)
      self.model.eval()

      self._forced_decoder_ids: Optional[list[list[int]]] = None
      try:
         # Whisper-style processor supports this; keep best-effort.
         self._forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="vi", task="transcribe"
         )
      except Exception:
         self._forced_decoder_ids = None

      print(f"ASR Model loaded: {model_id} on {self.device.type}")

   def transcribe(self, audio: object) -> Iterable[ASRSegment]:
      import torch

      if audio is None:
         return []

      audio_array = np.asarray(audio)
      if audio_array.ndim != 1:
         audio_array = np.squeeze(audio_array)
      if audio_array.ndim != 1:
         raise ValueError("Audio must be a 1-D mono waveform array")

      audio_array = audio_array.astype(np.float32, copy=False)

      inputs = self.processor(audio_array, sampling_rate=16000, return_tensors="pt")
      inputs = {
         k: (v.to(self.device, dtype=self._input_dtype) if torch.is_floating_point(v) else v.to(self.device))
         for k, v in inputs.items()
      }

      gen_kwargs = {}
      if self._forced_decoder_ids is not None:
         gen_kwargs["forced_decoder_ids"] = self._forced_decoder_ids

      with torch.inference_mode():
         predicted_ids = self.model.generate(**inputs, **gen_kwargs)

      text = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
      if not text:
         return []
      return [ASRSegment(text=text)]

   