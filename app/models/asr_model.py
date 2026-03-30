from __future__ import annotations

import os
import tempfile
from functools import lru_cache

from app.core.config import settings


class ASRModel:
	def __init__(self):
		# faster-whisper model sizes: tiny/base/small/medium/large-v3, or local path.
		# This repo historically uses "whisper" as a placeholder; map it to a valid default.
		model_name = (settings.ASR_MODEL_NAME or "").strip()
		if model_name.lower() == "whisper":
			model_name = "base"

		try:
			# Import lazily so environments without working ctranslate2 can still run OCR jobs.
			from faster_whisper import WhisperModel  # type: ignore

			self._model = WhisperModel(
				model_name,
				device=settings.ASR_DEVICE,
				compute_type=settings.ASR_COMPUTE_TYPE,
			)
		except Exception as e:
			raise RuntimeError(
				"ASR backend failed to initialize (faster-whisper/ctranslate2). "
				"On Windows this is commonly caused by missing VC++ runtime or an incompatible CPU instruction set. "
				"OCR jobs can still run; voice jobs will fail until ASR is fixed. "
				f"Original error: {e}"
			) from e

	def transcribe_file(self, file_path: str) -> str:
		segments, _info = self._model.transcribe(file_path)
		parts: list[str] = []
		for seg in segments:
			text = (seg.text or "").strip()
			if text:
				parts.append(text)
		return " ".join(parts).strip()

	def transcribe_bytes(self, audio_bytes: bytes, suffix: str = ".audio") -> str:
		if not audio_bytes:
			return ""

		tmp_path = ""
		try:
			with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
				f.write(audio_bytes)
				tmp_path = f.name
			return self.transcribe_file(tmp_path)
		finally:
			if tmp_path:
				try:
					os.remove(tmp_path)
				except OSError:
					pass


@lru_cache(maxsize=1)
def get_asr_model() -> ASRModel:
	return ASRModel()

