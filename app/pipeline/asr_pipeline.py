import numpy as np
import librosa

from services.extractor.voice import noise_reduce, volume_normalize


class ASRPipeline:

    def __init__(self, model):
        self.model = model

    def run(self, voice):

        audio_array, sr = librosa.load(voice, sr=16000, mono=True)

        audio_array = noise_reduce.reduce_noise(audio_array)

        audio_array = volume_normalize.normalize_volume(audio_array, volume=0.1)

        segments = self.model.transcribe(
            audio_array,
        )

        text = " ".join([seg.text for seg in segments])

        return {
            "error": None,
            "text": text
        }

