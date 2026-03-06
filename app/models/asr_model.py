'''
   Define asr model using Faster-whisper
'''

from faster_whisper import WhisperModel

class ASRModel:
   model_size = "small"

   def __init__ (self):
      print("Loading ASR Model...")

      self.model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
      print("ASR Model loaded")

   def transcribe (self, audio: object) -> str:
      segments, result = self.model.transcribe (audio = audio, beam_size=5, task= "transcribe",best_of=5, language="vi", word_timestamps=True, vad_filter=True)
      
      return segments

   



