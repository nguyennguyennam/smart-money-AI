'''
    Apply pedalboard to enhance the audio quality of a given audio signal.
'''

from pedalboard import Pedalboard, HighpassFilter, Compressor, LowShelfFilter, Gain, NoiseGate
import numpy as np

def enhance_audio_quality (audio : np.ndarray) -> np.ndarray:
    board =  Pedalboard([
        HighpassFilter(cutoff_frequency_hz=80),
        NoiseGate(threshold_db=-40, ratio=1.5, release_ms=250),
        Compressor(threshold_db=-20, ratio=2.0, attack_ms=5, release_ms=50),
        Gain(gain_db=5)
    ])

    enhanced_audio = board(audio, sample_rate=16000)
    return enhanced_audio