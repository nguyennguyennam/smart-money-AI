from __future__ import annotations

import numpy as np

try:
    import noisereduce as nr
except Exception:  # optional dependency
    nr = None

'''
    Apply noise reduction to an image using the Non-Local Means Denoising algorithm.
'''

def reduce_noise (original_audio: np.ndarray) -> np.ndarray:
    if nr is None:
        return original_audio

    reduced_noise_audio_nonSt = nr.reduce_noise(
        y=original_audio,
        sr=16000,
        prop_decrease=1.0,
        stationary=True,
    )
    return reduced_noise_audio_nonSt