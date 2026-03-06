import noisereduce as nr
import numpy as np

'''
    Apply noise reduction to an image using the Non-Local Means Denoising algorithm.
'''

def reduce_noise (original_audio: np.ndarray) -> np.ndarray:
    reduced_noise_audio_nonSt = nr.reduce_noise(
        y = original_audio,
        sr = 22000,
        prop_decrease=1.0,
        stationary=True
    )
    return reduced_noise_audio_nonSt