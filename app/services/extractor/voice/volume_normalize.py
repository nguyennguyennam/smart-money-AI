import numpy as np

'''
    Normalize the volume of an audio signal to a target level.
'''

def normalize_volume (audio: np.ndarray, volume: float) -> np.ndarray:
    # Calculate the current volume of the audio signal
    current_volume = np.sqrt(np.mean(audio**2))

    # Calculate the normalization factor
    if current_volume > 0:
        normalization_factor = volume / current_volume
    else:
        normalization_factor = 1.0

    # Apply the normalization factor to the audio signal
    normalized_audio = audio * normalization_factor

    return normalized_audio