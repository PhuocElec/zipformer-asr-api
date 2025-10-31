import numpy as np
from numpy.typing import NDArray
from scipy import signal


def resample_audio(audio: NDArray, orig_sr: int, target_sr: int) -> NDArray:
    """
    Resamples the given audio signal from the original sample rate to the target sample rate.

    **Parameters:**
    - `audio` (NDArray): Input audio signal as a NumPy array.
    - `orig_sr` (int): Original sample rate of the audio.
    - `target_sr` (int): Target sample rate to resample the audio.

    **Returns:**
    - `NDArray`: The resampled audio signal.
    """
    if orig_sr == target_sr:
        return audio

    gcd = np.gcd(orig_sr, target_sr)

    up = target_sr // gcd
    down = orig_sr // gcd

    resample_audio = signal.resample_poly(audio, up, down)

    return resample_audio

def calculate_dbfs(audio: NDArray[np.float32], eps=1e-10) -> float:
    """
    Calculates the decibels relative to full scale (dBFS) of an audio signal.

    **Parameters:**
    - `audio` (NDArray[np.float32]): Input audio signal as a NumPy array.
    - `eps` (float): A small value to avoid log of zero errors (default: 1e-10).

    **Returns:**
    - `float`: The dBFS value of the audio signal.
    """
    rms = np.sqrt(np.mean(np.square(audio)))
    rms = max(rms, eps)
    dbfs = 20 * np.log10(rms)
    return dbfs