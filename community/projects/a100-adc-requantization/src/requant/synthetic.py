"""Synthetic int16 benchmarks with known ground truth.

Three families, all rendered onto the int16 grid (the 16-bit baseline is the
rounded int16 signal, so requantisation is exactly a bit-reduction, as with
real ADC data):

* single tones across amplitudes from near full scale down to 1 LSB
  (amplitudes 12000, 3000, 750, 188, 47, 12, 3, 1), range bin 137,
  Doppler bin 3;
* a strong + weak two-tone pair (12000 @ bin 137 / Doppler 3,
  38 @ bin 357 / Doppler 20, i.e. about -50 dB below the strong tone)
  on a small measurement-noise floor (sigma = 2 LSB) -- the weak tone
  sinks below the requantisation noise as k drops;
* band-limited noise (bins 100..300 of the fast-time FFT, RMS 2000 LSB).
"""

from __future__ import annotations

import numpy as np

from .pipeline import INT16_MAX, INT16_MIN

TONE_RNG_BIN = 137
TONE_VEL_BIN = 3
WEAK_RNG_BIN = 357
WEAK_VEL_BIN = 20
TONE_AMPLITUDES = (12000, 3000, 750, 188, 47, 12, 3, 1)
TWO_TONE_STRONG = 12000
TWO_TONE_WEAK = 38
TWO_TONE_NOISE_SIGMA = 2.0
NOISE_RMS = 2000.0
NOISE_BAND = (100, 300)


def _to_int16(x: np.ndarray) -> np.ndarray:
    """Round onto the int16 storage grid (the synthetic 16-bit baseline)."""
    return np.clip(np.rint(x), INT16_MIN, INT16_MAX).astype(np.int32)


def single_tone(rng_nfft: int, nchirp: int, amplitude: float,
                rng_bin: int = TONE_RNG_BIN, vel_bin: int = TONE_VEL_BIN
                ) -> np.ndarray:
    """Cosine tone at an exact range/Doppler bin, [rng_nfft, nchirp] int16."""
    n = np.arange(rng_nfft)[:, None]
    c = np.arange(nchirp)[None, :]
    x = amplitude * np.cos(2 * np.pi * rng_bin * n / rng_nfft
                           + 2 * np.pi * vel_bin * c / nchirp)
    return _to_int16(x)


def two_tone(rng_nfft: int, nchirp: int, seed: int,
             strong: float = TWO_TONE_STRONG, weak: float = TWO_TONE_WEAK,
             noise_sigma: float = TWO_TONE_NOISE_SIGMA) -> np.ndarray:
    n = np.arange(rng_nfft)[:, None]
    c = np.arange(nchirp)[None, :]
    rng = np.random.default_rng(seed)
    x = strong * np.cos(2 * np.pi * TONE_RNG_BIN * n / rng_nfft
                        + 2 * np.pi * TONE_VEL_BIN * c / nchirp)
    x = x + weak * np.cos(2 * np.pi * WEAK_RNG_BIN * n / rng_nfft
                          + 2 * np.pi * WEAK_VEL_BIN * c / nchirp)
    x = x + rng.normal(0.0, noise_sigma, size=x.shape)
    return _to_int16(x)


def bandlimited_noise(rng_nfft: int, nchirp: int, seed: int,
                      rms: float = NOISE_RMS,
                      band: tuple[int, int] = NOISE_BAND) -> np.ndarray:
    rng = np.random.default_rng(seed)
    white = rng.normal(0.0, 1.0, size=(rng_nfft, nchirp))
    spec = np.fft.fft(white, axis=0)
    keep = np.zeros(rng_nfft, dtype=bool)
    keep[band[0]:band[1]] = True
    spec[~keep, :] = 0.0
    x = np.real(np.fft.ifft(spec, axis=0))
    x = x / np.sqrt(np.mean(x ** 2)) * rms
    return _to_int16(x)


def synthetic_cases(seed: int) -> dict[str, np.ndarray]:
    """All synthetic frames, keyed by case name (see module docstring)."""
    nsm, nchirp = 1024, 128
    cases: dict[str, np.ndarray] = {}
    for a in TONE_AMPLITUDES:
        cases[f"tone_A{a}"] = single_tone(nsm, nchirp, a)
    cases["tone_weak_alone"] = single_tone(nsm, nchirp, TWO_TONE_WEAK,
                                           rng_bin=WEAK_RNG_BIN,
                                           vel_bin=WEAK_VEL_BIN)
    cases["two_tone"] = two_tone(nsm, nchirp, seed)
    cases["bandlimited_noise"] = bandlimited_noise(nsm, nchirp, seed)
    return cases
