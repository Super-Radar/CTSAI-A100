"""Deterministic stress inputs and the official CTSAI-A100 ADC data loader.

All synthetic generators return ``numpy.int64`` sample arrays in LSB units on
the ADC's native int16 scale (full scale = [-32768, 32767]).  Generators are
seeded so every run of the CLI is reproducible.

The ADC loader reproduces the official MATLAB chain of
``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100``:

1. ``init_function/load_adc_data.m`` - the comma separated uint32 dump is
   flattened, truncated / zero-padded to ``rng_nfft/2 * nchirp`` words and
   reshaped (MATLAB column-major) to ``[rng_nfft/2 words, nchirp]``, then
   transposed to ``[nchirp, rng_nfft/2 words]``;
2. ``init_function/adc_mem_2_real_fixed.m`` - every 32-bit word is split into
   a high 16-bit and a low 16-bit **signed int16 real sample** (H first, L
   second), giving ``[nchirp, rng_nfft]`` real samples per RX channel;
3. ``FFT/do_fft_2d.m`` - the range FFT applies a symmetric Hann window of the
   *actual* sample count ``adc_sample_num`` and zero-pads to ``rng_nfft``.

``adc_sample_num`` follows ``init_function/get_radar_paramete.m``:
``round((adc_sample_end - adc_sample_start) * adc_freq / dec_factor)``,
capped at ``rng_nfft``:

* init0 (``cfg/CTASI-A100配置/sensor_config_init0.hxx``):
  round((42.8 - 2) * 25 / 1) = 1020, and 1020 <= rng_nfft=1024, so 1020
  windowed samples plus 4 zeros enter the 1024-point range FFT;
* init1 (``sensor_config_init1.hxx``):
  round((104.4 - 2) * 25 / 1) = 2560 > rng_nfft=2048, so the cap applies
  and all 2048 samples are used with no zero padding.

The MATLAB output normalizations ``fft(...)/rng_fft_scalar`` and
``fft(...)/vel_fft_scalar`` are pure output scalings and are omitted here;
they cancel in every relative error reported by this project.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

DEFAULT_SEED = 20261003
FULLSCALE_POS = 32767
FULLSCALE_NEG = -32768

# init0 profile used by the real-data experiment (single RX, range FFT).
INIT0 = {
    "rng_nfft": 1024,
    "nchirp": 256,
    "adc_sample_num": 1020,   # derived above from get_radar_paramete.m
}

DATA_FILES = [
    "adc_远波_20260512135827_f0_0_13_58_36_748.txt",
    "adc_远波_20260512135827_f0_1_13_58_45_367.txt",
    "adc_远波_20260512135827_f0_2_13_58_53_977.txt",
    "adc_远波_20260512135827_f0_3_13_59_02_596.txt",
    "adc_近波_20260512141857_f1_0_14_19_05_965.txt",
    "adc_近波_20260512141857_f1_1_14_19_14_570.txt",
    "adc_近波_20260512141857_f1_2_14_19_23_170.txt",
    "adc_近波_20260512141857_f1_3_14_19_31_788.txt",
]


# ---------------------------------------------------------------------------
# synthetic stress generators
# ---------------------------------------------------------------------------

def fullscale_multitone(
    n: int = 1024,
    seed: int = DEFAULT_SEED,
    bins: Sequence[int] = (37, 101, 239),
    weights: Sequence[float] = (1.0, 0.8, 0.6),
) -> np.ndarray:
    """Sum of integer-bin cosines with random phases, scaled near full scale."""
    rng = np.random.default_rng(seed)
    acc = np.zeros(n)
    for k, w in zip(bins, weights):
        acc += w * np.cos(2.0 * np.pi * k * np.arange(n) / n + rng.uniform(0.0, 2.0 * np.pi))
    acc *= 0.98 * FULLSCALE_POS / np.max(np.abs(acc))
    return np.rint(acc).astype(np.int64)


def integer_bin_tone(n: int = 1024, k: int = 200, amplitude: float = 12000.0) -> np.ndarray:
    """Single cosine at an integer FFT bin, rounded to integer samples."""
    x = amplitude * np.cos(2.0 * np.pi * k * np.arange(n) / n)
    return np.rint(x).astype(np.int64)


def dc_input(n: int = 1024, amplitude: int = 32000) -> np.ndarray:
    """Constant (DC) input at the given amplitude."""
    return np.full(n, int(amplitude), dtype=np.int64)


def impulse_input(n: int = 1024, amplitude: int = 32000) -> np.ndarray:
    """Single full-scale impulse at n=0."""
    x = np.zeros(n, dtype=np.int64)
    x[0] = int(amplitude)
    return x


def symbol_boundary(n: int = 1024) -> np.ndarray:
    """Alternating pattern touching both int16 code extremes.

    Even samples are +32767 and odd samples are -32768, so both the largest
    positive and the most negative int16 code words are exercised.
    """
    x = np.empty(n, dtype=np.int64)
    x[0::2] = FULLSCALE_POS
    x[1::2] = FULLSCALE_NEG
    return x


def bandlimited_noise(
    n: int = 1024,
    seed: int = DEFAULT_SEED,
    cutoff: float = 0.22,
    level: float = 0.9,
) -> np.ndarray:
    """Gaussian noise low-pass filtered below ``cutoff`` (fraction of Nyquist)."""
    rng = np.random.default_rng(seed)
    spectrum = rng.standard_normal(n)
    spec = np.fft.fft(spectrum)
    keep = max(1, int(round(cutoff * n)))
    spec[keep:-keep] = 0.0
    x = np.real(np.fft.ifft(spec))
    x *= level * FULLSCALE_POS / np.max(np.abs(x))
    return np.rint(x).astype(np.int64)


SYNTHETIC_INPUTS = (
    "fullscale_multitone",
    "integer_bin_tone",
    "dc",
    "impulse",
    "symbol_boundary",
    "bandlimited_noise",
)


def make_synthetic_inputs(n: int = 1024, seed: int = DEFAULT_SEED) -> Dict[str, np.ndarray]:
    """All stress inputs used by the sweeps, in a fixed order."""
    return {
        "fullscale_multitone": fullscale_multitone(n, seed),
        "integer_bin_tone": integer_bin_tone(n),
        "dc": dc_input(n),
        "impulse": impulse_input(n),
        "symbol_boundary": symbol_boundary(n),
        "bandlimited_noise": bandlimited_noise(n, seed),
    }


# ---------------------------------------------------------------------------
# official ADC data chain
# ---------------------------------------------------------------------------

def matlab_hann(n: int) -> np.ndarray:
    """Symmetric Hann window, identical to MATLAB ``hann(n)`` / ``numpy.hanning``."""
    if n == 1:
        return np.ones(1)
    w = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))
    return w


def load_adc_words(path) -> np.ndarray:
    """Load one ADC dump file: a single comma-separated line of uint32 words."""
    return np.loadtxt(path, delimiter=",", dtype=np.int64).ravel()


def adc_words_to_samples(
    words: np.ndarray,
    rng_nfft: int = INIT0["rng_nfft"],
    nchirp: int = INIT0["nchirp"],
) -> np.ndarray:
    """Official uint32-word to signed int16 sample conversion.

    Returns an ``[nchirp, rng_nfft]`` int64 array of real samples; row c holds
    the rng_nfft real samples of chirp c (H16 first, L16 second per word).
    """
    expected = (rng_nfft // 2) * nchirp
    if words.size < expected:
        raise ValueError(f"ADC dump too short: {words.size} words < {expected}")
    words = words[:expected]  # official chain truncates the extra trailing word
    # MATLAB reshape(col, [rng_nfft/2, nchirp]) is column-major; its transpose
    # equals this C-order reshape to [nchirp, rng_nfft/2].
    word_mat = words.reshape((nchirp, rng_nfft // 2))

    high = np.right_shift(word_mat.astype(np.int64), 16)          # floor(w / 2^16)
    low = np.bitwise_and(word_mat, 0xFFFF)                        # mod(w, 2^16)

    def to_signed16(v: np.ndarray) -> np.ndarray:
        return np.where(v >= 0x8000, v - 0x10000, v)

    samples = np.empty((nchirp, rng_nfft), dtype=np.int64)
    samples[:, 0::2] = to_signed16(high)   # first sample of each word pair: H16
    samples[:, 1::2] = to_signed16(low)    # second sample: L16
    return samples


def official_range_fft_input(
    chirp_samples: np.ndarray,
    adc_sample_num: int = INIT0["adc_sample_num"],
    rng_nfft: int = INIT0["rng_nfft"],
) -> np.ndarray:
    """One chirp's range-FFT input exactly as the official chain feeds it.

    Applies the symmetric Hann window of ``adc_sample_num`` taps to the first
    ``adc_sample_num`` samples, zero-pads to ``rng_nfft`` and rounds to the
    int16 LSB grid.  The SAME integer vector is fed to the float64 reference
    and to the fixed-point core, so the reported error isolates the FFT
    arithmetic wordlength (window coefficient quantization is out of scope).
    """
    win = matlab_hann(adc_sample_num)
    x = np.asarray(chirp_samples, dtype=np.float64).ravel()
    if x.size < adc_sample_num:
        raise ValueError("chirp shorter than adc_sample_num")
    windowed = x[:adc_sample_num] * win
    out = np.zeros(rng_nfft, dtype=np.int64)
    out[:adc_sample_num] = np.rint(windowed).astype(np.int64)
    return out


def load_official_range_inputs(
    path,
    rng_nfft: int = INIT0["rng_nfft"],
    nchirp: int = INIT0["nchirp"],
    adc_sample_num: int = INIT0["adc_sample_num"],
) -> np.ndarray:
    """Load an ADC dump and return the ``[nchirp, rng_nfft]`` range-FFT inputs."""
    samples = adc_words_to_samples(load_adc_words(path), rng_nfft, nchirp)
    return np.stack([official_range_fft_input(samples[c], adc_sample_num, rng_nfft)
                     for c in range(nchirp)])
