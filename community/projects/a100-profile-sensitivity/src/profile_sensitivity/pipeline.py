"""NumPy port of the official A100 MATLAB ADC processing chain.

Parity map (public ``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100``):

* ``init_function/adc_mem_2_real_fixed.m`` -> :func:`unpack_words`
  (uint32 word = high int16 first, low int16 second);
* ``init_function/load_adc_data.m``        -> :func:`load_cube`
  (truncate/pad to ``rng_nfft/2 * nchirp`` words, reshape column-major as
  ``[rng_nfft/2, nchirp]`` i.e. sample-fastest/chirp-slowest, then return the
  sample-major ``[rng_nfft, nchirp]`` cube that ``adcData`` holds in MATLAB);
* ``FFT/do_fft_2d.m``                      -> :func:`rv_image`
  (hann(adc_sample_num) over fast time, Range-FFT of rng_nfft, keep the first
  rng_nfft/2 bins, hann over chirps, Doppler-FFT of vel_nfft).

The MATLAB scripts additionally divide by ``rng_fft_scalar``/``vel_fft_scalar``
(0xff / 0x3f); that is a constant factor that only shifts the dB floor, so it
is omitted here and noted for honesty.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .profiles import Profile


# --------------------------------------------------------------------- load
def read_raw_words(path: Path) -> np.ndarray:
    """Read one channel capture: single-line comma-separated uint32 values."""
    with open(path, "r", encoding="ascii") as fh:
        text = fh.read()
    words = np.array(text.split(","), dtype=np.int64)
    return words.astype(np.uint32)


def unpack_words(words: np.ndarray) -> np.ndarray:
    """uint32 -> two consecutive signed int16 samples (high word first).

    Verbatim port of ``adc_mem_2_real_fixed.m``: sample 2k-1 comes from the
    high 16 bits, sample 2k from the low 16 bits, both as two's complement.
    """
    words = np.asarray(words, dtype=np.uint32)
    hi = (words >> 16).astype(np.int16)
    lo = (words & 0xFFFF).astype(np.int16)
    out = np.empty(2 * words.size, dtype=np.int16)
    out[0::2] = hi
    out[1::2] = lo
    return out


def samples_to_words(samples: np.ndarray) -> np.ndarray:
    """Inverse of :func:`unpack_words` (used to build synthetic captures)."""
    s = np.asarray(samples)
    assert s.ndim == 1 and s.size % 2 == 0, "need a flat, even-length stream"
    s16 = np.clip(np.round(s), -32768, 32767).astype(np.int16)
    hi = s16[0::2].astype(np.uint16).astype(np.uint32)
    lo = s16[1::2].astype(np.uint16).astype(np.uint32)
    return (hi << np.uint32(16)) | lo


def load_cube(raw_words: np.ndarray, profile: Profile) -> np.ndarray:
    """Truncate/pad + reshape + unpack into the ``[rng_nfft, nchirp]`` cube.

    Mirrors ``load_adc_data.m``: the flat word stream is reshaped to
    ``[rng_nfft/2, nchirp]`` in *column-major* order (fast-time index runs
    fastest, chirp index slowest), each word unpacks to two consecutive
    samples, and the result is transposed to MATLAB's ``adcData`` layout
    ``[sample, chirp]``.
    """
    expected = profile.expected_payload_words
    raw = np.asarray(raw_words, dtype=np.uint32)
    if raw.size < expected:
        raw = np.concatenate([raw, np.zeros(expected - raw.size, dtype=np.uint32)])
    elif raw.size > expected:
        raw = raw[:expected]

    packed = raw.reshape(profile.nchirp, profile.packed_words_per_chirp)
    hi = (packed >> 16).astype(np.int16)
    lo = (packed & 0xFFFF).astype(np.int16)
    cube = np.empty((profile.nchirp, profile.rng_nfft), dtype=np.float64)
    cube[:, 0::2] = hi
    cube[:, 1::2] = lo
    return cube.T  # -> [sample, chirp]


def load_file(path: Path, profile: Profile) -> np.ndarray:
    return load_cube(read_raw_words(path), profile)


# ----------------------------------------------------------------- FFT chain
def rv_image(cube: np.ndarray, profile: Profile) -> np.ndarray:
    """Windowed 2D-FFT -> RV image in dB, shape [rng_bin_num, vel_nfft]."""
    win_r = np.hanning(profile.adc_sample_num)          # rng_win == "hanning"
    x = cube[: profile.adc_sample_num, :] * win_r[:, None]
    rng_fft = np.fft.fft(x, n=profile.rng_nfft, axis=0)[: profile.rng_bin_num, :]

    win_v = np.hanning(profile.nchirp)                  # vel_win == "hanning"
    rng_fft *= win_v[None, :]

    vel_fft = np.fft.fft(rng_fft, n=profile.vel_nfft, axis=1)
    return 20.0 * np.log10(np.abs(vel_fft) + 1e-20)


def range_axis_m(profile: Profile) -> np.ndarray:
    return np.arange(profile.rng_bin_num) * profile.rng_res_1dfft_m


def doppler_bins_signed(profile: Profile) -> np.ndarray:
    """Doppler bin index in the standard (-N/2, ..., N/2-1] fftshift order."""
    sb = np.arange(profile.vel_nfft)
    sb[sb >= profile.vel_nfft // 2] -= profile.vel_nfft
    return np.fft.fftshift(sb)


def velocity_axis_mps(profile: Profile) -> np.ndarray:
    return doppler_bins_signed(profile) * profile.v_acu_mps


# ------------------------------------------------------------- peak picking
def dominant_peak(img_db: np.ndarray,
                  profile: Profile,
                  skip_doppler_edge: int = 2,
                  skip_range_bins: int = 2) -> dict:
    """Strongest cell away from DC-leakage and the Doppler band edges.

    Returns dict with (integer) bins, the signed Doppler index, and the
    apparent physical coordinates under ``profile``'s axis calibration.
    """
    kr_lo = skip_range_bins
    n = profile.vel_nfft
    sb = np.arange(n)                 # signed bins in FFT column order
    sb[n // 2:] -= n
    keep = np.abs(sb) <= n // 2 - skip_doppler_edge
    col_idx = np.nonzero(keep)[0]
    sub = img_db[kr_lo:, :][:, col_idx]
    r, c = np.unravel_index(np.argmax(sub), sub.shape)
    kr = int(r + kr_lo)
    kv_signed = int(sb[col_idx[c]])
    return {
        "k_range": kr,
        "k_doppler": kv_signed,
        "range_m": kr * profile.rng_res_1dfft_m,
        "vel_mps": kv_signed * profile.v_acu_mps,
        "power_db": float(img_db[kr, col_idx[c]]),
    }


def range_energy_stats(cube: np.ndarray, profile: Profile) -> dict:
    """Per-hypothesis range-profile descriptors used by the detector report."""
    win_r = np.hanning(profile.adc_sample_num)
    x = cube[: profile.adc_sample_num, :] * win_r[:, None]
    rng_fft = np.fft.fft(x, n=profile.rng_nfft, axis=0)[: profile.rng_bin_num, :]
    prof = np.abs(rng_fft).mean(axis=1)
    energy = prof ** 2
    total = energy.sum()
    p = energy / total
    entropy = float(-(p * np.log(p + 1e-30)).sum())
    return {
        "range_entropy": entropy,
        "frac_energy_first50bins": float(energy[:50].sum() / total),
    }
