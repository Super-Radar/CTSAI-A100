"""ADC unpacking and official-equivalent FFT chain (Python port).

This module reproduces, in NumPy, the reference MATLAB chain shipped with the
CTSAI-A100 repository:

    ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/
        init_function/load_adc_data.m        -- read txt, truncate/pad to
                                                [rng_nfft/2, nchirp], reshape
                                                and transpose
        init_function/adc_mem_2_real_fixed.m -- each uint32 word splits into
                                                two *signed int16 real* samples
                                                (H16 first, L16 second)
        init_function/get_radar_paramete.m  -- adc_sample_num derivation
        FFT/do_fft_2d.m                      -- hann window (actual sample
                                                count) -> Range FFT -> first
                                                half -> slow-time hann ->
                                                Doppler FFT

Key points kept identical to the MATLAB reference:

*   The txt files hold unsigned 32-bit decimal words, one line, comma
    separated, no header.  Each file carries one extra trailing value (0)
    beyond ``rng_nfft/2 * nchirp``; ``align_length`` reports it explicitly
    instead of silently truncating.
*   Column-major reshape in MATLAB (``reshape(col, [rng_nfft/2, nchirp])``)
    followed by transpose equals a C-order reshape to
    ``(nchirp, rng_nfft/2)`` in NumPy.
*   ``adc_mem_2_real_fixed`` produces *real* int16 samples (not complex I/Q):
    sample 1 of a pair is the high 16 bits (two's complement), sample 2 the
    low 16 bits.
*   Windows are MATLAB ``hann`` (periodic flavour), i.e.
    ``scipy.signal.windows.hann(n, sym=False)``.
*   Range FFT divides by ``rng_fft_scalar`` (0xff = 255); Doppler FFT divides
    by ``vel_fft_scalar`` (0x3f = 63).

``adc_sample_num`` derivation, transcribed from ``get_radar_paramete.m``::

    if dec_factor > 1:
        adc_sample_num = round((adc_sample_end - adc_sample_start)
                               * adc_freq / dec_factor) + 1
    else:
        adc_sample_num = round((adc_sample_end - adc_sample_start)
                               * adc_freq / dec_factor)
    adc_sample_num = min(adc_sample_num, rng_nfft)

With the shipped configs this gives:
    init0: round((42.8 us - 2 us) * 25 MHz) = round(1020.0) = 1020 samples
           (rng_nfft = 1024, so the last 4 FFT input rows stay zero);
    init1: round((104.4 us - 2 us) * 25 MHz) = round(2560.0) = 2560, capped to
           rng_nfft = 2048 (full-length windowing).
"""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import numpy as np
from scipy.signal.windows import hann


@dataclasses.dataclass(frozen=True)
class SensorConfig:
    """Subset of the MATLAB ``SensorConfig`` relevant to the FFT chain."""

    name: str
    rng_nfft: int
    vel_nfft: int
    nchirp: int
    adc_freq_mhz: float
    adc_sample_start_us: float
    adc_sample_end_us: float
    dec_factor: int = 1
    rng_win: str = "hann"          # platform cfg value "hanning"
    vel_win: str = "hann"
    rng_fft_scalar: int = 0xFF     # 255
    vel_fft_scalar: int = 0x3F     # 63
    fmcw_bandwidth_mhz: float = 0.0    # chirp slope context (0 -> unknown)
    fmcw_chirp_rampup_us: float = 0.0

    @property
    def chirp_slope_hz_per_s(self) -> float:
        """RF slope S = bandwidth / ramp-up time (Hz/s)."""
        if self.fmcw_bandwidth_mhz <= 0 or self.fmcw_chirp_rampup_us <= 0:
            raise ValueError("chirp slope context not configured")
        return (self.fmcw_bandwidth_mhz * 1e6) / (self.fmcw_chirp_rampup_us * 1e-6)

    def bin_range_meters(self, bin_index: int) -> float:
        """Approximate equivalent range of an FFT bin (c*f_b/(2*S))."""
        c = 299792458.0
        return (c * self.adc_freq_mhz * 1e6
                / (2.0 * self.chirp_slope_hz_per_s * self.rng_nfft)) * bin_index

    @property
    def adc_sample_num(self) -> int:
        """Number of windowed real samples per chirp (see module docstring)."""
        span = (self.adc_sample_end_us - self.adc_sample_start_us) * self.adc_freq_mhz
        if self.dec_factor > 1:
            n = int(round(span / self.dec_factor)) + 1
        else:
            n = int(round(span / self.dec_factor))
        return min(n, self.rng_nfft)

    @property
    def expected_word_count(self) -> int:
        return (self.rng_nfft // 2) * self.nchirp


#: Shipped sensor configs (cfg/CTASI-A100配置/sensor_config_init{0,1}..hxx).
SENSOR_CONFIGS: Dict[str, SensorConfig] = {
    "init0": SensorConfig(
        name="init0", rng_nfft=1024, vel_nfft=256, nchirp=256,
        adc_freq_mhz=25, adc_sample_start_us=2, adc_sample_end_us=42.8,
        fmcw_bandwidth_mhz=300, fmcw_chirp_rampup_us=43,
    ),
    "init1": SensorConfig(
        name="init1", rng_nfft=2048, vel_nfft=128, nchirp=128,
        adc_freq_mhz=25, adc_sample_start_us=2, adc_sample_end_us=104.4,
        fmcw_bandwidth_mhz=750, fmcw_chirp_rampup_us=104.65,
    ),
}


def load_adc_txt(path) -> np.ndarray:
    """Read one ADC capture txt (unsigned decimal words, comma separated)."""
    values = np.loadtxt(path, delimiter=",", dtype=np.int64)
    if values.ndim != 1:
        values = values.ravel()
    if values.size == 0 or values.min() < 0 or values.max() >= 2 ** 32:
        raise ValueError(f"{path}: expected unsigned 32-bit word values")
    return values


def align_length(values: np.ndarray, expected_len: int) -> Tuple[np.ndarray, dict]:
    """Truncate/pad to ``expected_len`` like ``load_adc_data.m``.

    Unlike the MATLAB code, every removed or padded element is reported
    explicitly (count + first extra value) so the known trailing glitch is
    visible instead of silent.
    """
    actual = int(values.size)
    report = {
        "actual_len": actual,
        "expected_len": int(expected_len),
        "extra_count": max(0, actual - expected_len),
        "first_extra_value": int(values[expected_len]) if actual > expected_len else None,
        "pad_count": max(0, expected_len - actual),
    }
    if actual > expected_len:
        values = values[:expected_len]
    elif actual < expected_len:
        values = np.pad(values, (0, expected_len - actual))
    return values, report


def unpack_adc(words: np.ndarray, rng_nfft: int, nchirp: int) -> np.ndarray:
    """uint32 words -> real int16 samples, shape ``[rng_nfft, nchirp]``.

    Equivalent to MATLAB ``reshape -> transpose -> adc_mem_2_real_fixed ->
    permute([2 1])``: per chirp, word j contributes its signed high 16 bits
    first, then its signed low 16 bits.
    """
    words = np.asarray(words, dtype=np.uint32)
    if words.size != (rng_nfft // 2) * nchirp:
        raise ValueError(
            f"expected {(rng_nfft // 2) * nchirp} words, got {words.size}"
        )
    per_chirp = words.reshape(nchirp, rng_nfft // 2)  # == MATLAB reshape+transpose
    high = (per_chirp >> 16).astype(np.int16).astype(np.float64)
    low = (per_chirp & 0xFFFF).astype(np.int16).astype(np.float64)
    samples = np.empty((nchirp, rng_nfft), dtype=np.float64)
    samples[:, 0::2] = high
    samples[:, 1::2] = low
    return samples.T  # [rng_nfft samples, nchirp]


def repack_adc(samples: np.ndarray) -> np.ndarray:
    """Inverse of :func:`unpack_adc` (round-trip helper for tests)."""
    per_chirp = np.asarray(samples, dtype=np.float64).T  # [nchirp, rng_nfft]
    high = np.clip(per_chirp[:, 0::2], -32768, 32767).astype(np.int64) & 0xFFFF
    low = np.clip(per_chirp[:, 1::2], -32768, 32767).astype(np.int64) & 0xFFFF
    words = (high << 16) | low
    return words.astype(np.uint32).ravel()


def hann_periodic(n: int) -> np.ndarray:
    """MATLAB ``hann(n)`` (periodic), i.e. ``hann(n, sym=False)`` in scipy."""
    return hann(int(n), sym=False)


def range_fft(samples: np.ndarray, cfg: SensorConfig) -> np.ndarray:
    """Official Range FFT: hann(adc_sample_num) + zero pad + FFT/scalar.

    Parameters
    ----------
    samples : ``[rng_nfft, nchirp]`` real matrix from :func:`unpack_adc`.

    Returns
    -------
    Complex matrix ``[rng_nfft, nchirp]``; rows are range bins (the platform
    uses rows ``0 .. rng_nfft/2-1``), columns are chirps.  The slow-time
    sequence of range bin ``b`` is ``R[b, :]``.
    """
    if samples.shape != (cfg.rng_nfft, cfg.nchirp):
        raise ValueError(f"expected {(cfg.rng_nfft, cfg.nchirp)}, got {samples.shape}")
    win = hann_periodic(cfg.adc_sample_num)
    x = np.zeros((cfg.rng_nfft, cfg.nchirp), dtype=np.float64)
    x[: cfg.adc_sample_num, :] = samples[: cfg.adc_sample_num, :] * win[:, None]
    return np.fft.fft(x, n=cfg.rng_nfft, axis=0) / float(cfg.rng_fft_scalar)


def doppler_fft(bin_seq: np.ndarray, cfg: SensorConfig) -> np.ndarray:
    """Official Doppler FFT of one bin's slow-time sequence.

    Slow-time ``hann(nchirp)`` window, FFT padded to ``vel_nfft`` bins,
    divided by ``vel_fft_scalar`` — matching ``do_fft_2d.m``'s velocity
    branch (``nvirtual_chirp == 1`` for both shipped profiles).
    """
    seq = np.asarray(bin_seq)
    if seq.shape[-1] != cfg.nchirp:
        raise ValueError(f"expected {cfg.nchirp} chirps, got {seq.shape[-1]}")
    win = hann_periodic(cfg.nchirp)
    return np.fft.fft(seq * win, n=cfg.vel_nfft, axis=-1) / float(cfg.vel_fft_scalar)


def load_range_matrix(path, cfg: SensorConfig) -> Tuple[np.ndarray, dict]:
    """txt capture -> complex range-bin x chirp matrix, plus length report."""
    raw = load_adc_txt(path)
    aligned, report = align_length(raw, cfg.expected_word_count)
    samples = unpack_adc(aligned, cfg.rng_nfft, cfg.nchirp)
    return range_fft(samples, cfg), report


def dominant_bin(range_matrix: np.ndarray, exclude_dc: bool = True) -> int:
    """Range bin with the largest chirp-averaged power (bin 0 excluded).

    Searches only the physically meaningful first half
    (``0 .. rng_nfft/2-1``) of the Range FFT output, as the platform does
    when it builds its range-Doppler image.
    """
    half = range_matrix.shape[0] // 2
    start = 1 if exclude_dc else 0
    power = np.mean(np.abs(range_matrix[start:half, :]) ** 2, axis=1)
    return int(start + np.argmax(power))
