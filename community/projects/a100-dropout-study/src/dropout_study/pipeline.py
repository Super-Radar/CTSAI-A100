"""Official CTSAI-A100 ADC -> Range-Doppler chain, reproduced in NumPy.

This module mirrors the official MATLAB platform
``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/`` so the
spectral-impact experiments can run offline without MATLAB:

* ``init_function/adc_mem_2_real_fixed.m`` -> :func:`unpack_words`
  (one uint32 word -> high-16 / low-16 signed int16 pair, high first)
* ``init_function/load_adc_data.m``        -> :func:`load_adc_words`
  (normalise length to ``rng_nfft/2 * nchirp`` words: truncate when longer,
  zero-pad when shorter; the extra trailing value present in the shipped
  example files is truncated exactly like the official loader, and every
  decision is recorded in a :class:`LoadReport`)
* ``FFT/do_fft_2d.m``                      -> :func:`rv_power_map`
  (hann range window over ``adc_sample_num`` samples -> Range FFT of
  ``rng_nfft`` points, first half kept -> hann velocity window over the
  surviving chirps -> Doppler FFT of ``vel_nfft`` points)
* ``ct_signal_processing_main_simple_CTASIA100.m`` -> :func:`rv_map_average`
  (RVimage = mean of the per-channel power maps)

Config values for ``init0`` (far wave, f0) and ``init1`` (near wave, f1) are
transcribed from ``cfg/CTASI-A100配置/sensor_config_init{0,1}.hxx``;
``adc_sample_num`` follows ``init_function/get_radar_paramete.m``
(``dec_factor == 1`` branch, no ``+1``).

The chain reproduced here is the same one validated natively in MATLAB by the
community project ``a100-matlab-adc-pipeline-validation``; this project does
not re-run MATLAB, it re-implements the arithmetic in NumPy.

Finally, :func:`synth_single_target_words` generates a synthetic single-target
ADC word stream (known range / velocity) used as ground truth by the study.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from scipy.signal.windows import hann

C_LIGHT = 299_792_458.0  # m/s (matches the 3e8 used by the official scripts closely enough for bin arithmetic)


@dataclass(frozen=True)
class RadarConfig:
    """Sensor parameters transcribed from a ``sensor_config_initN.hxx`` file."""

    name: str
    f0_ghz: float          # .fmcw_startfreq
    bw_mhz: float          # .fmcw_bandwidth
    rampup_us: float       # .fmcw_chirp_rampup
    period_us: float       # .fmcw_chirp_period
    nchirp: int            # .nchirp
    adc_freq_mhz: float    # .adc_freq
    sample_start_us: float # .adc_sample_start
    sample_end_us: float   # .adc_sample_end
    dec_factor: int        # .dec_factor
    rng_nfft: int          # .rng_nfft
    vel_nfft: int          # .vel_nfft
    rng_fft_scalar: int    # .rng_fft_scalar
    vel_fft_scalar: int    # .vel_fft_scalar

    @property
    def words_per_chirp(self) -> int:
        return self.rng_nfft // 2

    @property
    def samples_per_chirp(self) -> int:
        return self.rng_nfft

    @property
    def expected_words(self) -> int:
        return self.words_per_chirp * self.nchirp

    @property
    def adc_sample_num(self) -> int:
        """get_radar_paramete.m: dec_factor==1 -> round((end-start)*fs), no +1; capped at rng_nfft."""
        span = (self.sample_end_us - self.sample_start_us) * self.adc_freq_mhz / self.dec_factor
        n = round(span) + 1 if self.dec_factor > 1 else round(span)
        return int(min(n, self.rng_nfft))

    @property
    def lambda_m(self) -> float:
        return C_LIGHT / (self.f0_ghz * 1e9)

    @property
    def slope_hz_s(self) -> float:
        return self.bw_mhz * 1e6 / (self.rampup_us * 1e-6)

    @property
    def range_per_bin_m(self) -> float:
        """Metres per Range-FFT bin (first half of the rng_nfft-point FFT)."""
        return C_LIGHT * (self.adc_freq_mhz * 1e6 / self.rng_nfft) / (2.0 * self.slope_hz_s)

    @property
    def vel_per_bin_mps(self) -> float:
        """Metres/second per Doppler bin of the vel_nfft-point FFT over chirps spaced by the chirp period."""
        return self.lambda_m / (2.0 * self.vel_nfft * self.period_us * 1e-6)

    def range_bin_of(self, range_m: float) -> float:
        beat = 2.0 * self.slope_hz_s * range_m / C_LIGHT
        return beat * self.rng_nfft / (self.adc_freq_mhz * 1e6)

    def doppler_bin_of(self, vel_mps: float) -> float:
        fd = 2.0 * vel_mps / self.lambda_m
        return fd * self.period_us * 1e-6 * self.vel_nfft


INIT0 = RadarConfig(
    name="init0 (far wave, f0)",
    f0_ghz=76.3, bw_mhz=300.0, rampup_us=43.0, period_us=48.0,
    nchirp=256, adc_freq_mhz=25.0, sample_start_us=2.0, sample_end_us=42.8,
    dec_factor=1, rng_nfft=1024, vel_nfft=256,
    rng_fft_scalar=255, vel_fft_scalar=63,
)
INIT1 = RadarConfig(
    name="init1 (near wave, f1)",
    f0_ghz=76.119, bw_mhz=750.0, rampup_us=104.65, period_us=108.5,
    nchirp=128, adc_freq_mhz=25.0, sample_start_us=2.0, sample_end_us=104.4,
    dec_factor=1, rng_nfft=2048, vel_nfft=128,
    rng_fft_scalar=255, vel_fft_scalar=63,
)


@dataclass
class LoadReport:
    """Record of every length-normalisation decision made by :func:`load_adc_words`."""

    path: str
    sha256: str
    n_values: int
    expected_words: int
    n_extra_truncated: int
    n_zero_padded: int
    notes: list = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "path": self.path,
            "sha256": self.sha256,
            "n_values": self.n_values,
            "expected_words": self.expected_words,
            "n_extra_truncated": self.n_extra_truncated,
            "n_zero_padded": self.n_zero_padded,
            "notes": " | ".join(self.notes),
        }


def read_adc_txt(path) -> np.ndarray:
    """Read a headerless ADC dump: one line of comma-separated uint32 values."""
    text = Path(path).read_text(encoding="ascii").strip()
    tokens = text.split(",")
    return np.fromiter((int(t) for t in tokens), dtype=np.uint32, count=len(tokens))


def load_adc_words(path, cfg: RadarConfig):
    """Load one ADC txt file into the per-chirp word matrix ``[nchirp, words_per_chirp]``.

    Mirrors ``load_adc_data.m``: values beyond ``expected_words`` are truncated
    with a recorded warning; a short file is zero-padded with a recorded
    warning.  The flat word stream maps to chirps exactly like the official
    MATLAB ``reshape(col, [rng_nfft/2, nchirp])`` followed by a transpose:
    flat index ``w`` -> chirp ``w // words_per_chirp``, word ``w % words_per_chirp``.
    """
    raw = read_adc_txt(path)
    n = int(raw.size)
    exp = cfg.expected_words
    report = LoadReport(
        path=str(path),
        sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        n_values=n,
        expected_words=exp,
        n_extra_truncated=0,
        n_zero_padded=0,
    )
    if n > exp:
        report.n_extra_truncated = n - exp
        report.notes.append(
            f"file holds {n} values, {n - exp} more than the expected {exp}; "
            "extra trailing values truncated (official load_adc_data.m behaviour)"
        )
        words = raw[:exp].copy()
    elif n < exp:
        report.n_zero_padded = exp - n
        report.notes.append(
            f"file holds {n} values, {exp - n} fewer than the expected {exp}; "
            "zero-padded to length (official load_adc_data.m behaviour)"
        )
        words = np.concatenate([raw, np.zeros(exp - n, dtype=np.uint32)])
    else:
        report.notes.append(f"file length {n} matches the expected {exp}")
        words = raw.copy()
    return words.reshape(cfg.nchirp, cfg.words_per_chirp), report


def unpack_words(words: np.ndarray) -> np.ndarray:
    """uint32 words -> interleaved int16 samples (adc_mem_2_real_fixed.m).

    Each 32-bit word carries two consecutive signed 16-bit samples: the high
    half first, then the low half.  Input may be ``[nchirp, words_per_chirp]``
    or flat; output keeps the leading dimension(s).
    """
    w = np.asarray(words, dtype=np.uint32)
    hi = (w >> np.uint32(16)).astype(np.uint16).astype(np.int16)
    lo = (w & np.uint32(0xFFFF)).astype(np.uint16).astype(np.int16)
    out = np.empty(w.shape + (2,), dtype=np.int16)
    out[..., 0] = hi
    out[..., 1] = lo
    return out.reshape(w.shape[0], -1)


def pack_words(samples: np.ndarray) -> np.ndarray:
    """Inverse of :func:`unpack_words`: interleaved int16 samples -> uint32 words."""
    s = np.asarray(samples, dtype=np.int16)
    pair = s.reshape(s.shape[0], -1, 2)
    hi = pair[..., 0].astype(np.uint16).astype(np.uint32)
    lo = pair[..., 1].astype(np.uint16).astype(np.uint32)
    return (hi << np.uint32(16)) | lo


def rv_power_map(frame_samples, cfg: RadarConfig):
    """Range-Doppler power map of one channel (do_fft_2d.m + main script).

    ``frame_samples``: int matrix ``[nchirp_used, samples_per_chirp]`` where
    ``nchirp_used`` may be smaller than ``cfg.nchirp`` (reject strategy); the
    velocity window then covers the surviving chirps only, the Doppler FFT is
    still ``vel_nfft`` points (zero-padded).  Returns the power map as
    ``[rng_nfft/2, vel_nfft]`` (rows = range bins, cols = Doppler bins), or
    ``None`` when no chirp survives.
    """
    if frame_samples is None or np.asarray(frame_samples).shape[0] == 0:
        return None
    nchirp_used = int(np.asarray(frame_samples).shape[0])
    rng_win = hann(cfg.adc_sample_num, sym=True)  # MATLAB hann(N) is symmetric
    x = np.asarray(frame_samples, dtype=np.float64)[:, : cfg.adc_sample_num]
    x = x * rng_win[None, :]
    rng_fft = np.fft.fft(x, n=cfg.rng_nfft, axis=1) / float(cfg.rng_fft_scalar)
    half = rng_fft[:, : cfg.rng_nfft // 2]
    vel_win = hann(nchirp_used, sym=True)
    y = half * vel_win[:, None]
    vel_fft = np.fft.fft(y, n=cfg.vel_nfft, axis=0) / float(cfg.vel_fft_scalar)
    return (np.abs(vel_fft) ** 2).T


def rv_map_average(frames, cfg: RadarConfig):
    """Official RVimage: mean of the per-channel power maps; None if no usable channel."""
    maps = [m for m in (rv_power_map(f, cfg) for f in frames) if m is not None]
    if not maps:
        return None
    return np.mean(maps, axis=0)


def synth_single_target_words(
    cfg: RadarConfig,
    range_m: float,
    vel_mps: float,
    amp_lsb: float = 12000.0,
    noise_lsb: float = 2.0,
    seed: int = 20260929,
) -> np.ndarray:
    """Synthesise a single-target ADC word stream ``[nchirp, words_per_chirp]``.

    The beat signal of one point target at ``range_m`` moving at ``vel_mps``:
    a real cosine whose fast-time frequency is the beat frequency
    ``f_b = 2*S*R/c`` and whose slow-time phase advances by ``2*pi*f_d*T_c``
    per chirp with ``f_d = 2*v/lambda``.  Samples are int16-quantised with a
    small Gaussian noise floor so the frame is not perfectly noiseless.
    The expected FFT peak locations are ``cfg.range_bin_of(range_m)`` and
    ``cfg.doppler_bin_of(vel_mps)`` (Doppler bins are taken modulo vel_nfft).
    """
    rng = np.random.default_rng(seed)
    fs = cfg.adc_freq_mhz * 1e6
    n = np.arange(cfg.samples_per_chirp)
    m = np.arange(cfg.nchirp)[:, None]
    fb = 2.0 * cfg.slope_hz_s * range_m / C_LIGHT
    fd = 2.0 * vel_mps / cfg.lambda_m
    phase = 2.0 * np.pi * (fb * n[None, :] / fs + fd * m * cfg.period_us * 1e-6)
    sig = amp_lsb * np.cos(phase) + noise_lsb * rng.standard_normal(phase.shape)
    samples = np.clip(np.round(sig), -32768, 32767).astype(np.int16)
    return pack_words(samples)


__all__ = [
    "RadarConfig", "INIT0", "INIT1", "LoadReport",
    "read_adc_txt", "load_adc_words", "unpack_words", "pack_words",
    "rv_power_map", "rv_map_average", "synth_single_target_words", "replace",
]
