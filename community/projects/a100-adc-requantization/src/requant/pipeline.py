"""Unpacking and the official-equivalent spectral chain.

This module mirrors the reference MATLAB implementation shipped with the
repository:

* ``init_function/load_adc_data.m``
    - csv column of uint32 -> truncate/pad to [rng_nfft/2, nchirp],
      reshape then transpose to [nchirp, rng_nfft/2, ch];
    - every uint32 word splits into two *signed int16 real* samples
      (high 16 bits first, low 16 bits second; two's complement).
      Note: these are real-valued samples, not complex I/Q.
* ``init_function/adc_mem_2_real_fixed.m``
    - H16 = floor(w / 2^16), L16 = mod(w, 2^16), each mapped from
      [0, 2^16) to signed via subtracting 2^16 when >= 2^15.
* ``FFT/do_fft_2d.m``
    - fast time: hann(adc_sample_num) on the first adc_sample_num samples
      (MATLAB ``hann`` is the *symmetric* window), zero-pad the FFT to
      rng_nfft, divide by rng_fft_scalar, keep the first rng_nfft/2 bins;
    - slow time: hann(nchirp / nvirtual_chirp), FFT to vel_nfft, divide by
      vel_fft_scalar.

Divergences from the MATLAB code are deliberate and reported loudly rather
than silently: when a data file is longer than the reshape target the MATLAB
code truncates with a ``warning``; here :func:`load_adc_txt` records every
excess value (the demo files each carry exactly one trailing 0) and the CLI
prints the first excess value explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal.windows import hann

from .config import RadarCfg

INT16_MIN = -32768
INT16_MAX = 32767


# --------------------------------------------------------------------- load
@dataclass
class LoadResult:
    """Raw uint32 word stream plus an explicit account of any excess values."""

    words: np.ndarray                 # uint32, length cfg.words_per_channel
    n_values: int                     # values actually present in the file
    expected: int                     # cfg.words_per_channel
    excess_count: int                 # n_values - expected (>0 => truncated)
    excess_head: list                 # first excess values, reported not hidden
    padded: int                       # zeros appended when short (<0 never)
    source: str = ""

    @property
    def had_excess(self) -> bool:
        return self.excess_count > 0


def load_adc_txt(path: str | Path, cfg: RadarCfg) -> LoadResult:
    """Read one comma-separated uint32 ADC dump for a single RX channel.

    The file is a headerless single line of unsigned integers.  Values beyond
    ``cfg.words_per_channel`` are *reported* (first few kept in
    ``excess_head``) and then truncated to match the official reshape target,
    exactly like ``load_adc_data.m`` but loudly.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    vals = np.array([int(t) for t in text.split(",") if t.strip() != ""],
                    dtype=np.uint32)
    n_values = int(vals.size)
    expected = cfg.words_per_channel
    padded = 0
    if n_values > expected:
        excess_count = n_values - expected
        excess_head = [int(v) for v in vals[expected:expected + 4]]
        words = vals[:expected].copy()
    elif n_values < expected:
        excess_count = 0
        excess_head = []
        padded = expected - n_values
        words = np.concatenate([vals, np.zeros(padded, dtype=np.uint32)])
    else:
        excess_count = 0
        excess_head = []
        words = vals
    return LoadResult(words=words, n_values=n_values, expected=expected,
                      excess_count=excess_count, excess_head=excess_head,
                      padded=padded, source=str(path))


# ------------------------------------------------------------------ unpack
def unpack_words(words: np.ndarray, cfg: RadarCfg) -> np.ndarray:
    """uint32 words -> signed real samples, shape [rng_nfft, nchirp].

    Equivalent to MATLAB ``permute(adc_mem_2_real_fixed(adcTemp2'), [2 1])``
    with nvirtual_chirp = 1: the returned matrix is indexed
    [fast-time sample, chirp] exactly like ``adcData`` entering
    ``do_fft_2d.m``.
    """
    w = np.asarray(words, dtype=np.uint32)
    half = cfg.rng_nfft // 2
    if w.size != half * cfg.nchirp:
        raise ValueError(
            f"word count {w.size} != (rng_nfft/2)*nchirp = {half * cfg.nchirp}")
    # MATLAB reshape(col, [rng_nfft/2, nchirp]) fills column-major, i.e.
    # consecutive blocks of rng_nfft/2 words belong to consecutive chirps;
    # the subsequent transpose makes [nchirp, rng_nfft/2].  A C-order
    # reshape to (nchirp, half) is identical.
    w2 = w.reshape(cfg.nchirp, half)
    h = (w2 >> np.uint32(16)).astype(np.int32)
    low = (w2 & np.uint32(0xFFFF)).astype(np.int32)
    h = np.where(h >= 1 << 15, h - (1 << 16), h)
    low = np.where(low >= 1 << 15, low - (1 << 16), low)
    out = np.empty((cfg.nchirp, cfg.rng_nfft), dtype=np.int32)
    out[:, 0::2] = h    # H16 first ...
    out[:, 1::2] = low  # ... L16 second
    return out.T        # -> [rng_nfft, nchirp]


def pack_samples(samples: np.ndarray, cfg: RadarCfg) -> np.ndarray:
    """Inverse of :func:`unpack_words` (used by the round-trip test)."""
    s = np.asarray(samples, dtype=np.int64).T          # [nchirp, rng_nfft]
    h = (s[:, 0::2] & 0xFFFF).astype(np.uint32)
    low = (s[:, 1::2] & 0xFFFF).astype(np.uint32)
    return ((h << np.uint32(16)) | low).reshape(-1)


# ---------------------------------------------------------- spectral chain
def get_windows(cfg: RadarCfg) -> tuple[np.ndarray, np.ndarray]:
    """(range window, Doppler window), MATLAB ``hann(N)`` symmetric."""
    return (hann(cfg.adc_sample_num, sym=True),
            hann(cfg.nchirp // cfg.nvirtual_chirp, sym=True))


def spectral_chain(samples: np.ndarray, cfg: RadarCfg
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Window + 2D-FFT chain, mirroring ``FFT/do_fft_2d.m``.

    Parameters
    ----------
    samples : [rng_nfft, nchirp] integer/float sample matrix (adcData).

    Returns
    -------
    range_spectrum : [rng_nfft/2] mean |Range-FFT| over chirps.
    rd_map        : [rng_nfft/2, vel_nfft] |Range-Doppler| magnitude.
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.shape != (cfg.rng_nfft, cfg.nchirp):
        raise ValueError(f"sample shape {x.shape} != "
                         f"{(cfg.rng_nfft, cfg.nchirp)}")
    nsm = cfg.adc_sample_num
    rng_win, vel_win = get_windows(cfg)
    # Range windowing on the first adc_sample_num samples only.
    xw = np.zeros_like(x)
    xw[:nsm, :] = x[:nsm, :] * rng_win[:, None]
    # Range FFT, zero padded to rng_nfft, firmware scalar, first half bins.
    rng_f = np.fft.fft(xw, n=cfg.rng_nfft, axis=0) / float(cfg.rng_fft_scalar)
    half = cfg.rng_nfft // 2
    rng_f = rng_f[:half, :]
    # Doppler windowing + FFT on the retained range bins.
    vel_f = np.fft.fft(rng_f * vel_win[None, :], n=cfg.vel_nfft, axis=1) \
        / float(cfg.vel_fft_scalar)
    return np.abs(rng_f).mean(axis=1), np.abs(vel_f)
