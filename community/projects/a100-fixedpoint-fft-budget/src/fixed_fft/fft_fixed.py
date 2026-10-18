"""Fixed-point radix-2 FFT wordlength simulator for CTSAI-A100 budget studies.

The simulator models a B-bit two's-complement FFT datapath:

* input samples are signed integers (LSB units) fitted into the B-bit word
  with a minimal arithmetic right shift (recorded in ``input_shift``);
* radix-2 decimation-in-time butterflies run on exact ``numpy.int64``
  arithmetic, so the simulation is bit-exact and machine independent;
* twiddle factors are quantized to Q fractional bits in a separate
  coefficient word (``round(cos * 2**Q)`` / ``round(sin * 2**Q)``);
* complex products are truncated back to the datapath grid with an
  arithmetic right shift by Q (floor, as a simple hardware truncator would);
* after every stage the datapath optionally scales by 1 bit
  (``scale-1bit-each-stage``) using the same arithmetic right shift;
* values that leave [-2**(B-1), 2**(B-1)-1] are either saturated or wrapped
  (two's-complement overflow), and every occurrence is counted per stage.

IMPORTANT: this arithmetic format is a *simulation vehicle* for wordlength
budgeting only.  It does not model the internal number format, datapath
widths or scheduling of the CTSAI-A100 chip.  Only numerical error
statistics are reported; no runtime or memory measurements are made.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

SCALING_NONE = "none"
SCALING_SCALE_1BIT = "scale-1bit-each-stage"
SCALING_MODES = (SCALING_NONE, SCALING_SCALE_1BIT)

OVERFLOW_SATURATE = "saturate"
OVERFLOW_WRAP = "wrap"
OVERFLOW_MODES = (OVERFLOW_SATURATE, OVERFLOW_WRAP)

MIN_B = 8
MAX_B = 32
# Datapath values stay below 2**B and twiddle magnitudes at or below 2**Q,
# so complex products remain below 2**(B+Q).  The guard keeps every
# intermediate comfortably inside int64.
MAX_B_PLUS_Q = 60


@dataclass
class FixedFFTConfig:
    """Wordlength configuration of one simulation run."""

    B: int = 24                # internal datapath wordlength in bits (8..32)
    Q: int = 16                # twiddle coefficient fractional bits
    scaling: str = SCALING_NONE
    overflow: str = OVERFLOW_SATURATE

    def validate(self) -> None:
        if not isinstance(self.B, (int, np.integer)) or not (MIN_B <= self.B <= MAX_B):
            raise ValueError(f"B must be an integer in [{MIN_B}, {MAX_B}], got {self.B!r}")
        if not isinstance(self.Q, (int, np.integer)) or not (2 <= self.Q <= MAX_B_PLUS_Q - self.B):
            raise ValueError(
                f"Q must be an integer in [2, {MAX_B_PLUS_Q - self.B}] for B={self.B} "
                f"(int64 headroom guard B+Q<={MAX_B_PLUS_Q}), got {self.Q!r}"
            )
        if self.scaling not in SCALING_MODES:
            raise ValueError(f"scaling must be one of {SCALING_MODES}, got {self.scaling!r}")
        if self.overflow not in OVERFLOW_MODES:
            raise ValueError(f"overflow must be one of {OVERFLOW_MODES}, got {self.overflow!r}")


@dataclass
class FixedFFTResult:
    """Bit-exact result and per-stage statistics of one fixed-point FFT."""

    re: np.ndarray             # int64, real part of the N-point spectrum (LSB)
    im: np.ndarray             # int64, imaginary part (LSB)
    config: FixedFFTConfig
    n_fft: int
    n_input: int               # number of valid (non padded) input samples
    input_shift: int           # right shift applied to fit the input into B bits
    input_clipped: int         # samples clipped by fit_input (0 in normal use)
    stage_peak: List[int]      # max |stored value| after each stage (LSB)
    stage_overflow: List[int]  # stored values that left the B-bit range, per stage

    @property
    def total_overflow(self) -> int:
        return int(sum(self.stage_overflow))

    @property
    def n_stages(self) -> int:
        return len(self.stage_peak)

    @property
    def scaling(self) -> str:
        return self.config.scaling

    def spectrum(self) -> np.ndarray:
        """Spectrum as complex float, still in raw fixed-point LSB units."""
        return self.re.astype(np.float64) + 1j * self.im.astype(np.float64)

    def effective_gain(self) -> float:
        """Factor that maps the raw fixed-point spectrum back to input-LSB units.

        Undoes the input pre-shift and, for the 1-bit-per-stage schedule,
        the N = 2**n_stages total scaling.
        """
        gain = float(1 << self.input_shift)
        if self.config.scaling == SCALING_SCALE_1BIT:
            gain *= float(self.n_fft)
        return gain


def clip_to_word(x: np.ndarray, B: int) -> np.ndarray:
    """Saturate integer values into the B-bit two's-complement range."""
    lo = -(1 << (B - 1))
    hi = (1 << (B - 1)) - 1
    return np.clip(np.asarray(x, dtype=np.int64), lo, hi)


def fit_input(x: Sequence[int], B: int) -> Tuple[np.ndarray, int, int]:
    """Fit an integer input into the B-bit word.

    Applies the minimal arithmetic right shift so that every shifted sample
    lies inside [-2**(B-1), 2**(B-1)-1].  Because the two's-complement range
    is asymmetric, -2**(B-1) itself is representable and needs no shift.
    Returns ``(xq, shift, n_clipped)``; clipping only triggers for inputs
    that no shift can tame (e.g. INT64_MIN).
    """
    x = np.asarray(x, dtype=np.int64).ravel()
    lo = -(1 << (B - 1))
    hi = (1 << (B - 1)) - 1
    shift = 0
    xs = x
    while shift < 62:
        xs = np.right_shift(x, shift) if shift else x
        if xs.size == 0 or (xs.min() >= lo and xs.max() <= hi):
            break
        shift += 1
    n_clipped = int(np.count_nonzero((xs < lo) | (xs > hi)))
    return np.clip(xs, lo, hi).astype(np.int64), shift, n_clipped


def bit_reverse_permutation(n: int) -> np.ndarray:
    """Permutation index array of the radix-2 bit-reversal for length n."""
    bits = n.bit_length() - 1
    idx = np.arange(n, dtype=np.int64)
    rev = np.zeros(n, dtype=np.int64)
    for _ in range(bits):
        rev = (rev << np.int64(1)) | (idx & np.int64(1))
        idx = np.right_shift(idx, 1)
    return rev


def quantized_twiddles(n_fft: int, Q: int) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Quantized twiddle tables, one (cos, sin) pair of int64 arrays per stage.

    Stage s (1-based) has butterfly span 2**s and half = 2**(s-1) twiddles
    W = exp(-2j*pi*j/2**s) = (c - 1j*s) / 2**Q with c, s rounded to the
    nearest integer.  Exact values such as +-1 and +-1j quantize exactly.
    """
    n_stages = max(0, n_fft.bit_length() - 1)
    tables: List[Tuple[np.ndarray, np.ndarray]] = []
    for s in range(1, n_stages + 1):
        span = 1 << s
        half = span >> 1
        j = np.arange(half)
        angle = 2.0 * np.pi * j / span
        c = np.rint(np.cos(angle) * float(1 << Q)).astype(np.int64)
        sn = np.rint(np.sin(angle) * float(1 << Q)).astype(np.int64)
        tables.append((c, sn))
    return tables


def twiddle_quantization_error(n_fft: int, Q: int) -> float:
    """Largest absolute coefficient error introduced by Q-bit quantization."""
    err = 0.0
    for s, (c, sn) in enumerate(quantized_twiddles(n_fft, Q), start=1):
        span = 1 << s
        j = np.arange(c.size)
        angle = 2.0 * np.pi * j / span
        scale = float(1 << Q)
        err = max(err, float(np.max(np.abs(c / scale - np.cos(angle)))))
        err = max(err, float(np.max(np.abs(sn / scale - np.sin(angle)))))
    return err


def fft_fixed(
    x: Sequence[int],
    n_fft: Optional[int] = None,
    B: int = 24,
    Q: int = 16,
    scaling: str = SCALING_NONE,
    overflow: str = OVERFLOW_SATURATE,
) -> FixedFFTResult:
    """Run a B-bit fixed-point radix-2 DIT FFT over the real input ``x``.

    Parameters mirror :class:`FixedFFTConfig`.  ``n_fft`` defaults to the
    next power of two >= len(x); the input is zero-padded to ``n_fft``.
    """
    cfg = FixedFFTConfig(B=int(B), Q=int(Q), scaling=scaling, overflow=overflow)
    cfg.validate()

    xin = np.asarray(x, dtype=np.int64).ravel()
    if xin.size == 0:
        raise ValueError("input must contain at least one sample")
    if n_fft is None:
        n_fft = 1 << (xin.size - 1).bit_length()
    n_fft = int(n_fft)
    if n_fft < 1 or (n_fft & (n_fft - 1)) != 0:
        raise ValueError(f"n_fft must be a power of two, got {n_fft}")
    if n_fft < xin.size:
        raise ValueError(f"n_fft={n_fft} smaller than input length {xin.size}")

    xq, input_shift, input_clipped = fit_input(xin, cfg.B)

    re = np.zeros(n_fft, dtype=np.int64)
    re[: xq.size] = xq
    im = np.zeros(n_fft, dtype=np.int64)

    rev = bit_reverse_permutation(n_fft)
    re = re[rev]
    im = im[rev]

    lo = -(1 << (cfg.B - 1))
    hi = (1 << (cfg.B - 1)) - 1
    wrap_mask = np.int64((1 << cfg.B) - 1)
    wrap_sign = np.int64(1 << (cfg.B - 1))
    wrap_sub = np.int64(1 << cfg.B)
    scale = scaling == SCALING_SCALE_1BIT

    stage_peak: List[int] = []
    stage_overflow: List[int] = []
    tw_tables = quantized_twiddles(n_fft, cfg.Q)

    n_stages = n_fft.bit_length() - 1
    for s in range(1, n_stages + 1):
        half = 1 << (s - 1)
        span = 1 << s
        n_blocks = n_fft >> s
        c, sn = tw_tables[s - 1]

        block_r = re.reshape(n_blocks, span)
        block_i = im.reshape(n_blocks, span)
        ar = block_r[:, :half]
        ai = block_i[:, :half]
        br = block_r[:, half:]
        bi = block_i[:, half:]

        # complex product (br + 1j*bi) * (c - 1j*sn) / 2**Q, truncated (floor)
        tr = np.right_shift(br * c + bi * sn, np.int64(cfg.Q))
        ti = np.right_shift(bi * c - br * sn, np.int64(cfg.Q))

        ur = ar + tr
        ui = ai + ti
        vr = ar - tr
        vi = ai - ti

        if scale:
            ur = np.right_shift(ur, np.int64(1))
            ui = np.right_shift(ui, np.int64(1))
            vr = np.right_shift(vr, np.int64(1))
            vi = np.right_shift(vi, np.int64(1))

        over = (ur > hi) | (ur < lo)
        outs = []
        peak = 0
        for u in (ur, ui, vr, vi):
            over |= (u > hi) | (u < lo)
            if cfg.overflow == OVERFLOW_SATURATE:
                u = np.where(u > hi, np.int64(hi), np.where(u < lo, np.int64(lo), u))
            else:  # two's-complement wrap
                uw = u & wrap_mask
                u = np.where(uw >= wrap_sign, uw - wrap_sub, uw)
            peak = max(peak, int(np.abs(u).max()) if u.size else 0)
            outs.append(u.astype(np.int64))

        stage_overflow.append(int(over.sum()))
        stage_peak.append(peak)

        block_r[:, :half] = outs[0]
        block_i[:, :half] = outs[1]
        block_r[:, half:] = outs[2]
        block_i[:, half:] = outs[3]

    return FixedFFTResult(
        re=re,
        im=im,
        config=cfg,
        n_fft=n_fft,
        n_input=int(xin.size),
        input_shift=input_shift,
        input_clipped=input_clipped,
        stage_peak=stage_peak,
        stage_overflow=stage_overflow,
    )
