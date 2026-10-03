"""Lossy bit-reduction of int16 ADC samples at a fixed full scale.

Full scale is fixed at the int16 storage range [-32768, 32767]; the step for
a target word length of ``k`` bits is ``delta = 2**(16-k)``.  No per-file
normalisation is performed anywhere in this project: reducing the stored word
length must degrade small signals once ``delta`` approaches their amplitude,
and hiding that behind a per-file gain would defeat the budget study.

Two requantisation laws are implemented:

* ``round``  - round-half-away-from-zero to the k-bit grid.  In the no-clip
  region the error is bounded by ``delta/2`` and is (nearly) unbiased; the
  classic model gives quantisation-noise power ``delta**2 / 12``.
* ``trunc``  - drop the low ``16-k`` bits (two's-complement arithmetic right
  shift, i.e. floor).  The error ``x - Q(x)`` lies in ``[0, delta)`` and is
  biased by ``+delta/2`` on average.

The only clipping in the round law is the half-step overshoot at the positive
rail (e.g. 32767 can round up to 32768 and is clamped back to 32767); values
below that stay inside [-32768, 32767].  ``trunc`` never leaves the range.
"""

from __future__ import annotations

import numpy as np

from .pipeline import INT16_MAX, INT16_MIN

MIN_K = 1
MAX_K = 16


def check_k(k: int) -> int:
    if not isinstance(k, (int, np.integer)) or not (MIN_K <= int(k) <= MAX_K):
        raise ValueError(f"k must be an integer in [{MIN_K}, {MAX_K}], got {k!r}")
    return int(k)


def step_for_k(k: int) -> int:
    """Quantisation step (LSB count) for a k-bit grid of the int16 scale."""
    k = check_k(k)
    return 1 << (16 - k)


def requantize(x: np.ndarray, k: int, mode: str = "round") -> np.ndarray:
    """Requantise int16 samples onto the k-bit grid at fixed full scale.

    Parameters
    ----------
    x   : integer sample array (values assumed within int16).
    k   : target word length in bits, 1..16.  k = 16 is the identity.
    mode: "round" (round-half-away-from-zero) or "trunc" (drop low bits).

    Returns
    -------
    int32 array of the same shape, values on the k-bit grid.
    """
    k = check_k(k)
    mode = str(mode).lower()
    if mode not in ("round", "trunc"):
        raise ValueError(f"mode must be 'round' or 'trunc', got {mode!r}")
    x = np.asarray(x, dtype=np.int64)
    if k == MAX_K:
        return x.astype(np.int32)          # identity, bit-exact
    step = 1 << (16 - k)
    if mode == "trunc":
        # Two's-complement right shift == floor division by 2**shift.
        q = (x >> (16 - k)) << (16 - k)
        return q.astype(np.int32)
    # round-half-away-from-zero, in exact integer arithmetic.
    mag = np.abs(x) + (step >> 1)
    qmag = (mag // step) * step
    q = np.where(x < 0, -qmag, qmag)
    # Half-step clip only at the positive rail.
    q = np.clip(q, INT16_MIN, INT16_MAX)
    return q.astype(np.int32)


def quantization_error(x: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Signed error ``x - Q(x)`` in LSB units (positive = value reduced)."""
    return np.asarray(x, dtype=np.float64) - np.asarray(q, dtype=np.float64)


def on_grid(x: np.ndarray, k: int) -> np.ndarray:
    """True where x already lies on the k-bit grid (identity points)."""
    step = step_for_k(k)
    return np.asarray(x, dtype=np.int64) % step == 0
