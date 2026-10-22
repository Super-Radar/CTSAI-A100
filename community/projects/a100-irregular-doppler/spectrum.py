"""Blockwise nonuniform Fourier projection with explicit sample timestamps."""

import numpy as np


def spectrum(samples, times, frequencies, weights=None, block=128):
    """Return weighted complex projection relative to the first input time."""
    x = np.asarray(samples, dtype=complex)
    t = np.asarray(times, dtype=float)
    f = np.asarray(frequencies, dtype=float)
    if x.ndim != 1 or t.shape != x.shape or len(x) < 2 or f.ndim != 1 or not f.size:
        raise ValueError(
            "Expected equal 1D sample/time arrays and a nonempty frequency grid"
        )
    if (
        not np.isfinite(x).all()
        or not np.isfinite(t).all()
        or not np.isfinite(f).all()
        or np.any(np.diff(t) <= 0)
    ):
        raise ValueError("Finite values and strictly increasing times required")
    if not isinstance(block, int) or block < 1:
        raise ValueError("Positive integer block required")
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=float)
    if (
        w.shape != x.shape
        or not np.isfinite(w).all()
        or np.any(w < 0)
        or not np.any(w > 0)
    ):
        raise ValueError("Finite nonnegative weights with positive sum required")
    # Normalize before multiplying samples: an arbitrary weight scale must not
    # overflow either the denominator or the weighted sample vector.
    w = w / w.max()
    w = w / w.sum()
    t = t - t[0]
    result = np.empty(len(f), dtype=complex)
    for start in range(0, len(f), block):
        grid = f[start : start + block]
        result[start : start + block] = np.exp(-2j * np.pi * grid[:, None] * t) @ (
            w * x
        )
    return result


def sampling_window(times, frequencies, weights=None):
    return np.abs(spectrum(np.ones(len(times)), times, frequencies, weights)) ** 2
