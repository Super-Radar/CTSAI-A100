"""Independent, moving-block, and stationary-block bootstrap for frame statistics.

* independent (iid) bootstrap -- resamples frames with replacement; valid only
  without serial correlation;
* moving-block bootstrap (MBB, Kunsch 1989) -- resamples contiguous blocks of
  ``b`` frames laid back to back (trimmed to N);
* stationary-block bootstrap (SBB, Politis & Romano 1994) -- circular resampling
  with random geometric block lengths of mean ``b``.

The block-length reference follows the optimal-rate rule ``b* = N^(1/3)``
(rounded, minimum 2).  Confidence intervals are percentile intervals at a
requested level.  Series shorter than 30 frames return
``insufficient_samples``; zero-variance series return ``zero_variance`` -- no
interval is produced in either case.
"""

from __future__ import annotations

import numpy as np

MIN_N = 30


def block_length_reference(n: int) -> int:
    """Reference block length b* = round(N^(1/3)), at least 2."""
    return int(max(2, round(n ** (1.0 / 3.0))))


def iid_bootstrap_means(x: np.ndarray, B: int, rng: np.random.Generator) -> np.ndarray:
    n = x.size
    idx = rng.integers(0, n, size=(B, n))
    return x[idx].mean(axis=1)


def mbb_indices(n: int, b: int, B: int, rng: np.random.Generator) -> np.ndarray:
    """Moving-block resample indices, shape (B, n); blocks never exceed n-1."""
    b = int(b)
    if b < 1 or b > n:
        raise ValueError(f"block length must be in [1, {n}], got {b}")
    nb = int(np.ceil(n / b))
    starts = rng.integers(0, n - b + 1, size=(B, nb))
    idx = starts[:, :, None] + np.arange(b, dtype=np.int64)[None, None, :]
    return idx.reshape(B, nb * b)[:, :n]


def moving_block_bootstrap_means(x: np.ndarray, b: int, B: int, rng: np.random.Generator) -> np.ndarray:
    return x[mbb_indices(x.size, b, B, rng)].mean(axis=1)


def stationary_block_bootstrap_means(x: np.ndarray, b: float, B: int, rng: np.random.Generator) -> np.ndarray:
    """Stationary (Politis-Romano) bootstrap means; geometric mean block length b."""
    n = x.size
    b = float(b)
    if b < 1.0:
        raise ValueError("mean block length must be >= 1")
    p = 1.0 / b
    pos = rng.integers(0, n, size=B)
    rem = rng.geometric(p, size=B)
    out = np.empty((B, n), dtype=np.float64)
    for t in range(n):
        new = rem <= 0
        pos = np.where(new, rng.integers(0, n, size=B), pos)
        rem = np.where(new, rng.geometric(p, size=B), rem)
        out[:, t] = x[pos]
        pos = np.where(pos + 1 < n, pos + 1, 0)
        rem = rem - 1
    return out.mean(axis=1)


def percentile_ci(means: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap interval (lo, hi) at the requested level."""
    alpha = 1.0 - level
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def _preflight(x: np.ndarray) -> str | None:
    if x.size < MIN_N:
        return "insufficient_samples"
    if float(np.var(x)) == 0.0:
        return "zero_variance"
    return None


def summarize_ci(
    x: np.ndarray,
    method: str,
    b: int | float | None = None,
    B: int = 2000,
    level: float = 0.95,
    rng: np.random.Generator | None = None,
) -> dict:
    """Percentile bootstrap CI of the mean by 'iid' | 'mbb' | 'sbb'.

    Returns a dict with status and, when usable, lo/hi/width plus the sample
    mean and the block length actually used.
    """
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    rng = np.random.default_rng() if rng is None else rng
    out = {
        "status": "ok",
        "method": method,
        "n": n,
        "mean": float(x.mean()),
        "b": None,
        "lo": None,
        "hi": None,
        "width": None,
        "level": level,
    }
    bad = _preflight(x)
    if bad is not None:
        out["status"] = bad
        return out
    if method == "iid":
        means = iid_bootstrap_means(x, B, rng)
    elif method == "mbb":
        bb = int(b) if b is not None else block_length_reference(n)
        bb = int(min(max(bb, 1), n))
        out["b"] = bb
        means = moving_block_bootstrap_means(x, bb, B, rng)
    elif method == "sbb":
        bb = float(b) if b is not None else float(block_length_reference(n))
        out["b"] = bb
        means = stationary_block_bootstrap_means(x, bb, B, rng)
    else:
        raise ValueError(f"unknown method {method!r}")
    lo, hi = percentile_ci(means, level)
    out.update({"lo": lo, "hi": hi, "width": hi - lo})
    return out
