"""Autocorrelation, truncation-based effective sample size, and CUSUM segmentation.

Effective sample size
---------------------
The biased (1/N) sample autocorrelation ``rho_k`` is estimated up to
``max_lag = min(100, floor(0.05 N))`` lags.  The truncation-based effective
sample size is::

    N_eff(K) = N / (1 + 2 * sum_{k=1..K} rho_k)

The reported ``N_eff`` uses the initial-positive-sequence (IPS) truncation:
sum ``rho_k`` for ``k = 1, 2, ...`` until the first non-positive estimate
(capped at ``max_lag``).  For iid data this keeps only a couple of small
noise terms, so ``N_eff ~= N``; for positively autocorrelated data it captures
the persistence.  ``N_eff`` is clipped to ``[1, 10 N]``.  The full stability
curve ``N_eff(K)`` for ``K = 1..max_lag`` is also returned.

Stationary segmentation (predefined rule, documented in the README)
-------------------------------------------------------------------
Offline CUSUM test for one mean change at an unknown point: with centered
cumulative sums ``S_m`` of a segment of length ``n`` and a robust scale
``sigma_hat`` from first differences, the statistic is
``T = max_m |S_m| / (sigma_hat * sqrt(n))`` whose null limit is the supremum
of a Brownian bridge.  A split is accepted when ``T > CUSUM_THRESHOLD``
(1.63, the 1% critical value), both children are at least ``MIN_SEGMENT``
frames long, and at most ``MAX_CHANGE_POINTS`` splits are collected per
series; the procedure then recurses on both children.  Segments shorter than
``MIN_SEGMENT`` frames are reported as ``insufficient_samples`` instead of
receiving a confidence interval.
"""

from __future__ import annotations

import numpy as np

MAX_LAG_CAP = 100
LAG_FRACTION = 0.05
MIN_SEGMENT = 30
CUSUM_THRESHOLD = 1.63  # 1% critical value of sup|Brownian bridge|
MAX_CHANGE_POINTS = 20
NEFF_CLIP_MAX_FACTOR = 10.0


class ZeroVarianceError(ValueError):
    """Raised when a series has zero variance."""


class InsufficientSamplesError(ValueError):
    """Raised when fewer than MIN_SEGMENT frames are available."""


def max_lag_for(n: int) -> int:
    return int(max(1, min(MAX_LAG_CAP, np.floor(LAG_FRACTION * n))))


def autocorr(x: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """Biased sample autocorrelation rho_0..rho_max_lag (rho_0 = 1)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n < 2:
        raise InsufficientSamplesError("need at least 2 samples")
    xm = x - x.mean()
    denom = float(np.dot(xm, xm))
    if denom <= 0.0:
        raise ZeroVarianceError("zero-variance series")
    if max_lag is None:
        max_lag = max_lag_for(n)
    max_lag = int(min(max_lag, n - 1))
    full = np.correlate(xm, xm, mode="full")[n - 1 :]
    rho = full[: max_lag + 1] / (denom)
    rho[0] = 1.0
    return rho


def ips_truncation(rho: np.ndarray) -> int:
    """Initial-positive-sequence truncation lag K."""
    k = 0
    while k + 1 < rho.size and rho[k + 1] > 0.0:
        k += 1
    return int(k)


def neff_curve(n: int, rho: np.ndarray) -> np.ndarray:
    """N_eff(K) for K = 1..max_lag with a clipped denominator (same clip as N_eff)."""
    denom = 1.0 + 2.0 * np.cumsum(rho[1:])
    denom = np.clip(denom, 1.0 / (NEFF_CLIP_MAX_FACTOR * max(n, 1)), None)
    return np.clip(n / denom, 1.0, NEFF_CLIP_MAX_FACTOR * n)


def analyze_effective_n(x: np.ndarray) -> dict:
    """Status, autocorrelation, IPS-truncated N_eff and its stability curve."""
    x = np.asarray(x, dtype=np.float64)
    n = int(x.size)
    out = {
        "status": "ok",
        "n": n,
        "mean": float(x.mean()) if n else float("nan"),
        "sd": float(x.std(ddof=1)) if n > 1 else float("nan"),
        "rho": None,
        "max_lag": 0,
        "k_ips": 0,
        "neff": None,
        "neff_ratio": None,
        "curve": None,
    }
    if n < MIN_SEGMENT:
        out["status"] = "insufficient_samples"
        return out
    if float(np.var(x)) == 0.0:
        out["status"] = "zero_variance"
        return out
    rho = autocorr(x)
    k_ips = ips_truncation(rho)
    curve = neff_curve(n, rho)
    denom = 1.0 + 2.0 * float(np.sum(rho[1 : k_ips + 1]))
    denom = max(denom, 1.0 / (NEFF_CLIP_MAX_FACTOR * n))
    neff = float(np.clip(n / denom, 1.0, NEFF_CLIP_MAX_FACTOR * n))
    out.update(
        {
            "rho": rho,
            "max_lag": int(rho.size - 1),
            "k_ips": k_ips,
            "neff": neff,
            "neff_ratio": neff / n,
            "curve": curve,
        }
    )
    return out


def _cusum_best_split(seg: np.ndarray, threshold: float, min_seg: int):
    """Return split index m (left child length) if T exceeds threshold, else None."""
    n = seg.size
    if n < 2 * min_seg:
        return None
    sigma = float(np.sqrt(np.mean(np.diff(seg) ** 2) / 2.0))
    if sigma <= 0.0:
        return None
    s = np.cumsum(seg - seg.mean())
    lo, hi = min_seg, n - min_seg
    if hi < lo:
        return None
    stat = np.abs(s[lo - 1 : hi]) / (sigma * np.sqrt(n))
    j = int(np.argmax(stat))
    if stat[j] > threshold:
        return lo + j
    return None


def cusum_change_points(
    x: np.ndarray,
    threshold: float = CUSUM_THRESHOLD,
    min_seg: int = MIN_SEGMENT,
    max_cps: int = MAX_CHANGE_POINTS,
) -> list[int]:
    """Recursive CUSUM mean-change change-point locations (left-child lengths)."""
    x = np.asarray(x, dtype=np.float64)
    cps: list[int] = []

    def rec(a: int, b: int) -> None:
        if len(cps) >= max_cps:
            return
        m = _cusum_best_split(x[a:b], threshold, min_seg)
        if m is None:
            return
        split = a + m
        cps.append(split)
        rec(a, split)
        rec(split, b)

    rec(0, x.size)
    return sorted(cps)


def segments_of(x: np.ndarray, cps: list[int] | None = None) -> list[dict]:
    """Segments (start, end, status) implied by the change points."""
    x = np.asarray(x, dtype=np.float64)
    if cps is None:
        cps = cusum_change_points(x)
    bounds = [0, *cps, x.size]
    segs = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        status = "ok" if (b - a) >= MIN_SEGMENT else "insufficient_samples"
        segs.append({"start": int(a), "end": int(b), "n": int(b - a), "status": status})
    return segs
