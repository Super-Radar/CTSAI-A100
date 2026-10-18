"""Robust alternatives to the official mean-of-fixed-region noise estimator.

All estimators operate on the same input as the official method: the dB
values of the (channel-averaged) RV image inside a noise region.  For the
main comparisons the region is the *same* official min-of-halves region, so
any difference between estimators is attributable to the estimator alone,
not to the region.

Definitions (x = the n region values in dB):

* ``official_mean`` : ``mean(x)`` -- the official ``calculate_global_noise``.
* ``median``        : ``median(x)``.
* ``q25`` / ``q10`` : quantiles ``Q_0.25(x)`` / ``Q_0.10(x)`` (linear
  interpolation, numpy default).
* ``trim10`` / ``trim20``: trimmed means discarding ``floor(p*n)`` values
  from *each* tail before averaging (p = 0.10 / 0.20).
* ``mad_floor``     : ``median(x) + 1.4826 * MAD(x)`` with
  ``MAD(x) = median(|x - median(x)|)``; a robust "median + 1 sigma"
  (Gaussian-equivalent) level, where ``1.4826 = 1 / Phi^-1(3/4)`` makes
  ``1.4826 * MAD`` a consistent standard-deviation estimate for Gaussian
  data.  NOTE: this estimates a *different statistical target* than the
  official mean (median + 1 sigma rather than the mean of dB values); it is
  included because its *deviation under contamination / region change* is
  the quantity of interest in this study.

Bootstrap confidence intervals resample the region cells with replacement
(percentile CI) and are evaluated for any estimator.

Honesty note: no empty-scene ground truth exists for the public captures, so
nothing here claims an *absolute* noise floor; all conclusions are phrased
as deviations relative to each estimator's value on the uncontaminated data.
"""

from __future__ import annotations

from typing import Callable, Dict, Sequence, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Estimators
# ---------------------------------------------------------------------------


def official_mean(x: np.ndarray) -> float:
    return float(np.mean(np.ravel(x)))


def median(x: np.ndarray) -> float:
    return float(np.median(np.ravel(x)))


def quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(np.ravel(x), q))


def trimmed_mean(x: np.ndarray, frac: float) -> float:
    """Discard floor(frac*n) values from each tail, then average."""
    if not 0.0 <= frac < 0.5:
        raise ValueError("frac must be in [0, 0.5)")
    xs = np.sort(np.ravel(np.asarray(x, dtype=float)))
    n = xs.size
    k = int(np.floor(frac * n))
    if 2 * k >= n:
        raise ValueError("trimming fraction removes all samples")
    return float(np.mean(xs[k : n - k]))


def mad(x: np.ndarray) -> float:
    xf = np.ravel(np.asarray(x, dtype=float))
    med = np.median(xf)
    return float(np.median(np.abs(xf - med)))


def mad_noise_floor(x: np.ndarray) -> float:
    return median(x) + 1.4826 * mad(x)


#: Estimators used throughout the study, in plot/legend order.
ESTIMATORS: Dict[str, Callable[[np.ndarray], float]] = {
    "official_mean": official_mean,
    "median": median,
    "q25": lambda x: quantile(x, 0.25),
    "q10": lambda x: quantile(x, 0.10),
    "trim10": lambda x: trimmed_mean(x, 0.10),
    "trim20": lambda x: trimmed_mean(x, 0.20),
    "mad_floor": mad_noise_floor,
}

#: Short descriptions used in generated tables.
ESTIMATOR_DESCRIPTIONS = {
    "official_mean": "mean of region dB values (official calculate_global_noise)",
    "median": "median of region dB values",
    "q25": "25% quantile of region dB values",
    "q10": "10% quantile of region dB values",
    "trim10": "10%-tail trimmed mean of region dB values",
    "trim20": "20%-tail trimmed mean of region dB values",
    "mad_floor": "median + 1.4826*MAD (robust median+1sigma level)",
}


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------


def bootstrap_ci(
    x: np.ndarray,
    estimator: Callable[[np.ndarray], float],
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Tuple[float, float]:
    """Percentile bootstrap CI of ``estimator`` over resampled region cells."""
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=float).ravel()  # region may be 2-D
    n = x.size
    idx = rng.integers(0, n, size=(n_boot, n))
    samples = x[idx]
    stats = np.apply_along_axis(estimator, 1, samples)
    lo, hi = np.quantile(stats, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Synthetic peak injection (clearly marked as synthetic in all outputs)
# ---------------------------------------------------------------------------


def gaussian_peak_field(
    shape: Tuple[int, int],
    centers: Sequence[Tuple[int, int]],
    amp_db: float,
    sigma_rng: float = 2.0,
    sigma_vel: float = 2.0,
) -> np.ndarray:
    """Gaussian-shaped additive field in the dB domain (a multiplicative
    power contamination): ``peak(r, v) = amp * exp(-((r-r0)^2/(2 sr^2) +
    (v-v0)^2/(2 sv^2)))`` summed over the given centers (0-based bins)."""
    n_rng, n_vel = shape
    r = np.arange(n_rng)[:, None]
    v = np.arange(n_vel)[None, :]
    field = np.zeros((n_rng, n_vel), dtype=float)
    for r0, v0 in centers:
        field += amp_db * np.exp(
            -((r - r0) ** 2 / (2.0 * sigma_rng**2) + (v - v0) ** 2 / (2.0 * sigma_vel**2))
        )
    return field


def inject_gaussian_peak(
    rv_db: np.ndarray,
    centers: Sequence[Tuple[int, int]],
    amp_db: float,
    sigma_rng: float = 2.0,
    sigma_vel: float = 2.0,
) -> np.ndarray:
    """Return a *copy* of ``rv_db`` with synthetic Gaussian peaks added (dB)."""
    return rv_db + gaussian_peak_field(rv_db.shape, centers, amp_db, sigma_rng, sigma_vel)
