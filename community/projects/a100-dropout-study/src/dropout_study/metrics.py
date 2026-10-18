"""Spectral-level comparison metrics between a reference and a corrupted RV power map.

All metrics are computed on linear power maps ``P[r, d]`` (rows = range bins,
cols = Doppler bins, output of :func:`dropout_study.pipeline.rv_map_average`).
Peak positions are refined to sub-bin accuracy with a three-point parabolic
fit on the dB map, so "drift < 1 bin" is a meaningful continuous criterion.

Definitions (see README for the Chinese write-up of the same formulas):

* ``drift_range_bins`` / ``drift_doppler_bins``: refined peak position of the
  corrupted map minus that of the reference, per axis.
  ``drift_max_bins = max(|drift_range|, |drift_doppler|)``.
* ``peak_power_error_db = 10*log10(max(P_tst) / max(P_ref))``.
* ``total_energy_error_db = 10*log10(sum(P_tst) / sum(P_ref))``.
* ``spectral_correlation``: Pearson coefficient of the dB maps
  ``10*log10(P + 1e-30)`` (constant maps degenerate to 1.0 when identical,
  else 0.0).
* ``doppler_spurious_delta``: on each map's own dominant range bin, count the
  interior local maxima of the Doppler slice that exceed the slice maximum by
  less than 20 dB, excluding +-3 bins around the slice peak; report the
  corrupted count minus the reference count (so 0 means "as clean as the
  baseline").

Power is floored at 1e-30 before taking logarithms so an all-zero map yields
finite (very negative) dB values instead of -inf.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-30
_SPURIOUS_REL_DB = -20.0
_SPURIOUS_EXCLUDE = 3


def to_db(power: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(np.asarray(power, dtype=np.float64), _EPS))


def _parabolic_refine(y: np.ndarray, i: int) -> float:
    """Sub-bin position of the maximum near integer index ``i`` (parabolic fit)."""
    if 0 < i < len(y) - 1:
        a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
        den = a - 2.0 * b + c
        if den != 0.0:
            delta = 0.5 * (a - c) / den
            if abs(delta) <= 1.0:
                return i + delta
    return float(i)


def refined_peak(power: np.ndarray):
    """Refined ``(range_bin, doppler_bin)`` of the map maximum (sub-bin floats)."""
    p = to_db(power)
    ir, ic = np.unravel_index(int(np.argmax(p)), p.shape)
    r = _parabolic_refine(p[:, ic], ir)   # refine along the range axis
    d = _parabolic_refine(p[ir, :], ic)   # refine along the Doppler axis
    return float(r), float(d)


def peak_drift_bins(ref: np.ndarray, tst: np.ndarray) -> dict:
    rr, rd = refined_peak(ref)
    tr, td = refined_peak(tst)
    return {
        "drift_range_bins": tr - rr,
        "drift_doppler_bins": td - rd,
        "drift_max_bins": max(abs(tr - rr), abs(td - rd)),
    }


def peak_power_error_db(ref: np.ndarray, tst: np.ndarray) -> float:
    return float(to_db(np.asarray(tst).max()) - to_db(np.asarray(ref).max()))


def total_energy_error_db(ref: np.ndarray, tst: np.ndarray) -> float:
    return float(to_db(np.asarray(tst).sum()) - to_db(np.asarray(ref).sum()))


def spectral_correlation(ref: np.ndarray, tst: np.ndarray) -> float:
    a = to_db(ref).ravel()
    b = to_db(tst).ravel()
    sa, sb = a.std(), b.std()
    if sa == 0.0 or sb == 0.0:
        return 1.0 if np.allclose(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _count_spurious(slice_: np.ndarray) -> int:
    y = np.asarray(slice_, dtype=np.float64)
    n = y.size
    if n < 3:
        return 0
    peak = float(y.max())
    if peak <= 0.0:
        return 0
    thr = peak * 10.0 ** (_SPURIOUS_REL_DB / 10.0)
    imax = int(np.argmax(y))
    count = 0
    for i in range(1, n - 1):
        if (
            abs(i - imax) > _SPURIOUS_EXCLUDE
            and y[i] > thr
            and y[i] > y[i - 1]
            and y[i] >= y[i + 1]
        ):
            count += 1
    return count


def doppler_spurious_delta(ref: np.ndarray, tst: np.ndarray) -> int:
    r_ref = int(np.argmax(ref.max(axis=1)))
    r_tst = int(np.argmax(tst.max(axis=1)))
    return _count_spurious(tst[r_tst, :]) - _count_spurious(ref[r_ref, :])


def compare_maps(ref: np.ndarray, tst: np.ndarray) -> dict:
    """All five metrics in one dict (identity input -> exact zeros / one)."""
    out = peak_drift_bins(ref, tst)
    out["peak_power_error_db"] = peak_power_error_db(ref, tst)
    out["total_energy_error_db"] = total_energy_error_db(ref, tst)
    out["spectral_correlation"] = spectral_correlation(ref, tst)
    out["doppler_spurious_delta"] = doppler_spurious_delta(ref, tst)
    return out


METRIC_KEYS = [
    "drift_range_bins",
    "drift_doppler_bins",
    "drift_max_bins",
    "peak_power_error_db",
    "total_energy_error_db",
    "spectral_correlation",
    "doppler_spurious_delta",
]

__all__ = [
    "to_db", "refined_peak", "peak_drift_bins", "peak_power_error_db",
    "total_energy_error_db", "spectral_correlation", "doppler_spurious_delta",
    "compare_maps", "METRIC_KEYS",
]
