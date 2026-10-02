"""Geometric-closure residuals: how well does each hypothesis reproduce the
logged X/Y columns?

Residual of a single row = planar Euclidean distance between the logged point
and the point predicted from ``(range, angle)`` under a hypothesis::

    d = sqrt((X_log - X_pred)^2 + (Y_log - Y_pred)^2)

X- and Y-component residuals are reported separately as well.  Per file the
study ranks the four hypotheses by the *median* combined residual, reports the
p95/max of the winner and the separability margin
(``second_best_median - best_median``).  If the logged X/Y columns carry no
usable signal (all ~zero rows) the verdict is ``ABSTAIN`` instead of a
hypothesis name.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from .hypotheses import ANGLE_OFFSET_GRID, HYPOTHESES, HYPOTHESIS_ORDER, predict

__all__ = [
    "row_residuals",
    "residual_stats",
    "is_identifiable",
    "evaluate_file",
    "ABSTAIN",
]

ABSTAIN = "ABSTAIN"

# A logged |X| (or |Y|) below this threshold [m] is treated as "no signal"
# (RawTarget logs carry float-junk like 1.7e-31 in the X column).
SIGNAL_EPS_M = 1e-6

# Minimum fraction of rows with a non-degenerate logged position for the
# file-level verdict to be trusted.
MIN_SIGNAL_FRACTION = 0.5


def row_residuals(r, angle_deg, x, y, hyp: str, angle_offset_deg: float = 0.0) -> Dict[str, np.ndarray]:
    """Component and combined residuals for one hypothesis."""
    px, py = predict(hyp, r, angle_deg, angle_offset_deg)
    dx = np.asarray(x, dtype=float) - px
    dy = np.asarray(y, dtype=float) - py
    dist = np.hypot(dx, dy)
    return {"dx": dx, "dy": dy, "dist": dist}


def residual_stats(res: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Median/p95/max of the combined residual plus component medians [m]."""
    dist = np.asarray(res["dist"], dtype=float)
    return {
        "median_m": float(np.median(dist)),
        "p95_m": float(np.percentile(dist, 95)),
        "max_m": float(np.max(dist)),
        "median_dx_m": float(np.median(np.abs(res["dx"]))),
        "median_dy_m": float(np.median(np.abs(res["dy"]))),
    }


def is_identifiable(x, y, min_fraction: float = MIN_SIGNAL_FRACTION,
                    eps: float = SIGNAL_EPS_M) -> bool:
    """True when the logged X/Y columns carry a usable non-degenerate signal."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.any(ok):
        return False
    signal = np.hypot(x[ok], y[ok]) > eps
    return float(np.mean(signal)) >= min_fraction


def scan_angle_offset(r, angle_deg, x, y, hyp: str,
                      grid: Optional[np.ndarray] = None):
    """Sweep the angle zero-point offset; return (best_offset_deg, medians).

    ``medians`` is aligned with the swept grid so callers can see the whole
    profile; ``best_offset_deg`` is the grid point with the smallest median
    combined residual (ties resolved toward 0 deg by argmin order when the
    grid starts below zero; the grid here is symmetric so 0.0 wins ties).
    """
    if grid is None:
        grid = ANGLE_OFFSET_GRID
    medians = np.array([
        float(np.median(row_residuals(r, angle_deg, x, y, hyp, float(off))["dist"]))
        for off in grid
    ])
    best_i = int(np.argmin(medians))
    return float(grid[best_i]), medians


def evaluate_file(r, angle_deg, x, y, scan_offsets: bool = True) -> Dict[str, object]:
    """Full per-file closure evaluation.

    Returns a dict with per-hypothesis stats (at offset 0), the best and
    second-best hypothesis, the margin, an identifiability flag and the
    optional angle-offset scan of the winning hypothesis.
    """
    r = np.asarray(r, dtype=float)
    angle_deg = np.asarray(angle_deg, dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    identifiable = is_identifiable(x, y)

    hyp_stats: Dict[str, Dict[str, float]] = {}
    for name in HYPOTHESIS_ORDER:
        res = row_residuals(r, angle_deg, x, y, name, 0.0)
        hyp_stats[name] = residual_stats(res)

    ranked = sorted(HYPOTHESIS_ORDER, key=lambda n: hyp_stats[n]["median_m"])
    best, second = ranked[0], ranked[1]
    best_stats = hyp_stats[best]
    margin = hyp_stats[second]["median_m"] - best_stats["median_m"]

    result: Dict[str, object] = {
        "n_rows": int(len(r)),
        "identifiable": bool(identifiable),
        "best_hyp": best if identifiable else ABSTAIN,
        "second_hyp": second,
        "margin_m": float(margin),
        "best_stats": best_stats,
        "hyp_stats": hyp_stats,
        "angle_offset_best_deg": 0.0,
        "angle_offset_improves": False,
        "angle_offset_improvement_m": 0.0,
    }

    if scan_offsets and identifiable:
        best_off, medians = scan_angle_offset(r, angle_deg, x, y, best)
        base = hyp_stats[best]["median_m"]
        improvement = base - float(np.min(medians))
        result["angle_offset_best_deg"] = best_off
        result["angle_offset_improvement_m"] = float(improvement)
        result["angle_offset_improves"] = bool(improvement > 1e-9)

    return result
