"""Per-frame statistic series for effective-sample-size analysis.

Two families of frame-level statistics are studied:

* target count ``n_t`` -- number of detected targets in frame ``t``;
* grid-cell occupancy ``o_t`` -- indicator that a fixed polar-grid cell contains
  at least one detection in frame ``t``.

The polar grid is fixed over the whole record: range bins of ``dr = 5 m``
covering ``[0, 5 * ceil(r_max / 5))`` m (``r_max`` read from the record itself,
deterministic) and azimuth bins of ``dtheta = 10 deg`` covering
``[-180 deg, 180 deg)``.

Representative cells follow a fixed, documented rule: among cells occupied in
at least ``MIN_HITS`` frames with occupancy fraction inside ``(0.05, 0.95)``,
the two cells whose fraction is closest to 0.5 (ties broken by cell index).
If fewer than one cell passes, the band relaxes to ``(0.02, 0.98)``.
"""

from __future__ import annotations

import numpy as np

from .parser import ParseResult

R_STEP_M = 5.0
THETA_STEP_DEG = 10.0
N_THETA = int(round(360.0 / THETA_STEP_DEG))  # 36 bins over [-180, 180)
MIN_HITS = 30


def target_count_series(res: ParseResult) -> np.ndarray:
    """Per-frame target count n_t as a float series."""
    return res.target_counts().astype(np.float64)


def frame_times_s(res: ParseResult) -> np.ndarray:
    """Time axis in seconds since the first frame (epoch ms if available)."""
    ep = res.epoch_ms()
    if ep.size and np.all(ep > 0) and np.all(np.diff(ep) >= 0):
        return (ep - ep[0]) / 1000.0
    return np.arange(res.n_frames, dtype=np.float64)


def occupancy_grid(res: ParseResult, r_step: float = R_STEP_M, theta_step: float = THETA_STEP_DEG):
    """Build the frame x cell occupancy matrix of the fixed polar grid.

    Returns ``(occ, info)`` where ``occ`` is uint8 of shape
    ``(n_frames, n_r * n_theta)`` and ``info`` documents the grid geometry.
    """
    n_t = int(round(360.0 / theta_step))
    r_all = (
        np.concatenate([f.rng for f in res.frames]) if res.frames else np.array([], dtype=np.float64)
    )
    r_max = float(r_all.max()) if r_all.size else 0.0
    n_r = max(1, int(np.ceil(r_max / r_step)))
    occ = np.zeros((res.n_frames, n_r * n_t), dtype=np.uint8)
    n_out = 0
    for i, f in enumerate(res.frames):
        if f.n_targets == 0:
            continue
        rb = np.floor(f.rng / r_step).astype(np.int64)
        tb = np.floor((f.angle + 180.0) / theta_step).astype(np.int64)
        ok = (rb >= 0) & (rb < n_r) & (tb >= 0) & (tb < n_t)
        n_out += int((~ok).sum())
        cells = rb[ok] * n_t + tb[ok]
        occ[i, np.unique(cells)] = 1
    info = {
        "r_step_m": float(r_step),
        "theta_step_deg": float(theta_step),
        "n_r": int(n_r),
        "n_theta": int(n_t),
        "n_cells": int(occ.shape[1]),
        "r_max_observed_m": r_max,
        "r_grid_max_m": float(n_r * r_step),
        "n_detections_out_of_grid": n_out,
    }
    return occ, info


def occupancy_fractions(occ: np.ndarray) -> np.ndarray:
    return occ.mean(axis=0)


def select_representative_cells(occ: np.ndarray, min_hits: int = MIN_HITS):
    """Pick up to two representative cells by the documented occupancy rule."""
    hits = occ.sum(axis=0)
    frac = occupancy_fractions(occ)
    for lo, hi in ((0.05, 0.95), (0.02, 0.98)):
        valid = np.where((hits >= min_hits) & (frac > lo) & (frac < hi))[0]
        if valid.size:
            order = valid[np.argsort(np.abs(frac[valid] - 0.5), kind="stable")]
            chosen = [int(c) for c in order[:2]]
            return chosen, {
                "band": (float(lo), float(hi)),
                "n_valid": int(valid.size),
                "n_frames": int(occ.shape[0]),
                "fraction": {c: float(frac[c]) for c in chosen},
                "hits": {c: int(hits[c]) for c in chosen},
            }
    return [], {"band": None, "n_valid": 0, "n_frames": int(occ.shape[0]),
                "fraction": {}, "hits": {}}


def cell_center(cell: int, grid_info: dict) -> tuple[float, float]:
    """Center coordinates (r [m], theta [deg]) of a linear cell index."""
    n_t = grid_info["n_theta"]
    rb, tb = divmod(int(cell), n_t)
    r_c = (rb + 0.5) * grid_info["r_step_m"]
    th_c = -180.0 + (tb + 0.5) * grid_info["theta_step_deg"]
    return r_c, th_c


def occupancy_series(occ: np.ndarray, cell: int) -> np.ndarray:
    """Binary occupancy indicator o_t of one grid cell as a float series."""
    return occ[:, int(cell)].astype(np.float64)
