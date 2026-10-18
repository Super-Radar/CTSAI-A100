"""Time-weighted fidelity metrics for decimated target-frame sequences.

Semantic conventions (fixed for this project and asserted by the tests)
-----------------------------------------------------------------------
**Frame exposure weights.** Frame i "owns" the time interval between the
midpoints of its neighbouring frame gaps (the first/last frames own up to the
session boundary), so ``w_i`` sums exactly to ``t_last - t_0``.

**Weight conservation under sampling.** When only a subset of frames is
retained, the exposure of every dropped frame is merged into the *nearest
retained frame* (index-space Voronoi, ties to the earlier retained frame).
The retained frames therefore still carry the **full recording duration**:
``sum(merged_weights) == t_last - t_0`` for every sampler, which is what the
downstream time-weighted histograms normalise against.

**Time-weighted occupancy.** Every target in frame i contributes ``w_i`` to
its (range, angle) polar bin. Normalising gives the fraction of time-exposure
spent in each cell, so dense and sparse stretches of the log contribute
proportional to how long they actually lasted, not to how many frames they
happened to be split into.

**Short-lived response.** A cell's *observed duration* is the summed exposure
of frames that observe it. A cell with ``0 < duration < T`` (default T = 1 s)
is "short-lived"; after sampling it is *retained* iff at least one retained
frame observes it.

Distances use the total-variation convention ``L1 = 0.5 * sum |p - q|`` in
[0, 1] and the plain Euclidean ``L2``.
"""

from __future__ import annotations

import numpy as np

from .parser import Frame


# ---------------------------------------------------------------------------
# time weights
# ---------------------------------------------------------------------------
def frame_time_weights(times_ms) -> np.ndarray:
    """Midpoint-ownership exposure weights [ms]; ``sum == t_last - t_0``."""
    t = np.asarray(times_ms, dtype=np.float64)
    if t.ndim != 1 or len(t) < 1:
        raise ValueError("times_ms must be a 1-D array with at least one element")
    if len(t) == 1:
        return np.zeros(1)
    if np.any(np.diff(t) < 0):
        raise ValueError("times_ms must be non-decreasing")
    mids = 0.5 * (t[:-1] + t[1:])
    edges = np.concatenate(([t[0]], mids, [t[-1]]))
    return np.diff(edges)


def merged_time_weights(times_ms, keep_idx) -> np.ndarray:
    """Exposure weights of retained frames after merging dropped exposure.

    Every original frame is assigned to the nearest retained frame in index
    space (ties to the earlier retained frame) and its exposure weight is
    added there, so ``sum(result) == t_last - t_0`` regardless of the sampler.
    """
    w = frame_time_weights(times_ms)
    s = np.asarray(keep_idx, dtype=np.int64)
    s = np.sort(np.unique(s))
    n = len(w)
    if s.size == 0 or s[0] < 0 or s[-1] >= n:
        raise ValueError("keep_idx must be a non-empty subset of frame indices")
    if s.size == 1:
        return np.array([w.sum()])
    cuts = 0.5 * (s[:-1] + s[1:])
    assign = np.searchsorted(cuts, np.arange(n), side="left")
    return np.bincount(assign, weights=w, minlength=s.size)


# ---------------------------------------------------------------------------
# polar cell grid: time-weighted occupancy histogram + exposure
# ---------------------------------------------------------------------------
class CellGrid:
    """Polar histogram grid over (range, angle) with delta-r x delta-theta bins."""

    def __init__(self, dr: float, dtheta: float, r_max: float,
                 r_min: float = 0.0, theta_min: float = -90.0, theta_max: float = 90.0):
        if dr <= 0 or dtheta <= 0:
            raise ValueError("bin sizes must be positive")
        self.dr = float(dr)
        self.dtheta = float(dtheta)
        self.r_edges = np.arange(r_min, r_max + dr, dr)
        self.theta_edges = np.arange(theta_min, theta_max + dtheta, dtheta)
        self.shape = (len(self.r_edges) - 1, len(self.theta_edges) - 1)

    # -- internals ----------------------------------------------------------
    def _flat_ids(self, targets: np.ndarray) -> np.ndarray:
        """Flat cell id per target row; -1 when outside the grid."""
        r = targets[:, 0]
        th = targets[:, 1]
        ir = np.floor((r - self.r_edges[0]) / self.dr).astype(np.int64)
        it = np.floor((th - self.theta_edges[0]) / self.dtheta).astype(np.int64)
        inside = (
            (ir >= 0) & (ir < self.shape[0]) & (it >= 0) & (it < self.shape[1])
        )
        flat = np.where(inside, ir * self.shape[1] + it, -1)
        # clip exact right-edge values into the last bin
        on_r_edge = (r == self.r_edges[-1]) & (th >= self.theta_edges[0]) & (th <= self.theta_edges[-1])
        if np.any(on_r_edge & (ir == self.shape[0])):
            flat = np.where(on_r_edge & (ir == self.shape[0]),
                            (self.shape[0] - 1) * self.shape[1] + np.clip(it, 0, self.shape[1] - 1),
                            flat)
        return flat

    # -- public API ----------------------------------------------------------
    def histogram(self, frames, weights) -> tuple:
        """Time-weighted occupancy counts over the grid.

        Each target of frame i contributes ``weights[i]``. Returns
        ``(density, counts)`` where ``density`` sums to 1 (or all zeros when
        nothing falls inside the grid).
        """
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(frames):
            raise ValueError("weights length must match number of frames")
        ids: list = []
        ws: list = []
        for j, fr in enumerate(frames):
            if fr.targets.shape[0]:
                fid = self._flat_ids(fr.targets)
                ids.append(fid[fid >= 0])
                ws.append(np.full(int((fid >= 0).sum()), w[j]))
        if ids:
            all_ids = np.concatenate(ids)
            all_w = np.concatenate(ws)
        else:
            all_ids = np.zeros(0, dtype=np.int64)
            all_w = np.zeros(0)
        counts = np.bincount(
            all_ids, weights=all_w, minlength=self.shape[0] * self.shape[1]
        ).reshape(self.shape)
        total = counts.sum()
        density = counts / total if total > 0 else counts
        return density, counts

    def exposure(self, frames, weights) -> np.ndarray:
        """Per-cell observed duration [same unit as weights].

        Frame i adds ``weights[i]`` to every cell it observes (presence, not
        multiplicity: multiple targets in one cell count once).
        """
        w = np.asarray(weights, dtype=np.float64)
        if len(w) != len(frames):
            raise ValueError("weights length must match number of frames")
        acc = np.zeros(self.shape, dtype=np.float64)
        for j, fr in enumerate(frames):
            if fr.targets.shape[0] and w[j] > 0:
                fid = self._flat_ids(fr.targets)
                fid = np.unique(fid[fid >= 0])
                if fid.size:
                    flat_acc = np.zeros(self.shape[0] * self.shape[1])
                    np.add.at(flat_acc, fid, w[j])
                    acc += flat_acc.reshape(self.shape)
        return acc

    def cell_active_frames(self, frames, cell_rc) -> np.ndarray:
        """Indices of frames observing the (row, col) cell."""
        row, col = cell_rc
        want = row * self.shape[1] + col
        out = []
        for j, fr in enumerate(frames):
            if fr.targets.shape[0]:
                fid = self._flat_ids(fr.targets)
                if np.any(fid == want):
                    out.append(j)
        return np.array(out, dtype=np.int64)


# ---------------------------------------------------------------------------
# distances and statistics
# ---------------------------------------------------------------------------
def tv_l1(p, q) -> float:
    """Total-variation distance ``0.5 * sum |p - q|`` in [0, 1]."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(0.5 * np.abs(p - q).sum())


def l2_distance(p, q) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    return float(np.sqrt(np.sum((p - q) ** 2)))


def count_statistics(frames) -> dict:
    """Mean / median / p90 of the per-frame target count."""
    c = np.array([fr.n_targets for fr in frames], dtype=np.float64)
    return {
        "mean": float(c.mean()),
        "p50": float(np.percentile(c, 50)),
        "p90": float(np.percentile(c, 90)),
    }


def count_stat_errors(frames, keep_idx) -> dict:
    """Errors of per-frame count statistics after keeping ``keep_idx``."""
    full = count_statistics(frames)
    kept = [frames[j] for j in np.asarray(keep_idx, dtype=np.int64)]
    part = count_statistics(kept)
    return {
        "mean_abs_err": abs(full["mean"] - part["mean"]),
        "mean_rel_err": abs(full["mean"] - part["mean"]) / full["mean"]
        if full["mean"] > 0
        else 0.0,
        "p50_abs_err": abs(full["p50"] - part["p50"]),
        "p90_abs_err": abs(full["p90"] - part["p90"]),
    }


def short_response_metrics(full_exposure_s, sampled_exposure_s, T: float) -> dict:
    """Retention of short-lived cells after sampling.

    Parameters
    ----------
    full_exposure_s, sampled_exposure_s : 2-D arrays
        Per-cell observed durations [s] before / after sampling (from
        :meth:`CellGrid.exposure` with weights in seconds).
    T : float
        Short-lived threshold [s].
    """
    full = np.asarray(full_exposure_s)
    samp = np.asarray(sampled_exposure_s)
    short_mask = (full > 0) & (full < T)
    n_short = int(short_mask.sum())
    retained = int(((samp > 0) & short_mask).sum())
    return {
        "T_s": float(T),
        "n_short": n_short,
        "n_retained": retained,
        "n_missed": n_short - retained,
        "retention_rate": (retained / n_short) if n_short > 0 else 1.0,
        "short_mask": short_mask,
    }


# ---------------------------------------------------------------------------
# synthetic benchmarks
# ---------------------------------------------------------------------------
def _mk_frame(index, epoch_ms, targets, frame_nb=None):
    return Frame(
        index=index,
        frame_nb=frame_nb if frame_nb is not None else index,
        epoch_ms=int(epoch_ms),
        wall_time="",
        targets=np.asarray(targets, dtype=np.float64).reshape(-1, 3),
    )


def synthetic_pulse_session(
    n_frames: int = 300,
    dt_ms: int = 50,
    background=((5.25, -10.0), (8.25, 3.0), (12.25, 25.0), (20.25, -40.0)),
    pulses=((37, 3, (6.25, 15.0)), (120, 2, (9.75, -5.0))),
    snr: float = 30.0,
    jitter_m: float = 0.02,
    seed: int = 0,
):
    """Frame list with persistent background cells plus short pulse responses.

    ``pulses`` entries are ``(start_frame, length_frames, (r, theta))``: the
    response is present in frames ``[start, start+length)`` only, so its whole
    active window can be provably missed by some keep-every-k phases (see
    :func:`analytic_missing_phases`). Range coordinates sit at 0.5 m bin
    centers so the +/-0.02 m jitter never straddles a CellGrid bin edge.
    Returns ``(frames, times_ms)``.
    """
    rng = np.random.default_rng(seed)
    t0 = 1_700_000_000_000
    times = t0 + np.arange(n_frames, dtype=np.int64) * dt_ms
    frames = []
    for i in range(n_frames):
        tg = [[r + rng.uniform(-jitter_m, jitter_m), a, snr] for (r, a) in background]
        for start, length, (pr, pa) in pulses:
            if start <= i < start + length:
                tg.append([pr + rng.uniform(-jitter_m, jitter_m), pa, snr])
        frames.append(_mk_frame(i, times[i], tg))
    return frames, times


def analytic_missing_phases(start: int, length: int, k: int) -> list:
    """Keep-every-k phases that provably miss the frame interval entirely.

    With retained indices ``phi (mod k)``, the first retained index at or
    after ``start`` is ``start + ((phi - start) mod k)``; the interval
    ``[start, start+length-1]`` is completely missed iff that offset exceeds
    ``length - 1``. Phases are returned in increasing order.
    """
    if length < 1:
        raise ValueError(f"length must be >= 1, got {length}")
    return [phi for phi in range(k) if ((phi - start) % k) > length - 1]


def synthetic_irregular_session(
    sparse_dt_ms: int = 1000,
    dense_dt_ms: int = 100,
    n_sparse: int = 21,
    n_dense: int = 201,
    cell_a: tuple = (6.0, 10.0),
    cell_b: tuple = (15.0, -20.0),
    snr: float = 30.0,
):
    """Two equally-long halves with very different frame density.

    Cell A is observed only during the sparse (long-gap, dt=``sparse_dt_ms``)
    half and cell B only during the dense (short-gap, dt=``dense_dt_ms``)
    half; both halves span the same wall-clock duration. Under time weighting
    the two cells get (nearly) equal occupancy; under frame-count weighting
    cell B is over-weighted by ~ the density ratio. Returns
    ``(frames, times_ms)``.
    """
    t0 = 1_700_000_000_000
    t_sparse = t0 + np.arange(n_sparse, dtype=np.int64) * sparse_dt_ms
    t_dense = t_sparse[-1] + np.arange(1, n_dense + 1, dtype=np.int64) * dense_dt_ms
    times = np.concatenate([t_sparse, t_dense])
    frames = []
    for i, t in enumerate(times):
        if i < n_sparse:
            tg = [cell_a]
        else:
            tg = [cell_b]
        frames.append(_mk_frame(i, t, [[tg[0][0], tg[0][1], snr]]))
    return frames, times
