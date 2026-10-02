"""Metrics for dual-stream alignment: coverage, residuals, unmatched frames."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from stream_align.align import AlignResult
from stream_align.parser import StreamLog


def residual_stats(residuals_ms: list[int]) -> dict:
    """Median / p95 / max / mean of signed residuals; empty input -> n=0."""
    if not residuals_ms:
        return {"n": 0, "median_ms": None, "p95_ms": None, "max_ms": None,
                "mean_ms": None}
    arr = np.asarray(residuals_ms, dtype=np.int64)
    return {
        "n": int(arr.size),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "max_ms": int(np.max(np.abs(arr))),
        "mean_ms": float(np.mean(arr)),
    }


def unmatched_analysis(
    stream: StreamLog, result: AlignResult, other: StreamLog
) -> dict:
    """Analyse the frames of ``stream`` that no pair of ``result`` uses.

    Reports the unmatched count, where those frames sit inside the session
    (relative session progress in [0, 1]), and how far each unmatched frame is
    from the nearest frame of the *other* stream (nearest-neighbour gap in ms,
    computed with a KD-tree over the other stream's epochs).
    """
    if stream.kind == "raw":
        matched = {p[0] for p in result.pairs}
    else:
        matched = {p[1] for p in result.pairs}
    unmatched = [f for f in stream.frames if f.frame_idx not in matched]
    epochs_all = [f.epoch_ms for f in stream.frames if f.epoch_ms is not None]
    out = {
        "n_unmatched": len(unmatched),
        "n_stream": stream.n_frames,
        "rel_positions": [],
        "nn_gap_ms": [],
    }
    if unmatched and epochs_all:
        t0, t1 = min(epochs_all), max(epochs_all)
        span = t1 - t0 or 1
        out["rel_positions"] = [
            (f.epoch_ms - t0) / span
            for f in unmatched
            if f.epoch_ms is not None
        ]
    other_epochs = np.array(
        [f.epoch_ms for f in other.frames if f.epoch_ms is not None],
        dtype=np.float64,
    )
    if unmatched and other_epochs.size:
        tree = cKDTree(other_epochs.reshape(-1, 1))
        query = np.array(
            [f.epoch_ms for f in unmatched if f.epoch_ms is not None],
            dtype=np.float64,
        ).reshape(-1, 1)
        gaps, _ = tree.query(query, k=1)
        out["nn_gap_ms"] = [float(g) for g in gaps]
    return out


def summarize_unmatched(analysis: dict) -> dict:
    """Scalar summary of :func:`unmatched_analysis` for the CSV report."""
    nn = analysis["nn_gap_ms"]
    rel = analysis["rel_positions"]
    return {
        "n_unmatched": analysis["n_unmatched"],
        "unmatched_rel_mean": float(np.mean(rel)) if rel else None,
        "unmatched_nn_median_ms": float(np.median(nn)) if nn else None,
        "unmatched_nn_p95_ms": float(np.percentile(nn, 95)) if nn else None,
    }
