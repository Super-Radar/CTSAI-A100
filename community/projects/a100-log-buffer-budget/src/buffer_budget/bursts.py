"""Multi-scale arrival-rate burstiness metrics.

Everything here is offline counting on the parsed timestamp sequence:

* sliding-window message and byte arrival rates at several time scales
  (10 ms ... 1 s) and the per-scale peak ("envelope"),
* burst factor = peak window rate / mean rate,
* coefficient of variation of the 1 s window-rate series,
* per-identifier share of traffic,
* same-timestamp batch statistics (the host logger emits whole bursts of
  messages that share one timestamp).

No stochastic model is involved: identical inputs give identical outputs.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

#: Analysis window scales in seconds (10 ms, 50 ms, 100 ms, 500 ms, 1 s).
WINDOW_S: tuple[float, ...] = (0.01, 0.05, 0.10, 0.50, 1.00)


def _sorted_inputs(time_s: np.ndarray, n_bytes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    t = np.asarray(time_s, dtype=np.float64)
    b = np.asarray(n_bytes, dtype=np.int64)
    if t.size == 0:
        raise ValueError("empty arrival sequence")
    if b.size != t.size:
        raise ValueError("time_s and n_bytes must have equal length")
    order = np.argsort(t, kind="stable")  # defensive; ASC logs are time-ordered
    return t[order], b[order]


def sliding_window_rates(
    time_s: np.ndarray,
    n_bytes: np.ndarray,
    window_s: float,
    hop_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sliding-window arrival rates.

    Windows are half-open intervals ``[t, t + window_s)`` starting every
    ``hop_s`` from the first timestamp; windows extending past the last
    timestamp are dropped so the peak is never diluted by a partial window.
    When the whole log is shorter than one window, a single window anchored
    at the first timestamp is used.

    Returns ``(center_s, msg_rate, byte_rate)`` with rates in messages/s and
    bytes/s.
    """
    if window_s <= 0 or hop_s <= 0:
        raise ValueError("window_s and hop_s must be positive")
    t, b = _sorted_inputs(time_s, n_bytes)
    t0, t1 = float(t[0]), float(t[-1])

    byte_cum = np.concatenate(([0], np.cumsum(b)))  # len = n + 1

    starts = t0 + np.arange(int(np.floor((t1 - t0) / hop_s)) + 1) * hop_s
    starts = starts[starts + window_s <= t1 + 1e-9]
    if starts.size == 0:
        starts = np.array([t0])

    left = np.searchsorted(t, starts, side="left")
    right = np.searchsorted(t, starts + window_s, side="left")
    msg_rate = (right - left) / window_s
    byte_rate = (byte_cum[right] - byte_cum[left]) / window_s
    centers = starts + window_s / 2.0
    return centers, msg_rate, byte_rate


def window_scale_stats(
    time_s: np.ndarray,
    n_bytes: np.ndarray,
    windows: tuple[float, ...] = WINDOW_S,
    hop_divisor: int = 10,
) -> dict[float, dict[str, float]]:
    """Per-scale envelope statistics (hop = window / hop_divisor).

    Returns ``{window_s: {peak_msg_s, peak_bytes_s, n_windows}}``.
    """
    out: dict[float, dict[str, float]] = {}
    for w in windows:
        _, mr, br = sliding_window_rates(time_s, n_bytes, w, w / hop_divisor)
        out[w] = {
            "peak_msg_s": float(mr.max()),
            "peak_bytes_s": float(br.max()),
            "n_windows": int(mr.size),
        }
    return out


def mean_rates(time_s: np.ndarray, n_bytes: np.ndarray) -> tuple[float, float, float]:
    """Whole-log mean message rate, mean byte rate and duration in seconds."""
    t, b = _sorted_inputs(time_s, n_bytes)
    duration = float(t[-1] - t[0])
    if duration <= 0:
        return float("nan"), float("nan"), duration
    return (
        float(t.size) / duration,
        float(b.sum()) / duration,
        duration,
    )


def cv_1s(time_s: np.ndarray, n_bytes: np.ndarray) -> float:
    """Coefficient of variation of the 1 s window-rate series (dispersion of load)."""
    _, mr, _ = sliding_window_rates(time_s, n_bytes, 1.0, 0.1)
    if mr.size == 0 or float(np.mean(mr)) == 0.0:
        return float("nan")
    return float(stats.variation(mr))


def burst_factor(peak_msg_s: float, mean_msg_s: float) -> float:
    """Peak window rate divided by whole-log mean rate (>= 1, larger = burstier)."""
    if not (mean_msg_s > 0):
        return float("nan")
    return float(peak_msg_s) / float(mean_msg_s)


def per_id_counts(can_id: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Traffic share per identifier.

    Returns ``(ids_desc, counts_desc, fractions_desc)`` sorted by descending
    count; ``fractions`` sums to exactly 1.
    """
    ids, counts = np.unique(np.asarray(can_id, dtype=np.int64), return_counts=True)
    order = np.argsort(-counts, kind="stable")
    ids = ids[order]
    counts = counts[order]
    fractions = counts.astype(np.float64) / float(counts.sum())
    return ids, counts.astype(np.int64), fractions


def batch_stats(time_s: np.ndarray) -> tuple[int, int, float]:
    """Same-timestamp batch statistics: ``(n_batches, max_batch, mean_batch)``.

    A batch is a maximal group of messages sharing one timestamp; the logger
    emits bursts this way.
    """
    t = np.asarray(time_s, dtype=np.float64)
    if t.size == 0:
        raise ValueError("empty arrival sequence")
    t = np.sort(t)
    change = np.empty(t.size, dtype=bool)
    change[0] = True
    change[1:] = t[1:] != t[:-1]
    n_batches = int(change.sum())
    starts = np.flatnonzero(change)
    sizes = np.diff(np.append(starts, t.size))
    return n_batches, int(sizes.max()), float(sizes.mean())
