"""Deterministic single-server queue simulation for offline buffer budgeting.

Model (an *assumed* consumer, not the real bus):

* Arrivals are the log timestamps in order.  Messages sharing one timestamp
  form a batch and are enqueued atomically at that instant (the logger emits
  whole bursts under one timestamp).
* One server consumes one message every ``1 / rate`` seconds whenever the
  queue is non-empty (work-conserving, deterministic service time).
* ``capacity`` bounds the queue length measured in messages.  When a batch
  arrives and ``backlog + batch_size > capacity``, the excess messages are
  dropped immediately and counted as overflow (no reordering, no blocking).
* ``capacity=None`` means an unbounded queue (pure backlog study).

Useful exact property of this model: the smallest capacity that avoids any
overflow at a given rate is ``demand_peak`` — the maximum over arrival events
of ``backlog_just_before + batch_size`` — because drops happen exactly at the
arrival where that demand exceeds capacity, and never elsewhere.

The simulation is fully deterministic integer event arithmetic; no randomness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class QueueResult:
    """Outcome of one queue simulation."""

    rate: float  # messages/s
    capacity: float  # messages; math.inf when unbounded
    n_arrivals: int
    max_backlog: int  # peak queue length over the whole run
    final_backlog: int  # queue length at the last arrival
    overflow: int  # messages dropped due to full capacity
    completed: int  # messages fully consumed by the last arrival instant
    demand_peak: int  # max over events of backlog_before + batch_size


def group_arrivals(time_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Collapse a (sorted) timestamp sequence into ``(event_times, batch_sizes)``."""
    t = np.asarray(time_s, dtype=np.float64)
    if t.size == 0:
        raise ValueError("empty arrival sequence")
    t = np.sort(t)
    uniq, first = np.unique(t, return_index=True)
    sizes = np.diff(np.append(first, t.size))
    return uniq, sizes.astype(np.int64)


def simulate(
    time_s: np.ndarray,
    rate: float,
    capacity: int | None = None,
    eps: float = 1e-9,
) -> QueueResult:
    """Run the deterministic single-server queue over the arrival sequence.

    ``eps`` guards the completion-time flooring against float round-off
    (e.g. ``floor(4 * 0.5)`` must be 4, not 3).
    """
    if rate <= 0:
        raise ValueError("rate must be > 0")
    if capacity is not None and int(capacity) < 0:
        raise ValueError("capacity must be >= 0")

    events, sizes = group_arrivals(time_s)
    rate = float(rate)
    cap = math.inf if capacity is None else int(capacity)

    backlog = 0
    busy_since = None  # time the server last started on non-empty queue
    done_ref = 0  # completions already applied since busy_since
    completed = 0
    overflow = 0
    max_backlog = 0
    demand_peak = 0

    for t, k in zip(events.tolist(), sizes.tolist()):
        # -- service between the previous event and now -------------------
        if backlog > 0:
            completions_now = math.floor((t - busy_since) * rate + eps)
            delta = min(completions_now - done_ref, backlog)
            backlog -= delta
            completed += delta
            if backlog == 0:
                busy_since = None
                done_ref = 0
            else:
                done_ref = completions_now

        # -- batch arrival (atomic enqueue of the whole same-ts batch) ----
        demand = backlog + k
        if demand > demand_peak:
            demand_peak = demand
        if cap is not math.inf and demand > cap:
            take = max(int(cap) - backlog, 0)
            overflow += k - take
            backlog += take
        else:
            backlog = demand
        if backlog > max_backlog:
            max_backlog = backlog
        if backlog > 0 and busy_since is None:
            busy_since = t
            done_ref = 0

    return QueueResult(
        rate=rate,
        capacity=cap,
        n_arrivals=int(events.size),
        max_backlog=int(max_backlog),
        final_backlog=int(backlog),
        overflow=int(overflow),
        completed=int(completed),
        demand_peak=int(demand_peak),
    )


def sweep_backlog_vs_rate(
    time_s: np.ndarray,
    rates: list[float] | tuple[float, ...],
    capacity: int | None = None,
) -> list[QueueResult]:
    """Queue results for each rate (same arrivals, capacity held fixed)."""
    return [simulate(time_s, r, capacity) for r in rates]


def sweep_capacity(
    time_s: np.ndarray,
    rate_grid: list[float] | tuple[float, ...],
    capacity_grid: list[int] | tuple[int, ...],
) -> list[QueueResult]:
    """Queue results over a rate x capacity grid (row-major: rate outer)."""
    return [
        simulate(time_s, r, c)
        for r in rate_grid
        for c in capacity_grid
    ]


def required_capacity(time_s: np.ndarray, rate: float) -> int:
    """Smallest queue capacity with zero overflow at this rate (exact).

    Equal to ``demand_peak`` of the unbounded run; see module docstring.
    """
    return simulate(time_s, rate, capacity=None).demand_peak


def min_rate_no_overflow(
    results: list[QueueResult],
) -> float | None:
    """Smallest rate with zero overflow among results sorted by ascending rate."""
    for res in sorted(results, key=lambda x: x.rate):
        if res.overflow == 0:
            return res.rate
    return None
