"""Frame-level decimation and sampling strategies for target logs.

Three independent samplers are provided:

1. :func:`uniform_keep_indices` / :func:`enumerate_uniform_phases`
   Deterministic keep-every-k decimation. For each k the *phase*
   ``phi in {0, ..., k-1}`` selects frames ``phi, phi+k, phi+2k, ...``; the
   complete phase set is exactly all k distinct residue classes, so enumerating
   it covers every deterministic outcome of the scheme.

2. :func:`stratified_keep_indices`
   Time-stratified random sampling: the recording interval ``[t0, t_end]`` is
   split into B equal-**duration** strata and ``per_stratum`` frames are drawn
   uniformly (without replacement) from the frames whose timestamps fall in
   each stratum. A stratum with too few candidates contributes what it has; an
   empty stratum (possible when a stratum is shorter than the longest frame
   gap) falls back to the nearest unused frame. The RNG is
   ``numpy.random.default_rng(seed)`` so a fixed seed makes the draw
   deterministic.

3. :func:`byte_budget_k`
   Byte-budget mode: given a target compression ratio C (original bytes /
   retained bytes ~ n / m), the smallest integer k with ``ceil(C)`` reaches the
   budget, because keep-every-k retains ``m ~ n/k`` whole frame blocks.
"""

from __future__ import annotations

import math

import numpy as np


def uniform_keep_indices(n: int, k: int, phase: int) -> np.ndarray:
    """Indices ``phase, phase+k, phase+2k, ...`` below ``n`` (keep-every-k)."""
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if not 0 <= phase < k:
        raise ValueError(f"phase must satisfy 0 <= phase < k, got phase={phase}, k={k}")
    return np.arange(phase, n, k, dtype=np.int64)


def enumerate_uniform_phases(n: int, k: int) -> dict:
    """All k phase outcomes of keep-every-k as ``{phase: indices}``.

    The union of the k index sets is exactly ``range(n)`` and the sets are
    pairwise disjoint (residue classes modulo k), which downstream tests rely
    on as the completeness property of the enumeration.
    """
    return {phase: uniform_keep_indices(n, k, phase) for phase in range(k)}


def stratified_keep_indices(
    times_ms, n_keep: int, seed: int, per_stratum: int = 1
) -> np.ndarray:
    """Duration-stratified random sample of frame indices.

    Parameters
    ----------
    times_ms : array-like
        Frame timestamps in ms (non-decreasing).
    n_keep : int
        Approximate number of frames to keep (B = ceil(n_keep / per_stratum)
        strata are created; deduplication may trim the result slightly).
    seed : int
        Seed for ``numpy.random.default_rng``; same seed -> same sample.
    per_stratum : int
        Frames drawn from each stratum (default 1).
    """
    times = np.asarray(times_ms, dtype=np.int64)
    n = len(times)
    if n_keep < 1 or n_keep > n:
        raise ValueError(f"n_keep must be in [1, {n}], got {n_keep}")

    n_strata = int(math.ceil(n_keep / per_stratum))
    edges = np.linspace(times[0], times[-1], n_strata + 1)
    rng = np.random.default_rng(seed)

    chosen: list[int] = []
    used: set[int] = set()
    for j in range(n_strata):
        lo, hi = edges[j], edges[j + 1]
        if j == n_strata - 1:
            cand = np.where((times >= lo) & (times <= hi))[0]
        else:
            cand = np.where((times >= lo) & (times < hi))[0]
        cand = np.array([c for c in cand if c not in used], dtype=np.int64)
        if cand.size == 0:
            mid = 0.5 * (lo + hi)
            remaining = np.array(
                [c for c in range(n) if c not in used], dtype=np.int64
            )
            if remaining.size == 0:
                break
            pick = [int(remaining[np.argmin(np.abs(times[remaining] - mid))])]
        else:
            take = min(per_stratum, cand.size)
            pick = rng.choice(cand, size=take, replace=False).tolist()
        chosen.extend(int(c) for c in pick)
        used.update(int(c) for c in pick)

    return np.array(sorted(chosen), dtype=np.int64)


def byte_budget_k(compression_ratio: float) -> int:
    """Smallest keep-every-k step whose compression reaches ``compression_ratio``.

    Retaining ``m ~ n/k`` of n frames shrinks the log by ~k, so the minimal
    integer k reaching a target ratio C is ``ceil(C)`` (clamped to >= 1).
    """
    c = float(compression_ratio)
    if not np.isfinite(c):
        raise ValueError(f"compression_ratio must be finite, got {compression_ratio}")
    return max(1, int(math.ceil(c - 1e-9)))
