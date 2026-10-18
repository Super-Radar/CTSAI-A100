"""Threshold sweeps and selection-bias metrics.

All metrics are retention-oriented: without ground truth we never claim
detection accuracy, we only quantify *how much* and *where* a threshold
removes observations, how the removed sets of different dimensions overlap,
and how the surviving distribution shifts relative to the unfiltered one.
"""

from __future__ import annotations

import numpy as np

from .filters import DIMENSION_META, FilterSpec, single_dim_spec

__all__ = [
    "retention",
    "stratified_retention",
    "sweep_dimension",
    "pairwise_jaccard",
    "joint_grid",
    "distribution_shift",
    "RANGE_BAND_EDGES",
    "ANGLE_SECTOR_EDGES",
    "SPEED_BIN_EDGES",
    "QUANTILE_LEVELS",
    "FIELDS",
]

# Stratification grids (edges are lower-inclusive, upper-exclusive).
RANGE_BAND_EDGES = (0.0, 50.0, 100.0, 150.0, 200.0, np.inf)
ANGLE_SECTOR_EDGES = (0.0, 20.0, 40.0, 60.0, np.inf)
SPEED_BIN_EDGES = (0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, np.inf)

QUANTILE_LEVELS = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)

FIELDS = {
    "range_m": ("range", "m"),
    "angle_deg": ("angle", "deg"),
    "snr_db": ("snr", "dB"),
    "speed_ms": ("speed", "m/s"),
}

_STRATA = {
    "range_band": lambda cloud: (cloud.range_m, RANGE_BAND_EDGES),
    "angle_sector": lambda cloud: (np.abs(cloud.angle_deg), ANGLE_SECTOR_EDGES),
    "speed_bin": lambda cloud: (np.abs(cloud.speed_ms), SPEED_BIN_EDGES),
}


def retention(keep: np.ndarray) -> float:
    """Overall retained fraction of a boolean kept mask."""
    n = int(keep.size)
    if n == 0:
        return float("nan")
    return float(np.count_nonzero(keep)) / n


def _fmt_edge(value: float) -> str:
    if np.isinf(value):
        return "+inf"
    return f"{value:g}"


def stratified_retention(cloud, keep: np.ndarray, stratum: str) -> list[dict]:
    """Per-band totals / kept counts for one stratification grid.

    ``stratum`` is one of ``range_band`` / ``angle_sector`` / ``speed_bin``.
    Band counts partition the cloud, so summing the returned ``n_total`` and
    ``n_kept`` columns reproduces the overall counts exactly.
    """
    values, edges = _STRATA[stratum](cloud)
    band = np.clip(np.searchsorted(edges, values, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b in range(len(edges) - 1):
        sel = band == b
        total = int(np.count_nonzero(sel))
        kept = int(np.count_nonzero(sel & keep))
        out.append(
            {
                "stratum": stratum,
                "band": f"[{_fmt_edge(edges[b])},{_fmt_edge(edges[b + 1])})",
                "n_total": total,
                "n_kept": kept,
                "retention": kept / total if total else float("nan"),
            }
        )
    return out


def sweep_dimension(cloud, dimension: str, values, strata=()) -> list[dict]:
    """Single-dimension threshold sweep with optional stratified rows.

    One row per threshold with ``stratum == "ALL"``, plus one row per band
    for every stratification name passed in ``strata``.
    """
    meta = DIMENSION_META[dimension]
    rows = []
    for value in values:
        keep = single_dim_spec(dimension, value).kept_mask(cloud)
        rows.append(
            {
                "dimension": dimension,
                "threshold": float(value),
                "unit": meta.unit,
                "stratum": "ALL",
                "band": "",
                "n_total": int(keep.size),
                "n_kept": int(np.count_nonzero(keep)),
                "retention": retention(keep),
            }
        )
        for name in strata:
            for rec in stratified_retention(cloud, keep, name):
                rows.append(
                    {
                        "dimension": dimension,
                        "threshold": float(value),
                        "unit": meta.unit,
                        **rec,
                    }
                )
    return rows


def deletion_mask(cloud, spec: FilterSpec) -> np.ndarray:
    """Complement of the kept mask: ``True`` marks a removed target."""
    return ~spec.kept_mask(cloud)


def pairwise_jaccard(deletions: dict) -> list[dict]:
    """Pairwise overlap metrics of per-dimension deletion sets.

    ``deletions`` maps dimension name -> boolean deletion mask. Three
    complementary views are reported per pair (A, B):

    * ``jaccard`` = ``|A & B| / |A | B|``: symmetric set similarity
      (identical -> 1.0, disjoint -> 0.0, two empty sets defined as 1.0);
    * ``containment_a`` = ``|A & B| / |A|``: fraction of A's deletions that
      are re-deleted by B (an empty A is contained in anything -> 1.0);
    * ``containment_b``: same with roles swapped.

    Jaccard hides one-sided absorption when the two deletion sets have very
    different sizes (a tiny set fully inside a huge one still has a small
    Jaccard), which is exactly the regime a wide near-zero-speed deletion
    creates -- hence the containment columns.
    """
    names = list(deletions)
    rows = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            da, db = deletions[a], deletions[b]
            inter = int(np.count_nonzero(da & db))
            union = int(np.count_nonzero(da | db))
            n_a = int(np.count_nonzero(da))
            n_b = int(np.count_nonzero(db))
            rows.append(
                {
                    "dim_a": a,
                    "dim_b": b,
                    "n_del_a": n_a,
                    "n_del_b": n_b,
                    "intersection": inter,
                    "union": union,
                    "jaccard": inter / union if union else 1.0,
                    "containment_a": inter / n_a if n_a else 1.0,
                    "containment_b": inter / n_b if n_b else 1.0,
                }
            )
    return rows


def joint_grid(cloud, dim_a: str, values_a, dim_b: str, values_b) -> list[dict]:
    """Two-dimensional joint threshold grid with an independence baseline.

    For every (a, b) pair the table reports the joint retention, the product
    of the two single-dimension retentions (the value expected if the two
    deletion sets overlapped exactly as independent selections would), and
    their difference ``interaction``:

    * ``interaction < 0`` -> the two thresholds delete a *shared* subset
      (double hit: one observation lost to both dimensions at once);
    * ``interaction > 0`` -> the deletions are more disjoint than
      independence would give (milder combined loss).
    """
    rows = []
    marginals_a = {
        float(v): retention(single_dim_spec(dim_a, v).kept_mask(cloud)) for v in values_a
    }
    marginals_b = {
        float(v): retention(single_dim_spec(dim_b, v).kept_mask(cloud)) for v in values_b
    }
    for va in values_a:
        for vb in values_b:
            keep = FilterSpec(
                **{DIMENSION_META[dim_a].spec_field: float(va),
                   DIMENSION_META[dim_b].spec_field: float(vb)}
            ).kept_mask(cloud)
            rj = retention(keep)
            rprod = marginals_a[float(va)] * marginals_b[float(vb)]
            rows.append(
                {
                    "dim_a": dim_a,
                    "value_a": float(va),
                    "dim_b": dim_b,
                    "value_b": float(vb),
                    "n_kept": int(np.count_nonzero(keep)),
                    "retention": rj,
                    "retention_indep": rprod,
                    "interaction": rj - rprod,
                }
            )
    return rows


def distribution_shift(cloud, spec: FilterSpec, quantiles=QUANTILE_LEVELS) -> list[dict]:
    """Per-field mean/quantile positions before vs. after filtering.

    Comparing quantiles of the same field across the unfiltered corpus and
    the survivor subset shows *where in the distribution* the removals are
    taken from -- the selection footprint left by the threshold.
    """
    keep = spec.kept_mask(cloud)
    rows = []
    for attr, (name, unit) in FIELDS.items():
        x = getattr(cloud, attr)
        xs = x[keep]
        entries = [("mean", float(x.mean()), float(xs.mean()))]
        for q in quantiles:
            entries.append(
                (f"q{int(round(q * 100)):02d}", float(np.quantile(x, q)), float(np.quantile(xs, q)))
            )
        for stat, before, after in entries:
            rows.append(
                {
                    "field": name,
                    "unit": unit,
                    "stat": stat,
                    "before": before,
                    "after": after,
                    "delta": after - before,
                }
            )
    return rows
