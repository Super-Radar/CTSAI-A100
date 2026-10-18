"""Independently togglable threshold filter operators.

Each operator mirrors one filtering dimension of the official
``radarViewer.py`` ``RadarFilter`` concept (range / angle / speed / SNR
thresholds; read-only conceptual reference, the official tool is not
modified or executed here). Every dimension is switched on by setting its
threshold and switched off by leaving it ``None``, so selection effects of
each dimension and of any combination can be studied in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["DIMENSIONS", "DIMENSION_META", "FilterSpec", "single_dim_spec"]


@dataclass(frozen=True)
class DimensionMeta:
    """Human-facing metadata of one filter dimension."""

    spec_field: str  # FilterSpec keyword
    label: str  # short English label (figures / CSVs)
    unit: str  # physical unit
    description: str  # keep-rule in words


DIMENSION_META = {
    "range_max": DimensionMeta("range_max", "range max", "m",
                               "keep range <= R_max"),
    "angle_max": DimensionMeta("angle_max", "|angle| max", "deg",
                               "keep |angle| <= A_max"),
    "speed_min": DimensionMeta("speed_min", "|speed| min", "m/s",
                               "keep |speed| >= v_min (near-zero / static removal)"),
    "speed_max": DimensionMeta("speed_max", "|speed| max", "m/s",
                               "keep |speed| <= v_max"),
    "snr_min": DimensionMeta("snr_min", "snr min", "dB",
                             "keep snr >= s_min"),
}
DIMENSIONS = tuple(DIMENSION_META)


@dataclass(frozen=True)
class FilterSpec:
    """Threshold set; any dimension left ``None`` is not applied.

    Monotonicity contract relied on by the tests: tightening any single
    threshold (raising a ``*_max`` bound, lowering a ``*_min`` bound or the
    range floor, raising ``speed_min``) can only remove points, so the
    survivor set is always a subset of the looser setting's survivor set.
    """

    range_max: float | None = None  # keep range <= range_max (m)
    range_min: float | None = None  # keep range >= range_min (m)
    angle_max: float | None = None  # keep |angle| <= angle_max (deg)
    speed_min: float | None = None  # keep |speed| >= speed_min (m/s)
    speed_max: float | None = None  # keep |speed| <= speed_max (m/s)
    snr_min: float | None = None  # keep snr >= snr_min (dB)

    def kept_mask(self, cloud) -> np.ndarray:
        """Boolean mask over the cloud; ``True`` marks a retained target."""
        keep = np.ones(len(cloud), dtype=bool)
        if self.range_max is not None:
            keep &= cloud.range_m <= self.range_max
        if self.range_min is not None:
            keep &= cloud.range_m >= self.range_min
        if self.angle_max is not None:
            keep &= np.abs(cloud.angle_deg) <= self.angle_max
        if self.speed_min is not None:
            keep &= np.abs(cloud.speed_ms) >= self.speed_min
        if self.speed_max is not None:
            keep &= np.abs(cloud.speed_ms) <= self.speed_max
        if self.snr_min is not None:
            keep &= cloud.snr_db >= self.snr_min
        return keep


def single_dim_spec(dimension: str, value: float) -> FilterSpec:
    """A :class:`FilterSpec` with exactly one dimension active."""
    if dimension not in DIMENSION_META:
        raise KeyError(f"unknown filter dimension: {dimension!r}")
    return FilterSpec(**{DIMENSION_META[dimension].spec_field: float(value)})
