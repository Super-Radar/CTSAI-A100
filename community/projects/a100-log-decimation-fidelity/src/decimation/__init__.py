"""a100-log-decimation-fidelity: offline fidelity bounds for frame-level decimation of
RadarTools RawTarget point-cloud logs.

Submodules
----------
- ``parser``     : multi-frame RawTarget CSV -> frame sequence (timestamps + per-frame targets).
- ``samplers``   : deterministic uniform keep-every-k phase enumeration, time-stratified
                   random sampling, and byte-budget k inversion.
- ``fidelity``   : time-weighted polar occupancy histograms, L1/L2 distances, short-lived
                   cell retention, per-frame count statistics, and synthetic benchmarks.
- ``cli``        : reproducible experiment driver (fixed seed 20261003).
"""

from .parser import Frame, TargetLog, parse_target_log
from .samplers import (
    byte_budget_k,
    enumerate_uniform_phases,
    stratified_keep_indices,
    uniform_keep_indices,
)
from .fidelity import (
    CellGrid,
    analytic_missing_phases,
    count_statistics,
    frame_time_weights,
    merged_time_weights,
    synthetic_irregular_session,
    synthetic_pulse_session,
    tv_l1,
    l2_distance,
)

__all__ = [
    "Frame",
    "TargetLog",
    "parse_target_log",
    "uniform_keep_indices",
    "enumerate_uniform_phases",
    "stratified_keep_indices",
    "byte_budget_k",
    "CellGrid",
    "frame_time_weights",
    "merged_time_weights",
    "tv_l1",
    "l2_distance",
    "count_statistics",
    "analytic_missing_phases",
    "synthetic_pulse_session",
    "synthetic_irregular_session",
]
