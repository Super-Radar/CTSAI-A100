"""a100-filter-selection-bias: threshold filtering selection-bias study.

Offline analysis of selection bias and interaction effects introduced by
range / angle / speed / SNR threshold filtering on CTSAI-A100 RawTarget
point-cloud logs (RadarTools example target logs, CTLRR-300-V2_1).
"""

__version__ = "1.0.0"

from .parser import PointCloud, parse_files, parse_text
from .filters import DIMENSIONS, DIMENSION_META, FilterSpec, single_dim_spec
from .sweep import (
    pairwise_jaccard,
    joint_grid,
    sweep_dimension,
    stratified_retention,
    distribution_shift,
    retention,
)

__all__ = [
    "__version__",
    "PointCloud",
    "parse_files",
    "parse_text",
    "DIMENSIONS",
    "DIMENSION_META",
    "FilterSpec",
    "single_dim_spec",
    "pairwise_jaccard",
    "joint_grid",
    "sweep_dimension",
    "stratified_retention",
    "distribution_shift",
    "retention",
]
