"""a100-coordinate-consistency: coordinate convention & geometric closure
study for CTSAI-A100 / RadarTools RawTarget & TraTarget target logs."""

__version__ = "1.0.0"

from .closure import ABSTAIN, evaluate_file, residual_stats, row_residuals  # noqa: F401
from .census import FOCUS_FIELDS, file_census  # noqa: F401
from .hypotheses import HYPOTHESES, HYPOTHESIS_ORDER, predict, wrap180  # noqa: F401
from .parser import Frame, TargetLog, parse_file, parse_string  # noqa: F401
