"""Conservation checks for re-binned density fields.

The invariant verified here is the discrete continuity equation of the
re-binning transfer: representing the same detection mass on a different grid
must not create or destroy weight, i.e.

    sum(density * A)  ==  sum(counts)      (relative tolerance 1e-9)

where ``counts`` is the count field the density was derived from.  For a
single exposure frame (``n_frames == 1``) this is the literal identity above;
for an ``F``-frame capture both sides scale by ``1 / F`` and the identity is
checked against the exposure-normalised count map ``counts / F``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_RTOL = 1e-9


@dataclass
class ConservationReport:
    """Result of one conservation check."""

    left: float          # sum(density * A)
    right: float         # sum(counts) (or sum(counts / n_frames))
    abs_error: float
    rel_error: float
    n_frames: int
    passed: bool

    def as_line(self, label: str) -> str:
        status = "PASS" if self.passed else "FAIL"
        return (
            f"[{status}] {label}: sum(density*A)={self.left:.9g} "
            f"sum(counts)={self.right:.9g} rel_err={self.rel_error:.3e} "
            f"(n_frames={self.n_frames}, rtol={DEFAULT_RTOL:.0e})"
        )


def check_conservation(
    density: np.ndarray,
    areas: np.ndarray,
    counts: np.ndarray,
    n_frames: int = 1,
    rtol: float = DEFAULT_RTOL,
) -> ConservationReport:
    """Verify ``sum(density * A) == sum(counts)`` within relative tolerance.

    ``density`` and ``areas`` describe one grid representation;
    ``counts / n_frames`` is the exposure-normalised detection mass the
    density field must integrate to.
    """
    density = np.asarray(density, dtype=float)
    areas = np.asarray(areas, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if not (density.shape == areas.shape):
        raise ValueError("density and areas shape mismatch")
    left = float(np.sum(density * areas))
    right = float(np.sum(counts) / float(n_frames))
    abs_error = abs(left - right)
    denom = max(abs(left), abs(right), 1e-300)
    rel_error = abs_error / denom
    return ConservationReport(
        left=left,
        right=right,
        abs_error=abs_error,
        rel_error=rel_error,
        n_frames=int(n_frames),
        passed=bool(rel_error <= rtol),
    )


def check_rebin_conservation(
    density_src: np.ndarray,
    areas_src: np.ndarray,
    density_dst: np.ndarray,
    areas_dst: np.ndarray,
    rtol: float = DEFAULT_RTOL,
) -> ConservationReport:
    """Verify a re-binned field preserves the area-integrated density mass.

    Checks ``sum(density_dst * A_dst) == sum(density_src * A_src)`` — the
    quantity that :func:`polar_density.binning.rebin_polar` conserves by
    construction; the check guards against edge or convention regressions.
    """
    density_src = np.asarray(density_src, dtype=float)
    areas_src = np.asarray(areas_src, dtype=float)
    density_dst = np.asarray(density_dst, dtype=float)
    areas_dst = np.asarray(areas_dst, dtype=float)
    left = float(np.sum(density_dst * areas_dst))
    right = float(np.sum(density_src * areas_src))
    abs_error = abs(left - right)
    denom = max(abs(left), abs(right), 1e-300)
    rel_error = abs_error / denom
    return ConservationReport(
        left=left,
        right=right,
        abs_error=abs_error,
        rel_error=rel_error,
        n_frames=1,
        passed=bool(rel_error <= rtol),
    )
