"""Coordinate-convention hypothesis family for polar -> cartesian mapping.

Each hypothesis maps a radar report ``(range r [m], angle theta [deg])`` to a
predicted cartesian pair ``(X, Y)`` so it can be compared with the ``X``/``Y``
columns actually logged by the RadarTools host software.

Hypotheses (theta converted to radians, evaluated with plain trig):

=====  ============================  ===========================
name   predicted X                   predicted Y
=====  ============================  ===========================
H1     ``-r * sin(theta)``           ``r * cos(theta)``
H2     ``+r * sin(theta)``           ``r * cos(theta)``
H3     ``r * cos(theta)``            ``r * sin(theta)``
H4     ``-r * cos(theta)``           ``r * sin(theta)``
=====  ============================  ===========================

H1 is the convention suggested by a spot check of session
``2026_05_22_13_41_37`` (TraTarget): angle in degrees measured clockwise from
the forward axis, X pointing left, Y pointing forward.  The closure study in
``closure.py`` quantifies how well each hypothesis reproduces the logged
cartesian columns across the whole corpus.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

__all__ = ["HYPOTHESES", "HYPOTHESIS_ORDER", "predict", "wrap180", "ANGLE_OFFSET_GRID"]

Predictor = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]

HYPOTHESES: Dict[str, Predictor] = {
    "H1": lambda r, t: (-r * np.sin(t), r * np.cos(t)),
    "H2": lambda r, t: (r * np.sin(t), r * np.cos(t)),
    "H3": lambda r, t: (r * np.cos(t), r * np.sin(t)),
    "H4": lambda r, t: (-r * np.cos(t), r * np.sin(t)),
}

HYPOTHESIS_ORDER: Tuple[str, ...] = ("H1", "H2", "H3", "H4")

# Optional angle-zero offset scan grid [deg], +-5 deg in 0.5 deg steps.
ANGLE_OFFSET_GRID: np.ndarray = np.arange(-5.0, 5.0 + 1e-9, 0.5)


def wrap180(angle_deg):
    """Wrap angles (array or scalar) to the half-open interval [-180, 180)."""
    a = np.asarray(angle_deg, dtype=float)
    w = (a + 180.0) % 360.0 - 180.0
    if np.isscalar(angle_deg) or w.ndim == 0:
        return float(w)
    return w


def predict(name: str, r, angle_deg, angle_offset_deg: float = 0.0):
    """Predict (X, Y) for hypothesis ``name`` given range [m] and angle [deg].

    ``angle_offset_deg`` shifts the angle zero point before evaluation; it is
    swept over :data:`ANGLE_OFFSET_GRID` by the closure study.
    """
    if name not in HYPOTHESES:
        raise KeyError(f"unknown hypothesis {name!r}")
    r = np.asarray(r, dtype=float)
    theta = np.deg2rad(np.asarray(angle_deg, dtype=float) + angle_offset_deg)
    return HYPOTHESES[name](r, theta)
