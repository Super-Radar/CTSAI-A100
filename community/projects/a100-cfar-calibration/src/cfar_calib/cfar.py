"""CA-CFAR with the exact topology of the community validator, plus theory.

Topology (identical to ``community/projects/a100-matlab-adc-pipeline-validation/
validator/run_reference_validation.py::cfar``):

* 2D cell-averaging ring kernel on the power map ``(range bins, doppler bins)``
  with one-sided training widths ``training = (tr, td)`` and guard widths
  ``guard = (gr, gd)``; defaults ``(8, 6)`` / ``(2, 2)`` -> N = 332 training
  cells;
* the bounded range axis is zero ("constant") padded, the periodic Doppler
  axis is wrap padded;
* detection = ``power > alpha * Z`` AND 8-neighbourhood local maximum AND
  range border of ``tr + gr`` rows excluded.

Threshold-factor theory shipped with this module:

* single-look exponential noise (what the validator assumes)::

      P_fa = (1 + alpha / N)^(-N)      alpha = N * (P_fa^(-1/N) - 1)

* K-look noise (K channels power-averaged, still iid cells).  Writing
  Y ~ Gamma(K, rate K) for the cell power and S for the sum over the N
  training cells, Y / (Y + S) ~ Beta(K, N*K), which gives the exact::

      P_fa = P(Y > (alpha/N) S) = 1 - I_{alpha/(alpha+N)}(K, N*K)

  where I is the regularised incomplete beta function.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
from scipy.special import betainc

TRAINING = (8, 6)  # one-sided training cells (range, doppler) - validator default
GUARD = (2, 2)     # one-sided guard cells (range, doppler)   - validator default


def kernel_count(tr: int = TRAINING[0], td: int = TRAINING[1],
                 gr: int = GUARD[0], gd: int = GUARD[1]) -> int:
    """Number of training cells in the ring kernel."""
    outer = (2 * (tr + gr) + 1) * (2 * (td + gd) + 1)
    inner = (2 * gr + 1) * (2 * gd + 1)
    return outer - inner


# --------------------------------------------------------------------------
# threshold-factor <-> Pfa conversions (iid cell model)
# --------------------------------------------------------------------------

def alpha_from_pfa_single_look(pfa: float, n_cells: int) -> float:
    """Threshold factor for iid single-look exponential noise."""
    if not 0.0 < pfa < 1.0:
        raise ValueError("pfa must be in (0, 1)")
    return float(n_cells * (pfa ** (-1.0 / n_cells) - 1.0))


def pfa_single_look(alpha: float, n_cells: int) -> float:
    """Pfa of the threshold rule for iid single-look exponential noise."""
    return float((1.0 + alpha / n_cells) ** (-n_cells))


def pfa_k_look(alpha: float, n_cells: int, k: int) -> float:
    """Exact Pfa for iid Gamma(K, K) cells (K-channel power averaging)."""
    if k < 1:
        raise ValueError("k >= 1 required")
    if alpha <= 0.0:
        return 1.0
    x = float(alpha) / (float(alpha) + float(n_cells))
    return float(1.0 - betainc(k, n_cells * k, x))


def alpha_k_look(pfa: float, n_cells: int, k: int) -> float:
    """Invert :func:`pfa_k_look` by bisection (needs *smaller* alpha as K grows)."""
    if k == 1:
        return alpha_from_pfa_single_look(pfa, n_cells)
    hi = alpha_from_pfa_single_look(pfa, n_cells)
    lo = hi * 1e-6
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if pfa_k_look(mid, n_cells, k) > pfa:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# sliding-window operators (vectorised, padding semantics of the validator)
# --------------------------------------------------------------------------

def _box_sum(a: np.ndarray, radius: int, axis: int, mode: str) -> np.ndarray:
    """Sum over a ``(2*radius+1)`` window along ``axis`` (cumsum based)."""
    if radius <= 0:
        return a.astype(np.float64, copy=True)
    pad = [(0, 0)] * a.ndim
    pad[axis] = (radius, radius)
    p = np.pad(a, pad, mode=mode)
    c = np.cumsum(p, axis=axis, dtype=np.float64)
    n = a.shape[axis]
    zero_shape = list(c.shape)
    zero_shape[axis] = 1
    c = np.concatenate([np.zeros(zero_shape, dtype=np.float64), c], axis=axis)
    idx_hi = np.arange(n) + 2 * radius + 1
    idx_lo = np.arange(n)
    return np.take(c, idx_hi, axis=axis) - np.take(c, idx_lo, axis=axis)


def ca_noise_estimate(
    power: np.ndarray,
    training: Tuple[int, int] = TRAINING,
    guard: Tuple[int, int] = GUARD,
) -> np.ndarray:
    """Mean training-cell level Z with the validator's padding semantics."""
    tr, td = training
    gr, gd = guard
    outer_r, outer_d = tr + gr, td + gd
    outer = _box_sum(_box_sum(power, outer_r, 0, "constant"), outer_d, 1, "wrap")
    inner = _box_sum(_box_sum(power, gr, 0, "constant"), gd, 1, "wrap")
    return (outer - inner) / float(kernel_count(tr, td, gr, gd))


def local_max_mask(power: np.ndarray) -> np.ndarray:
    """8-neighbourhood local maximum (np.roll -> both axes wrap)."""
    lm = np.ones(power.shape, dtype=bool)
    for dr in (-1, 0, 1):
        for dd in (-1, 0, 1):
            if dr or dd:
                lm &= power >= np.roll(power, (dr, dd), axis=(0, 1))
    return lm


def interior_mask(shape: Tuple[int, int],
                  training: Tuple[int, int] = TRAINING,
                  guard: Tuple[int, int] = GUARD) -> np.ndarray:
    """Cells the validator keeps decisions on (range border dropped, Doppler
    is periodic so no border is dropped there)."""
    m = np.ones(shape, dtype=bool)
    b = training[0] + guard[0]
    m[:b] = False
    m[-b:] = False
    return m


def cfar_detect(
    power: np.ndarray,
    alpha: float,
    training: Tuple[int, int] = TRAINING,
    guard: Tuple[int, int] = GUARD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Full community detector: threshold + local maximum + range border."""
    z = ca_noise_estimate(power, training, guard)
    mask = power > alpha * z
    mask &= local_max_mask(power)
    mask &= interior_mask(power.shape, training, guard)
    return mask, z


def db_ring_stats(
    db: np.ndarray,
    training: Tuple[int, int] = TRAINING,
    guard: Tuple[int, int] = GUARD,
) -> Tuple[np.ndarray, np.ndarray]:
    """Local mean and std of a dB(power) map over the same CA training ring.

    Used only to model the community misuse "threshold = mean_dB + k * std_dB".
    Edge padding is used for the bounded axis; on interior cells (at least
    ``tr + gr`` rows from the range borders) the ring never touches the pad,
    so all statistics reported by this project are computed on interior
    cells only and are unaffected by the pad values.
    """
    tr, td = training
    gr, gd = guard
    outer_r, outer_d = tr + gr, td + gd
    s1 = _box_sum(_box_sum(db, outer_r, 0, "edge"), outer_d, 1, "wrap")
    s1 -= _box_sum(_box_sum(db, gr, 0, "edge"), gd, 1, "wrap")
    s2 = _box_sum(_box_sum(db * db, outer_r, 0, "edge"), outer_d, 1, "wrap")
    s2 -= _box_sum(_box_sum(db * db, gr, 0, "edge"), gd, 1, "wrap")
    n = float(kernel_count(tr, td, gr, gd))
    mu = s1 / n
    var = np.maximum(s2 / n - mu * mu, 0.0)
    return mu, np.sqrt(var)
