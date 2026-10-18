"""Slow-time phase perturbation models (intra-frame, per chirp).

All models inject a per-chirp phase offset phi_k on the selected range bin's
complex slow-time sequence s_k via ``s_k * exp(j*phi_k)``.  Amplitudes and
the number of samples are untouched — the perturbation is *pure phase*.

Models
------
1. ``iid``       -- independent Gaussian phases, phi_k ~ N(0, sigma^2).
2. ``ar1``       -- first-order AR phase process
                        phi_k = a * phi_{k-1} + eps_k,  eps_k ~ N(0, s_eps^2)
                    Two parameterisations:
                    * marginal   : stationary std of phi fixed to sigma
                                   (s_eps = sigma*sqrt(1-a^2), stationary
                                   start phi_0 ~ N(0, sigma^2));
                    * innovation : s_eps fixed, marginal std grows as
                                   s_eps/sqrt(1-a^2) (a -> 1 becomes a
                                   random walk).
3. ``rw``        -- random walk, phi_k = phi_{k-1} + eps_k,
                    eps_k ~ N(0, sigma_step^2) (the a -> 1 limit of AR(1)).

The characteristic correlation length of AR(1) is ``1/(1-a)`` chirps.

AR(1) processes are generated with ``scipy.signal.lfilter`` (vectorised
across Monte-Carlo trials) plus a transient burn-in so that the stationary
law holds for the returned segment.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def correlation_length(a: float) -> float:
    """AR(1) correlation length 1/(1-a) in chirps."""
    if not (0.0 <= a < 1.0):
        raise ValueError("a must satisfy 0 <= a < 1")
    return 1.0 / (1.0 - a)


def apply_phase_perturbation(seq: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Multiply each chirp's complex sample by exp(j*phi_k).

    ``|s_k|`` and the sample count are invariant by construction.
    """
    seq = np.asarray(seq)
    phi = np.asarray(phi, dtype=np.float64)
    if seq.shape[-1] != phi.shape[-1]:
        raise ValueError(
            f"sequence length {seq.shape[-1]} != phase length {phi.shape[-1]}"
        )
    return seq * np.exp(1j * phi)


def iid_gaussian_phase(rng: np.random.Generator, n: int, sigma: float) -> np.ndarray:
    """IID Gaussian phases phi_k ~ N(0, sigma^2), sigma in rad."""
    return rng.normal(0.0, sigma, size=n)


def _ar1_batch(rng: np.random.Generator, trials: int, n: int, a: float,
               sigma_eps: float, sigma_init: float,
               burn_in: int | None = None) -> np.ndarray:
    """Stationary-start AR(1) phases, shape ``(trials, n)``."""
    if not (0.0 <= a < 1.0):
        raise ValueError("a must satisfy 0 <= a < 1")
    if burn_in is None:
        burn_in = int(max(500, 20.0 / max(1e-9, 1.0 - a)))
    total = burn_in + int(n)
    eps = rng.normal(0.0, sigma_eps, size=(trials, total))
    phi_prev = rng.normal(0.0, sigma_init, size=(trials, 1))
    # y[k] = eps[k] + a * y[k-1], with y[-1] = phi_prev (stationary start).
    zi = a * phi_prev
    phi = lfilter([1.0], [1.0, -a], eps, axis=1, zi=zi)[0]
    return phi[:, burn_in:]


def ar1_phase(rng: np.random.Generator, n: int, a: float,
              sigma_eps: float, burn_in: int | None = None) -> np.ndarray:
    """AR(1) phase process with innovation std sigma_eps (stationary start)."""
    return _ar1_batch(rng, 1, n, a, sigma_eps,
                      sigma_eps / np.sqrt(1.0 - a * a), burn_in)[0]


def ar1_phase_marginal(rng: np.random.Generator, n: int, a: float,
                       sigma: float, burn_in: int | None = None) -> np.ndarray:
    """AR(1) phase process whose *stationary marginal std* equals sigma.

    sigma_eps = sigma * sqrt(1-a^2); phi_0 drawn from the stationary law
    N(0, sigma^2); a transient burn-in removes start-up effects.
    """
    return _ar1_batch(rng, 1, n, a, sigma * np.sqrt(1.0 - a * a), sigma,
                      burn_in)[0]


def ar1_phase_marginal_batch(rng: np.random.Generator, trials: int, n: int,
                             a: float, sigma: float,
                             burn_in: int | None = None) -> np.ndarray:
    """Batched :func:`ar1_phase_marginal`, shape ``(trials, n)``."""
    return _ar1_batch(rng, trials, n, a, sigma * np.sqrt(1.0 - a * a), sigma,
                      burn_in)


def ar1_phase_innovation_batch(rng: np.random.Generator, trials: int, n: int,
                               a: float, sigma_eps: float,
                               burn_in: int | None = None) -> np.ndarray:
    """Batched AR(1) with fixed innovation std, shape ``(trials, n)``."""
    return _ar1_batch(rng, trials, n, a, sigma_eps,
                      sigma_eps / np.sqrt(1.0 - a * a), burn_in)


def random_walk_phase(rng: np.random.Generator, n: int,
                      sigma_step: float) -> np.ndarray:
    """Random-walk phase, steps N(0, sigma_step^2), phi_0 = 0."""
    return np.cumsum(rng.normal(0.0, sigma_step, size=n))


def random_walk_phase_batch(rng: np.random.Generator, trials: int, n: int,
                            sigma_step: float) -> np.ndarray:
    """Batched random-walk phases, shape ``(trials, n)``."""
    return np.cumsum(
        rng.normal(0.0, sigma_step, size=(trials, n)), axis=1)
