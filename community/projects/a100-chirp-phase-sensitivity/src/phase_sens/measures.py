"""Coherent-integration and Doppler-spectrum measures + theory anchors.

Gain definitions (both normalized so that a perfectly coherent sequence
gives 1 and incoherent phases drive the value toward the window-dependent
coherent floor):

*   Unweighted coherent integration gain::

        G      = |sum_k s_k|^2 / (sum_k |s_k|)^2

*   Weighted gain (slow-time window ``w_k``, e.g. the platform's hann)::

        G_w    = |sum_k w_k s_k|^2 / (sum_k w_k |s_k|)^2

Doppler measures (applied to the official Doppler FFT output
``X_m = sum_k w_k s_k exp(-j 2 pi m k / M)``):

*   ``peak_amplitude``      -- max_m |X_m|;
*   ``neighborhood_energy_ratio`` -- energy within +-``guard`` bins of the
  peak (circular) divided by total spectral energy;
*   ``second_central_moment`` -- spread of the normalised power spectrum
  around the peak-centred circular mean, in bins^2::

        d_i   = signed distance of bin i to the peak bin (wrapped to
                (-M/2, M/2])
        p_i   = |X_i|^2 / sum_i |X_i|^2
        mu    = sum_i p_i d_i
        M2    = sum_i p_i (d_i - mu)^2

Theory anchors (derivations in the project README):

*   IID Gaussian phase, equal amplitudes, no window::

        E[G] = 1/N + (1 - 1/N) exp(-sigma^2)

*   IID Gaussian phase, equal amplitudes, window ``w``::

        E[G_w] = r + (1 - r) exp(-sigma^2),   r = sum w_k^2 / (sum w_k)^2

*   Jointly Gaussian phases with Cov(phi_k, phi_l) = C(k, l)::

        E[exp(j(phi_k - phi_l))] = exp(-(Var phi_k + Var phi_l - 2 C)/2)

    which specialises to (N = sequence length, h = |k - l|):

    - AR(1), stationary marginal std sigma::

        E[G] = 1/N + (2/N^2) sum_{h=1}^{N-1} (N-h) exp(-sigma^2 (1 - a^h))

    - AR(1), innovation std sigma_eps (stationary var sigma_eps^2/(1-a^2))::

        E[G] = 1/N + (2/N^2) sum_h (N-h) exp(-sigma_eps^2 (1 - a^h)/(1 - a^2))

    - random walk with step std sigma_step::

        E[G] = 1/N + (2/N^2) sum_h (N-h) exp(-sigma_step^2 h / 2)

These expressions assume the underlying (unperturbed) sequence has been
*demodulated to DC* — i.e. its unperturbed phases align (a zero-Doppler
tone, or a non-zero-Doppler target after removing the dominant Doppler ramp
via :func:`demodulate_peak`).  The plain sum ``sum_k s_k`` is only a
coherent integrator for an aligned sequence; a bin-centred tone at a
non-zero Doppler bin sums to zero.  The formulas are validated on the
synthetic benchmark and are *not* inverted against the captures to make
hardware claims.
"""

from __future__ import annotations

from typing import Dict

import numpy as np

from .pipeline import SensorConfig, doppler_fft


# --------------------------------------------------------------------------
# Coherent integration gain
# --------------------------------------------------------------------------

def coherent_gain(seq: np.ndarray) -> float:
    """Unweighted gain ``|sum s_k|^2 / (sum |s_k|)^2`` (1 = fully coherent)."""
    seq = np.asarray(seq)
    denom = np.sum(np.abs(seq))
    if denom == 0:
        raise ValueError("all-zero sequence")
    return float(np.abs(np.sum(seq)) ** 2 / denom ** 2)


def coherent_gain_weighted(seq: np.ndarray, w: np.ndarray) -> float:
    """Windowed gain ``|sum w_k s_k|^2 / (sum w_k |s_k|)^2``."""
    seq = np.asarray(seq)
    w = np.asarray(w, dtype=np.float64)
    if seq.shape[-1] != w.shape[-1]:
        raise ValueError("sequence and window length mismatch")
    denom = np.sum(w * np.abs(seq))
    if denom == 0:
        raise ValueError("degenerate window or all-zero sequence")
    return float(np.abs(np.sum(w * seq)) ** 2 / denom ** 2)


def coherent_gain_matrix(seqs: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Vectorised gain for ``[trials, N]`` sequences (w=None -> unweighted)."""
    seqs = np.asarray(seqs)
    if w is None:
        denom = np.sum(np.abs(seqs), axis=-1)
        num = np.abs(np.sum(seqs, axis=-1)) ** 2
    else:
        w = np.asarray(w, dtype=np.float64)
        denom = np.sum(w * np.abs(seqs), axis=-1)
        num = np.abs(np.sum(w * seqs, axis=-1)) ** 2
    return num / denom ** 2


def demodulate_peak(seq: np.ndarray) -> np.ndarray:
    """Remove the dominant Doppler ramp so gain measures see aligned phases.

    Multiplies by ``exp(-j 2 pi m* k / N)`` where ``m*`` is the peak bin of
    the *unwindowed* N-point FFT of ``seq``.  ``sum_k s_k`` is only a
    coherent integrator for an aligned (DC-referenced) sequence, so any
    non-zero-Doppler sequence must be demodulated before ``coherent_gain``.
    """
    s = np.asarray(seq)
    m = int(np.argmax(np.abs(np.fft.fft(s, axis=-1)), axis=-1))
    k = np.arange(s.shape[-1])
    return s * np.exp(-1j * 2.0 * np.pi * m * k / s.shape[-1])


# --------------------------------------------------------------------------
# Theory anchors
# --------------------------------------------------------------------------

def theoretical_gain_iid(sigma: float, n: int) -> float:
    """E[G] for IID N(0, sigma^2) phases, equal amplitudes, no window."""
    sigma = float(sigma)
    return 1.0 / n + (1.0 - 1.0 / n) * np.exp(-sigma ** 2)


def theoretical_gain_weighted_iid(sigma: float, w: np.ndarray) -> float:
    """E[G_w] for IID N(0, sigma^2) phases under window ``w``."""
    w = np.asarray(w, dtype=np.float64)
    r = float(np.sum(w ** 2) / np.sum(w) ** 2)
    return r + (1.0 - r) * np.exp(-float(sigma) ** 2)


def _pair_average(exp_factors: np.ndarray, n: int) -> float:
    """1/N + (2/N^2) sum_h (N-h) * factor_h  for pair-averaging formulas."""
    h = np.arange(1, n)
    return float(1.0 / n + (2.0 / n ** 2) * np.sum((n - h) * exp_factors))


def theoretical_gain_ar1(sigma: float, a: float, n: int,
                          param: str = "marginal") -> float:
    """E[G] under stationary AR(1) phases.

    ``param='marginal'``: ``sigma`` is the stationary marginal std.
    ``param='innovation'``: ``sigma`` is the innovation std sigma_eps.
    """
    if not (0.0 <= a < 1.0):
        raise ValueError("a must satisfy 0 <= a < 1")
    h = np.arange(1, n)
    if param == "marginal":
        factors = np.exp(-float(sigma) ** 2 * (1.0 - a ** h))
    elif param == "innovation":
        factors = np.exp(-float(sigma) ** 2 * (1.0 - a ** h) / (1.0 - a * a))
    else:
        raise ValueError("param must be 'marginal' or 'innovation'")
    return _pair_average(factors, n)


def theoretical_gain_rw(sigma_step: float, n: int) -> float:
    """E[G] under random-walk phases with step std sigma_step."""
    h = np.arange(1, n)
    return _pair_average(np.exp(-float(sigma_step) ** 2 * h / 2.0), n)


# --------------------------------------------------------------------------
# Doppler spectrum measures
# --------------------------------------------------------------------------

def doppler_metrics(bin_seq: np.ndarray, cfg: SensorConfig,
                    guard: int = 3) -> Dict[str, float]:
    """Peak / concentration / spread measures of the Doppler spectrum.

    Returns a dict with ``peak_amplitude``, ``peak_bin``,
    ``neighborhood_energy_ratio`` (energy within +-``guard`` bins of the
    peak over total energy) and ``second_central_moment`` (bins^2, peak
    centred; 0 for a noiseless bin-centred tone before perturbation).
    """
    x = np.asarray(bin_seq, dtype=complex)[None, :]
    batch = doppler_metrics_trials(x, cfg, guard=guard)
    out = {
        "peak_amplitude": float(batch["peak_amplitude"][0]),
        "peak_bin": int(batch["peak_bin"][0]),
        "neighborhood_energy_ratio": float(batch["neighborhood_energy_ratio"][0]),
        "second_central_moment": float(batch["second_central_moment"][0]),
    }
    return out


def doppler_metrics_trials(bin_seqs: np.ndarray, cfg: SensorConfig,
                           guard: int = 3) -> Dict[str, np.ndarray]:
    """Batched Doppler metrics for ``[trials, nchirp]`` perturbed sequences."""
    x = doppler_fft(bin_seqs, cfg)
    power = np.abs(x) ** 2
    m = power.shape[-1]
    peak_bins = np.argmax(power, axis=-1)
    peak_amp = np.sqrt(power.max(axis=-1))
    total = power.sum(axis=-1)
    p = power / total[:, None]
    trials = power.shape[0]
    nb = np.empty(trials)
    m2 = np.empty(trials)
    idx = np.arange(m)
    for t in range(trials):
        b0 = int(peak_bins[t])
        d = (idx - b0) % m
        d = np.where(d > m // 2, d - m, d)
        mu = float(np.sum(p[t] * d))
        m2[t] = np.sum(p[t] * (d - mu) ** 2)
        shifted = np.roll(power[t], -b0)
        nb[t] = (shifted[: guard + 1].sum() + shifted[m - guard:].sum()) / total[t]
    return {
        "peak_amplitude": peak_amp,
        "peak_bin": peak_bins,
        "neighborhood_energy_ratio": nb,
        "second_central_moment": m2,
    }


def adjacent_phase_diff(bin_seq: np.ndarray) -> np.ndarray:
    """Wrapped adjacent-chirp phase differences angle(s_{k+1} conj(s_k))."""
    s = np.asarray(bin_seq)
    return np.angle(s[1:] * np.conj(s[:-1]))


def circular_stats(angles: np.ndarray) -> Dict[str, float]:
    """Circular mean / resultant length / circular std of wrapped angles."""
    z = np.exp(1j * np.asarray(angles, dtype=np.float64))
    mean_vec = z.mean()
    r_bar = float(np.abs(mean_vec))
    circ_std = float(np.sqrt(max(0.0, -2.0 * np.log(max(r_bar, 1e-12)))))
    return {
        "mean_angle_rad": float(np.angle(mean_vec)),
        "resultant_length": r_bar,
        "circular_std_rad": circ_std,
    }
