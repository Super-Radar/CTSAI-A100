"""Window functions with coherent gain (CG) and equivalent noise bandwidth (ENBW).

Two families of windows appear in this project, on purpose:

* ``periodic`` windows (``sym=False``, scipy ``fftbins=True``) are the DFT-even
  standard of spectral analysis.  For the cosine windows used here
  (hann / hamming / blackman) they have closed-form CG and ENBW, so the
  synthetic benchmark in :mod:`window_study.study` compares measured values
  against theory with a tight (1e-6) tolerance.
* ``symmetric`` windows (``sym=True``) equal MATLAB's ``hann(N)`` /
  ``hamming(N)`` defaults.  The official A100 MATLAB reference chain
  (``FFT/do_fft_2d.m``) windows with ``hann(adc_sample_num)``, i.e. a
  *symmetric* window, so the real-data pipeline in :mod:`window_study.pipeline`
  reproduces the chain with ``sym=True``.

Core quantities (window ``w`` of length ``N``, FFT length ``nfft``):

* coherent gain ``CG = sum(w) / N`` -- the factor by which a bin-centred tone
  is scaled by the window; the unbiased amplitude estimate of a real tone of
  amplitude ``A`` at DFT bin ``k`` is ``A_hat = 2*|X[k]| / sum(w)``.
* equivalent noise bandwidth ``ENBW = nfft * sum(w^2) / sum(w)^2`` in DFT bins
  of the (possibly zero-padded) FFT -- for white noise this is the width of the
  rectangle that passes the same noise power.
* highest sidelobe level in dB relative to the mainlobe peak.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import get_window

WINDOW_NAMES = ("rect", "hann", "hamming", "blackman")

# Periodic cosine-window coefficients w[n] = a0 - a1*cos(2*pi*n/N) + a2*cos(4*pi*n/N),
# matching scipy.signal.get_window for these names.
_COEFFS = {
    "rect": (1.0, 0.0, 0.0),
    "hann": (0.5, 0.5, 0.0),
    "hamming": (0.54, 0.46, 0.0),
    "blackman": (0.42, 0.5, 0.08),
}

# Classical highest-sidelobe levels of the large-N DTFT (dB), as quoted in the
# standard literature (F. J. Harris, Proc. IEEE 1978); finite-N and
# symmetric-vs-periodic choices shift these by up to a few tenths of a dB.
THEORY_SIDELOBE_DB = {"rect": -13.3, "hann": -31.5, "hamming": -42.7, "blackman": -58.1}


def make_window(name: str, n: int, sym: bool = False) -> np.ndarray:
    """Return a window of length ``n``.

    ``sym=False`` -> periodic (DFT-even) window, default for spectral analysis.
    ``sym=True``  -> symmetric window, identical to MATLAB ``hann(N)`` etc.,
    used to reproduce the official A100 MATLAB chain.
    """
    if name == "rect":
        return np.ones(n, dtype=float)
    if name not in WINDOW_NAMES:
        raise ValueError(f"unknown window {name!r}; expected one of {WINDOW_NAMES}")
    return np.asarray(get_window(name, n, fftbins=not sym), dtype=float)


def coherent_gain(w: np.ndarray) -> float:
    """CG = mean(w) = sum(w)/N (window gain on a bin-centred tone)."""
    return float(np.sum(w) / len(w))


def enbw_bins(w: np.ndarray, nfft: int | None = None) -> float:
    """ENBW expressed in DFT bins of an FFT padded to ``nfft`` (default: no padding)."""
    if nfft is None:
        nfft = len(w)
    sw = float(np.sum(w))
    return float(nfft * np.sum(np.square(w)) / (sw * sw))


def analytic_metrics(name: str) -> dict:
    """Closed-form CG / ENBW / classical sidelobe level for the periodic window.

    For the periodic cosine windows the cosine sums vanish exactly, so
    ``CG = a0`` and ``ENBW = (a0^2 + a1^2/2 + a2^2/2) / a0^2`` hold to machine
    precision for any length N.
    """
    if name not in _COEFFS:
        raise ValueError(f"no analytic metrics for {name!r}")
    a0, a1, a2 = _COEFFS[name]
    cg = a0
    enbw = (a0 * a0 + a1 * a1 / 2.0 + a2 * a2 / 2.0) / (a0 * a0)
    return {
        "window": name,
        "cg": cg,
        "cg_db": 20.0 * np.log10(cg),
        "enbw_bins": enbw,
        "sidelobe_db": THEORY_SIDELOBE_DB[name],
    }


def window_kernel(w: np.ndarray, npad: int) -> np.ndarray:
    """Zero-padded FFT of the window, normalised to unit peak, first half only."""
    W = np.fft.fft(w, npad)[: npad // 2]
    return W / W[0]


def kernel_first_null(W: np.ndarray) -> int:
    """Index (padded bins) of the first local minimum after the mainlobe peak."""
    m = np.abs(W)
    idx = np.where((m[1:-1] < m[:-2]) & (m[1:-1] <= m[2:]))[0] + 1
    return int(idx[0])


def kernel_sidelobe_db(W: np.ndarray) -> float:
    """Highest sidelobe (dB rel. mainlobe) of a unit-peak window kernel."""
    k0 = kernel_first_null(W)
    db = 20.0 * np.log10(np.abs(W[k0:]) + 1e-300)
    return float(np.max(db))


def _cross_down(m: np.ndarray, target: float, start: int, step: int) -> float:
    """First crossing of ``target`` walking from ``start`` in direction ``step``,
    with linear interpolation between the two straddling samples."""
    j = start
    n = len(m)
    if step > 0:
        while j < n - 1 and m[j] > target:
            j += 1
    else:
        while j > 0 and m[j] > target:
            j -= 1
    prev = j - step
    if m[j] == m[prev]:
        return float(j - step / 2.0)
    frac = (target - m[prev]) / (m[j] - m[prev])
    return prev + step * frac


def kernel_3db_width_bins(w: np.ndarray, npad: int = 65536) -> float:
    """Mainlobe width at -3.01 dB, in bins of the original N-length DFT.

    A real window has a magnitude-symmetric DTFT, so the width is twice the
    right-hand -3 dB crossing.
    """
    W = window_kernel(w, npad)
    m = 20.0 * np.log10(np.abs(W) + 1e-300)
    target = -10.0 * np.log10(2.0)  # -3.0103 dB
    half_right = _cross_down(m, target, 0, +1)
    return 2.0 * half_right * len(w) / npad


def measured_3db_width_bins(db: np.ndarray, k: int) -> float:
    """Measured -3 dB width (bins) of the spectral peak at index ``k``.

    ``db`` is a dB spectrum; linear interpolation is used between bins.
    """
    target = db[k] - 10.0 * np.log10(2.0)
    right = _cross_down(db, target, k, +1)
    left = _cross_down(db, target, k, -1)
    return float(right - left)
