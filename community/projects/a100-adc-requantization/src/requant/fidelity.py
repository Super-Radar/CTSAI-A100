"""Fidelity metrics between a requantised run and its 16-bit baseline.

All spectral magnitudes come from :func:`requant.pipeline.spectral_chain`
(range spectrum = mean |Range-FFT| over chirps; rd_map = |Range-Doppler|).
Metrics are deliberately *relative to the k=16 baseline of the same file* --
never normalised per requantisation run -- so the budget they express cannot
be masked by a per-run gain.
"""

from __future__ import annotations

import numpy as np


def time_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """Time-domain RMSE in LSB (1 LSB = one int16 code)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    return float(np.sqrt(np.mean((a - b) ** 2)))


def spectrum_rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMSE between two magnitude spectra (range spectrum or RD map)."""
    return time_rmse(a, b)          # same algebra


def normalize_peak(x: np.ndarray) -> float:
    """Peak magnitude of a spectrum/map (used to make RMSE relative)."""
    return float(np.max(np.abs(x)))


def band_energy_retention(map_k: np.ndarray, map_base: np.ndarray,
                          band_mask: np.ndarray) -> float:
    """Energy inside a *predefined* band: E_k(band) / E_16(band).

    ``band_mask`` must be derived from the 16-bit baseline only (see
    :func:`predefined_band`), never from the requantised run.
    """
    e_k = float(np.sum(np.asarray(map_k, dtype=np.float64)[band_mask] ** 2))
    e_b = float(np.sum(np.asarray(map_base, dtype=np.float64)[band_mask] ** 2))
    if e_b == 0.0:
        raise ValueError("baseline band energy is zero")
    return e_k / e_b


def predefined_band(baseline_rd: np.ndarray, frac: float = 0.01) -> np.ndarray:
    """Boolean mask of the top-``frac`` energy bins of the baseline RD map.

    "预定义频带" for *measured* data: selected once from the 16-bit baseline
    and then held fixed for every k, so retention is a pure budget measure.
    """
    e = np.asarray(baseline_rd, dtype=np.float64).ravel() ** 2
    n = max(1, int(round(frac * e.size)))
    idx = np.argpartition(e, e.size - n)[-n:]
    mask = np.zeros(e.size, dtype=bool)
    mask[idx] = True
    return mask.reshape(np.asarray(baseline_rd).shape)


def signal_band(shape: tuple[int, int],
                boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    """Ground-truth band for synthetic cases: union of rectangular boxes.

    Each box is (range_bin, doppler_bin, d_range, d_doppler); ``d_doppler``
    may exceed the map to mean "all Doppler bins".
    """
    m = np.zeros(shape, dtype=bool)
    for r, v, dr, dv in boxes:
        m[max(0, r - dr):r + dr + 1,
          max(0, v - dv):min(shape[1], v + dv + 1)] = True
    return m


def peak_drift(map_k: np.ndarray, map_base: np.ndarray) -> dict:
    """Peak amplitude ratio and bin displacement of the RD-map maximum."""
    mk = np.asarray(map_k, dtype=np.float64)
    mb = np.asarray(map_base, dtype=np.float64)
    pk = np.unravel_index(int(np.argmax(mk)), mk.shape)
    pb = np.unravel_index(int(np.argmax(mb)), mb.shape)
    return {
        "peak_amp_ratio": float(mk[pk] / mb[pb]),
        "rng_bin_shift": int(pk[0] - pb[0]),
        "vel_bin_shift": int(pk[1] - pb[1]),
    }


def weak_tone_visibility(rd_map: np.ndarray, rng_bin: int, vel_bin: int,
                         search: int = 3, ring: int = 25, guard: int = 6,
                         strong_bin: tuple[int, int] = (137, 3)) -> dict:
    """Local-SNR visibility of a known weak tone in the RD map.

    The weak-tone peak is searched +/-``search`` bins around the expected
    (range, Doppler) bin.  The noise floor is the median of a surrounding
    annulus (range +/-``ring`` bins, all Doppler) that excludes a +/-``guard``
    zone around both the weak tone and the strong tone's bin (default
    (137, 3), the two-tone benchmark carrier) -- a *local* floor is
    essential: requantisation distortion is sparse, so the global map median
    collapses at low k and would fake high visibility.

    "Visible" is defined as ratio_db >= +10 dB in summary.md.
    """
    m = np.asarray(rd_map, dtype=np.float64)
    r0 = max(0, rng_bin - search)
    r1 = min(m.shape[0], rng_bin + search + 1)
    v0 = max(0, vel_bin - search)
    v1 = min(m.shape[1], vel_bin + search + 1)
    peak = float(np.max(m[r0:r1, v0:v1]))
    rr = np.arange(max(0, rng_bin - ring), min(m.shape[0], rng_bin + ring + 1))
    vv = np.arange(m.shape[1])
    sel = m[np.ix_(rr, vv)]
    keep = ((np.abs(rr - rng_bin) > guard)[:, None]
            & (np.abs(rr - strong_bin[0]) > guard)[:, None]
            & (np.abs(vv - vel_bin) > guard)[None, :]
            & (np.abs(vv - strong_bin[1]) > guard)[None, :])
    floor = float(np.median(sel[keep])) if keep.any() else float(np.median(sel))
    if floor <= 0.0:
        return {"weak_peak": peak, "floor": floor, "ratio_db": float("inf")}
    return {"weak_peak": peak, "floor": floor,
            "ratio_db": float(20.0 * np.log10(peak / floor))}
