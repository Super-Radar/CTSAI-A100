"""Azimuth-only manifold ambiguity diagnostics, not a hardware DOA validator."""

import numpy as np


def steering(positions, angles):
    p = np.asarray(positions, dtype=float)
    a = np.asarray(angles, dtype=float)
    if (
        p.ndim != 1
        or not len(p)
        or a.ndim != 1
        or not len(a)
        or not np.isfinite(p).all()
        or not np.isfinite(a).all()
        or np.any(np.abs(a) > 90)
    ):
        raise ValueError(
            "Finite 1D wavelength positions and angles in [-90,90] required"
        )
    return np.exp(
        2j * np.pi * (p - p[0])[:, None] * np.sin(np.deg2rad(a))[None, :]
    ) / np.sqrt(len(p))


def diagnose(positions, angles, min_separation=15.0, threshold=0.95):
    a = np.asarray(angles, dtype=float)
    if (
        not np.isfinite([min_separation, threshold]).all()
        or min_separation <= 0
        or not 0 < threshold <= 1
    ):
        raise ValueError(
            "Positive separation and coherence threshold in (0,1] required"
        )
    manifold = steering(positions, a)
    if len(a) < 2 or np.any(np.diff(a) <= 0):
        raise ValueError("Increasing angle grid required")
    coherence = np.clip(np.abs(manifold.conj().T @ manifold), 0, 1)
    eligible = np.abs(a[:, None] - a[None, :]) >= min_separation
    if not eligible.any():
        raise ValueError("No pairs meet separation criterion")
    scores = np.where(eligible, coherence, -np.inf)
    competitor = np.argmax(scores, axis=1)
    maxima = scores[np.arange(len(a)), competitor]
    # Individual angles may have no eligible competitor.
    competitor_angle = np.where(np.isfinite(maxima), a[competitor], np.nan)
    maxima = np.where(np.isfinite(maxima), maxima, np.nan)
    return {
        "coherence": coherence,
        "competitor_deg": competitor_angle,
        "max_coherence": maxima,
        "ambiguous": np.nan_to_num(maxima, nan=-1) >= threshold,
    }


def peak_candidates(snapshot, positions, angles, relative_power=0.99):
    x = np.asarray(snapshot, dtype=complex)
    d = steering(positions, angles)
    if (
        x.shape != (d.shape[0],)
        or not np.isfinite(x).all()
        or not np.isfinite(relative_power)
        or not 0 < relative_power <= 1
    ):
        raise ValueError("Invalid snapshot or relative power threshold")
    # Complex magnitudes can overflow despite both components being finite.
    scale = max(np.max(np.abs(x.real)), np.max(np.abs(x.imag)))
    if scale == 0:
        return np.array([], dtype=float)
    scaled = x.real / scale + 1j * (x.imag / scale)
    p = np.abs(d.conj().T @ scaled) ** 2
    local = (p >= np.r_[-np.inf, p[:-1]]) & (p >= np.r_[p[1:], -np.inf])
    return np.asarray(angles)[local & (p >= relative_power * p.max())]
