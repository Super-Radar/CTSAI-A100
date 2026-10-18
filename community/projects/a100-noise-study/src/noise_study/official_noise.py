"""Line-faithful re-implementation of ``FFT/calculate_global_noise.m``.

The official estimator (source:
``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/FFT/calculate_global_noise.m``)
is:

1. ``RVimage`` = channel-averaged power ``mean(|Radar_cube|²)`` over the
   ``nvirtual_array`` RX channels, converted to dB via
   ``db(..., 'power')`` == ``10*log10``;
2. noise region along **range**: bins ``round(0.75*rng_nfft/2) .. round(0.8*rng_nfft/2)``
   (MATLAB 1-based, inclusive; identical for both velocity sub-regions);
3. noise region along **velocity**: two half-bands
   ``6 .. round(vel_nfft/2 - 6)`` and ``round(vel_nfft/2 + 6) .. round(vel_nfft - 6)``;
4. the two half-band sub-images are combined with an **element-wise min**;
5. ``noiseValue`` = mean over the whole combined region.

For ``sensor_config_init0`` (rng_nfft 1024 / vel_nfft 256) this is range bins
384..410 (MATLAB 1-based) and velocity bins 6..122 and 134..250, i.e. a
27 x 117 region; for ``init1`` (2048/128) range bins 768..819 and velocity
bins 6..58 and 70..122, i.e. 52 x 53 cells.

All indices handled with MATLAB round-half-away-from-zero semantics and
explicit 1-based -> 0-based conversion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .pipeline import SensorConfig, matlab_round, power_to_db, rv_power


@dataclass(frozen=True)
class NoiseRegionBounds:
    """1-based inclusive MATLAB indices of the official noise region."""

    rng_start: int
    rng_end: int
    vel1_start: int
    vel1_end: int
    vel2_start: int
    vel2_end: int

    @property
    def rng_size(self) -> int:
        return self.rng_end - self.rng_start + 1

    @property
    def vel_size(self) -> int:
        return self.vel1_end - self.vel1_start + 1

    @property
    def region_cells(self) -> int:
        return self.rng_size * self.vel_size

    # 0-based half-open Python slices
    @property
    def rng_slice(self) -> slice:
        return slice(self.rng_start - 1, self.rng_end)

    @property
    def vel1_slice(self) -> slice:
        return slice(self.vel1_start - 1, self.vel1_end)

    @property
    def vel2_slice(self) -> slice:
        return slice(self.vel2_start - 1, self.vel2_end)

    def summary(self) -> dict:
        return {
            "range_bins_matlab_1based": f"{self.rng_start}..{self.rng_end}",
            "vel_half1_bins_matlab_1based": f"{self.vel1_start}..{self.vel1_end}",
            "vel_half2_bins_matlab_1based": f"{self.vel2_start}..{self.vel2_end}",
            "region_cells": self.rng_size * self.vel_size,
        }


def noise_region_bounds(cfg: SensorConfig) -> NoiseRegionBounds:
    """Compute the official noise-region indices exactly as the .m file."""
    half_rng = cfg.rng_nfft / 2
    return NoiseRegionBounds(
        rng_start=matlab_round(0.75 * half_rng),
        rng_end=matlab_round(0.8 * half_rng),
        vel1_start=6,
        vel1_end=matlab_round(cfg.vel_nfft / 2 - 6),
        vel2_start=matlab_round(cfg.vel_nfft / 2 + 6),
        vel2_end=matlab_round(cfg.vel_nfft - 6),
    )


def official_region(rv_db: np.ndarray, cfg: SensorConfig) -> np.ndarray:
    """Element-wise min of the two velocity half-band sub-images (step 4)."""
    b = noise_region_bounds(cfg)
    region1 = rv_db[b.rng_slice, b.vel1_slice]
    region2 = rv_db[b.rng_slice, b.vel2_slice]
    if region1.shape != (b.rng_size, b.vel_size) or region2.shape != region1.shape:
        raise ValueError("noise region indices out of bounds for the given image")
    # MATLAB computes noiseRegion(i, j) = min(noiseRegion1(i, j), noiseRegion2(i, j))
    return np.minimum(region1, region2)


def calculate_global_noise(vel_fft: np.ndarray, cfg: SensorConfig) -> float:
    """Official global noise value in dB, from the Radar_cube (velFftData).

    ``vel_fft`` shape: ``(rng_nfft/2, vel_nfft, nvirtual_array)`` complex.
    """
    rv_db = power_to_db(rv_power(vel_fft))
    return float(np.mean(official_region(rv_db, cfg)))


def calculate_noise_from_rv_db(rv_db: np.ndarray, cfg: SensorConfig) -> float:
    """Same estimator on an already channel-averaged dB RV image."""
    return float(np.mean(official_region(rv_db, cfg)))


def region_mean(rv_db: np.ndarray, rng_slice: slice, vel_slices) -> float:
    """Official-style mean of dB values over an arbitrary region.

    Used by the region-scan experiment.  ``vel_slices`` is either a single
    slice (single band) or a pair of slices combined with the element-wise
    min like the official method.
    """
    if isinstance(vel_slices, (tuple, list)):
        s1, s2 = vel_slices
        region = np.minimum(rv_db[rng_slice, s1], rv_db[rng_slice, s2])
    else:
        region = rv_db[rng_slice, vel_slices]
    return float(np.mean(region))
