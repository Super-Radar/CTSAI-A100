"""Regression tests for the official calculate_global_noise re-implementation.

The reference behaviour is
``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/FFT/calculate_global_noise.m``;
each test computes the expected value with an independent plain-Python
transcription of the .m file (explicit loops) rather than calling the code
under test.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from noise_study.official_noise import (
    calculate_global_noise,
    calculate_noise_from_rv_db,
    noise_region_bounds,
    official_region,
)
from noise_study.pipeline import SensorConfig, matlab_round, power_to_db


# ---------------------------------------------------------------------------
# MATLAB round semantics
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [(409.6, 410), (768.0, 768), (819.2, 819), (0.5, 1), (2.5, 3), (1.5, 2),
     (-0.5, -1), (-2.5, -3), (-1.4, -1)],
)
def test_matlab_round_rounds_half_away_from_zero(value, expected):
    assert matlab_round(value) == expected


# ---------------------------------------------------------------------------
# Region bounds for the two public profiles
# ---------------------------------------------------------------------------


def make_cfg(rng_nfft: int, vel_nfft: int) -> SensorConfig:
    return SensorConfig(
        fmcw_startfreq=76.3,
        fmcw_bandwidth=300,
        fmcw_chirp_rampup=43,
        fmcw_chirp_period=48,
        tx_groups=[0, 0, 1, 0],
        nchirp=vel_nfft,
        adc_freq=25,
        dec_factor=1,
        adc_sample_start=2,
        adc_sample_end=42.8,
        rng_win=2,
        vel_win=2,
        rng_nfft=rng_nfft,
        vel_nfft=vel_nfft,
        rng_fft_scalar=255,
        vel_fft_scalar=63,
    )


def test_noise_region_bounds_init0():
    b = noise_region_bounds(make_cfg(1024, 256))
    assert (b.rng_start, b.rng_end) == (384, 410)      # round(0.75*512), round(0.8*512)
    assert (b.vel1_start, b.vel1_end) == (6, 122)      # 6, round(128-6)
    assert (b.vel2_start, b.vel2_end) == (134, 250)    # round(128+6), round(256-6)
    assert b.rng_size == 27 and b.vel_size == 117
    assert b.region_cells == 27 * 117


def test_noise_region_bounds_init1():
    b = noise_region_bounds(make_cfg(2048, 128))
    assert (b.rng_start, b.rng_end) == (768, 819)      # round(0.75*1024), round(0.8*1024)
    assert (b.vel1_start, b.vel1_end) == (6, 58)
    assert (b.vel2_start, b.vel2_end) == (70, 122)
    assert b.rng_size == 52 and b.vel_size == 53


# ---------------------------------------------------------------------------
# Independent .m transcription used as the oracle
# ---------------------------------------------------------------------------


def matlab_reference_noise(rv_db, rng_nfft, vel_nfft):
    """Plain-Python transcription of calculate_global_noise.m (1-based loops)."""
    # RVimage conversion is done by the caller; here rv_db is already dB.
    r1s, r1e = matlab_round(0.75 * rng_nfft / 2), matlab_round(0.8 * rng_nfft / 2)
    v1s, v1e = 6, matlab_round(vel_nfft / 2 - 6)
    v2s, v2e = matlab_round(vel_nfft / 2 + 6), matlab_round(vel_nfft - 6)
    rng_size = len(range(r1s, r1e + 1))
    vel_size = len(range(v1s, v1e + 1))
    total, count = 0.0, 0
    for i in range(rng_size):
        for j in range(vel_size):
            a = rv_db[r1s - 1 + i][v1s - 1 + j]
            b = rv_db[r1s - 1 + i][v2s - 1 + j]
            total += min(a, b)
            count += 1
    return total / count


def test_official_region_matches_matlab_transcription():
    rng = np.random.default_rng(7)
    cfg = make_cfg(1024, 256)
    rv_db = rng.normal(20.0, 3.0, size=(cfg.rng_nfft // 2, cfg.vel_nfft))
    expected = matlab_reference_noise(rv_db.tolist(), cfg.rng_nfft, cfg.vel_nfft)
    assert calculate_noise_from_rv_db(rv_db, cfg) == pytest.approx(expected, rel=1e-12)


def test_min_of_halves_is_elementwise():
    cfg = make_cfg(1024, 256)
    b = noise_region_bounds(cfg)
    rv_db = np.zeros((cfg.rng_nfft // 2, cfg.vel_nfft))
    rv_db[b.rng_slice, b.vel1_slice] = 10.0
    rv_db[b.rng_slice, b.vel2_slice] = 5.0
    region = official_region(rv_db, cfg)
    assert np.all(region == 5.0)
    assert calculate_noise_from_rv_db(rv_db, cfg) == 5.0


def test_calculate_global_noise_from_radar_cube_constant():
    """Constant-magnitude Radar_cube -> noiseValue == 10*log10(mean power)."""
    cfg = make_cfg(1024, 256)
    magnitudes = [2.0, 3.0, 4.0, 5.0]  # per channel |X|
    cube = np.empty((cfg.rng_nfft // 2, cfg.vel_nfft, 4), dtype=complex)
    for ch, m in enumerate(magnitudes):
        cube[:, :, ch] = m * np.exp(1j * 0.3)
    expected = 10.0 * math.log10(np.mean(np.square(magnitudes)))
    assert calculate_global_noise(cube, cfg) == pytest.approx(expected, rel=1e-12)


def test_calculate_global_noise_known_cells():
    """Hand-built sparse cube: only region cells carry power."""
    cfg = make_cfg(1024, 256)
    b = noise_region_bounds(cfg)
    # background 1 unit power per channel; boost two aligned cells by 9x
    cube = np.ones((cfg.rng_nfft // 2, cfg.vel_nfft, 4), dtype=complex)
    r0 = b.rng_slice.start
    for v in (b.vel1_slice.start, b.vel2_slice.start):
        cube[r0, v, :] = 3.0  # power 9
    region = official_region(power_to_db(np.mean(np.abs(cube) ** 2, axis=2)), cfg)
    # min-combined region has exactly one 9x cell (both halves boosted)
    expected_cells = 27 * 117
    expected_mean = (
        (expected_cells - 1) * 10 * math.log10(1.0) + 10 * math.log10(9.0)
    ) / expected_cells
    assert float(np.mean(region)) == pytest.approx(expected_mean, rel=1e-12)
