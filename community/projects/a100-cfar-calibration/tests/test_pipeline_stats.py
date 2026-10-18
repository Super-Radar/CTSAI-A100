"""Chain output statistics: exponential single-look, Gamma(K) after averaging."""

import numpy as np

from cfar_calib.cfar import interior_mask
from cfar_calib.pipeline import RadarConfig, range_doppler_power, synth_real_noise

CFG = RadarConfig(n_chirps=32, adc_sample_num=256, rng_nfft=256, vel_nfft=32)
FRAMES = 8


def cell_stats(k, rho, seed=51):
    rng = np.random.default_rng(seed)
    cells = []
    for _ in range(FRAMES):
        power = range_doppler_power(synth_real_noise(rng, CFG, k, rho), CFG)
        cells.append(power[interior_mask(power.shape)])
    cells = np.concatenate(cells)
    mean = cells.mean()
    # noise-only map -> normalised second central moment: 1.0 for K=1, 1/K for K
    cv2 = ((cells - mean) ** 2).mean() / mean ** 2
    return mean, cv2, cells.size


def test_output_shape_and_finiteness():
    power = range_doppler_power(
        synth_real_noise(np.random.default_rng(50), CFG, 4, 0.0), CFG)
    assert power.shape == (CFG.rng_bins, CFG.vel_nfft)
    assert np.isfinite(power).all()
    assert (power > 0).all()


def test_single_channel_power_is_exponential():
    _, cv2, n = cell_stats(1, 0.0)
    assert n > 20_000
    assert abs(cv2 - 1.0) < 0.10, cv2


def test_k4_independent_average_is_gamma4():
    _, cv2, n = cell_stats(4, 0.0)
    assert n > 20_000
    assert abs(cv2 - 0.25) < 0.06, cv2


def test_k4_fully_correlated_collapses_to_single_channel():
    _, cv2, n = cell_stats(4, 1.0)
    assert abs(cv2 - 1.0) < 0.12, cv2
