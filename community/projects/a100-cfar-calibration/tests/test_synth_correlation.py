"""Shared-component correlation synthesis checks."""

import numpy as np

from cfar_calib.pipeline import RadarConfig, synth_real_noise

CFG = RadarConfig(n_chirps=16, adc_sample_num=2048, rng_nfft=2048, vel_nfft=16)


def channel_corr(x, i, j):
    a = x[..., i].ravel() - x[..., i].mean()
    b = x[..., j].ravel() - x[..., j].mean()
    return float(np.dot(a, b) / np.sqrt(np.dot(a, a) * np.dot(b, b)))


def test_amplitude_correlation_matches_rho():
    for rho in (0.3, 0.7):
        x = synth_real_noise(np.random.default_rng(41), CFG, 4, rho)
        est = np.mean([channel_corr(x, 0, j) for j in (1, 2, 3)])
        assert abs(est - rho) < 0.03


def test_rho_one_gives_identical_channels():
    x = synth_real_noise(np.random.default_rng(42), CFG, 4, 1.0)
    for j in (1, 2, 3):
        np.testing.assert_array_equal(x[..., 0], x[..., j])


def test_rho_zero_gives_independent_channels():
    x = synth_real_noise(np.random.default_rng(43), CFG, 4, 0.0)
    for j in (1, 2, 3):
        assert abs(channel_corr(x, 0, j)) < 0.03


def test_power_domain_correlation_is_rho_squared():
    rho = 0.6
    x = synth_real_noise(np.random.default_rng(44), CFG, 2, rho)
    p = x ** 2
    est = channel_corr(p, 0, 1)
    assert abs(est - rho ** 2) < 0.05
