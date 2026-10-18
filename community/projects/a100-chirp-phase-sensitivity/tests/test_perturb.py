"""Tests for the phase perturbation models."""

import numpy as np

from phase_sens.measures import coherent_gain, coherent_gain_matrix
from phase_sens.pipeline import SENSOR_CONFIGS, doppler_fft, hann_periodic
from phase_sens.perturb import (
    apply_phase_perturbation,
    ar1_phase_marginal,
    ar1_phase_marginal_batch,
    correlation_length,
    iid_gaussian_phase,
    random_walk_phase,
    random_walk_phase_batch,
)


def test_zero_perturbation_is_identity():
    rng = np.random.default_rng(3)
    seq = (rng.normal(size=64) + 1j * rng.normal(size=64))
    out = apply_phase_perturbation(seq, np.zeros(64))
    # gain unchanged and Doppler spectrum unchanged (bitwise)
    assert coherent_gain(out) == coherent_gain(seq)
    cfg = SENSOR_CONFIGS["init1"]
    seq = rng.normal(size=cfg.nchirp) + 1j * rng.normal(size=cfg.nchirp)
    assert np.array_equal(
        doppler_fft(apply_phase_perturbation(seq, np.zeros(cfg.nchirp)), cfg),
        doppler_fft(seq, cfg),
    )


def test_pure_phase_rotation_preserves_amplitudes_and_energy():
    rng = np.random.default_rng(4)
    seq = rng.normal(size=128) + 1j * rng.normal(size=128)
    phi = rng.uniform(-np.pi, np.pi, size=128)
    out = apply_phase_perturbation(seq, phi)
    assert out.shape == seq.shape
    assert np.allclose(np.abs(out), np.abs(seq), rtol=1e-12)
    total_before = np.sum(np.abs(seq) ** 2)
    total_after = np.sum(np.abs(out) ** 2)
    assert np.isclose(total_after, total_before, rtol=1e-12)


def test_ar1_lag1_correlation_matches_parameter():
    rng = np.random.default_rng(5)
    a = 0.85
    phi = ar1_phase_marginal(rng, 20000, a, sigma=0.4)
    emp = np.corrcoef(phi[:-1], phi[1:])[0, 1]
    assert abs(emp - a) < 0.03
    # stationary marginal std also matches sigma
    assert abs(np.std(phi, ddof=1) - 0.4) < 0.02


def test_ar1_marginal_std_across_a_values():
    rng = np.random.default_rng(6)
    for a in (0.0, 0.9, 0.99):
        phi = ar1_phase_marginal(rng, 10000, a, sigma=0.7)
        assert abs(np.std(phi, ddof=1) - 0.7) < 0.03


def test_random_walk_increment_std():
    rng = np.random.default_rng(7)
    phi = random_walk_phase(rng, 50000, sigma_step=0.11)
    assert abs(np.std(np.diff(phi), ddof=1) - 0.11) < 0.005


def test_correlation_length_formula():
    assert correlation_length(0.0) == 1.0
    assert np.isclose(correlation_length(0.99), 100.0)


def test_correlated_phase_loses_less_at_same_marginal_sigma():
    # fixed marginal sigma: stronger correlation => higher coherent gain
    rng = np.random.default_rng(8)
    n, sigma, trials = 256, 0.5, 300
    iid_phis = rng.normal(0.0, sigma, size=(trials, n))
    ar_phis = ar1_phase_marginal_batch(rng, trials, n, 0.99, sigma)
    base = np.zeros(n)
    g_iid = coherent_gain_matrix(np.exp(1j * (iid_phis + base)))
    g_ar = coherent_gain_matrix(np.exp(1j * (ar_phis + base)))
    assert g_ar.mean() > g_iid.mean() + 0.02


def test_iid_zero_sigma_returns_zero_phase():
    rng = np.random.default_rng(9)
    phi = iid_gaussian_phase(rng, 32, 0.0)
    assert np.all(phi == 0.0)
    phi_rw = random_walk_phase_batch(rng, 4, 32, 0.0)
    assert np.all(phi_rw == 0.0)
