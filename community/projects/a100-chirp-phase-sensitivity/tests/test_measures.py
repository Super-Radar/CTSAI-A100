"""Tests for measures and theory anchors (synthetic benchmarks)."""

import numpy as np

from phase_sens.measures import (
    circular_stats,
    coherent_gain,
    coherent_gain_matrix,
    coherent_gain_weighted,
    demodulate_peak,
    doppler_metrics,
    doppler_metrics_trials,
    theoretical_gain_ar1,
    theoretical_gain_iid,
    theoretical_gain_rw,
    theoretical_gain_weighted_iid,
)
from phase_sens.pipeline import SENSOR_CONFIGS, hann_periodic
from phase_sens.perturb import apply_phase_perturbation


def _tone_perturbed(rng, sigma, n=256, m=0, trials=400):
    base = 2 * np.pi * m * np.arange(n) / n
    phis = rng.normal(0.0, sigma, size=(trials, n)) + base[None, :]
    return np.exp(1j * phis)


def test_gain_bounds_and_coherent_reference():
    seq = np.exp(1j * np.linspace(0, 0, 64))  # all phases equal
    assert np.isclose(coherent_gain(seq), 1.0)
    rng = np.random.default_rng(21)
    seq = rng.normal(size=64) + 1j * rng.normal(size=64)
    g = coherent_gain(seq)
    assert 0.0 < g <= 1.0


def test_gain_iid_matches_theory():
    rng = np.random.default_rng(22)
    for sigma in (0.1, 0.3, 0.6, 1.0):
        g = coherent_gain_matrix(_tone_perturbed(rng, sigma)).mean()
        assert abs(g - theoretical_gain_iid(sigma, 256)) < 0.01


def test_gain_weighted_matches_theory():
    rng = np.random.default_rng(23)
    win = hann_periodic(256)
    for sigma in (0.2, 0.7, 1.2):
        # windowed sum only stays aligned for a DC tone (m=0)
        g = coherent_gain_matrix(_tone_perturbed(rng, sigma, m=0), win).mean()
        assert abs(g - theoretical_gain_weighted_iid(sigma, win)) < 0.01


def test_gain_floor_at_large_sigma():
    rng = np.random.default_rng(24)
    n = 128
    g = coherent_gain_matrix(
        np.exp(1j * rng.normal(0, 5.0, size=(400, n)))).mean()
    assert abs(g - 1.0 / n) < 0.3 / n  # E[G] -> 1/N, MC tolerance
    assert g > 0


def test_gain_ar1_theory_both_parameterisations():
    rng = np.random.default_rng(25)
    from phase_sens.perturb import (
        ar1_phase_innovation_batch, ar1_phase_marginal_batch)
    n, trials = 256, 600
    for a, sigma in ((0.5, 0.6), (0.95, 0.5)):
        phis = ar1_phase_marginal_batch(rng, trials, n, a, sigma)
        g = coherent_gain_matrix(np.exp(1j * phis)).mean()
        assert abs(g - theoretical_gain_ar1(sigma, a, n, "marginal")) < 0.012
    phis = ar1_phase_innovation_batch(rng, trials, n, 0.9, 0.12)
    g = coherent_gain_matrix(np.exp(1j * phis)).mean()
    assert abs(g - theoretical_gain_ar1(0.12, 0.9, n, "innovation")) < 0.015


def test_gain_rw_matches_theory():
    rng = np.random.default_rng(26)
    from phase_sens.perturb import random_walk_phase_batch
    n, trials = 256, 3000
    for step in (0.03, 0.1, 0.25):
        phis = random_walk_phase_batch(rng, trials, n, step)
        g = coherent_gain_matrix(np.exp(1j * phis)).mean()
        assert abs(g - theoretical_gain_rw(step, n)) < 0.03


def test_doppler_peak_degrades_with_sigma():
    cfg = SENSOR_CONFIGS["init0"]
    rng = np.random.default_rng(27)
    n, m = cfg.nchirp, 32
    seq = np.exp(1j * 2 * np.pi * m * np.arange(n) / n)
    ratios = []
    for sigma in (0.1, 1.0):
        phis = rng.normal(0.0, sigma, size=(60, n))
        pert = seq[None, :] * np.exp(1j * phis)
        met = doppler_metrics_trials(pert, cfg)
        ref = doppler_metrics_trials(seq[None, :], cfg)["peak_amplitude"][0]
        ratios.append((met["peak_amplitude"] / ref).mean())
    assert ratios[1] < ratios[0] - 0.1


def test_second_moment_and_neighborhood_spread_with_sigma():
    cfg = SENSOR_CONFIGS["init0"]
    rng = np.random.default_rng(28)
    n, m = cfg.nchirp, 32
    seq = np.exp(1j * 2 * np.pi * m * np.arange(n) / n)
    m2_0 = doppler_metrics_trials(seq[None, :], cfg)["second_central_moment"][0]
    # periodic hann kernel: energy only at bins m, m+-1 with weights
    # (2/3, 1/6, 1/6) -> second central moment = 2*(1/6)*1^2 = 1/3
    assert abs(m2_0 - 1.0 / 3.0) < 1e-6
    phis = rng.normal(0.0, 1.0, size=(60, n))
    met = doppler_metrics_trials(seq[None, :] * np.exp(1j * phis), cfg)
    assert met["second_central_moment"].mean() > 1.0
    assert met["neighborhood_energy_ratio"].mean() < 0.9
    ref_nbhd = doppler_metrics_trials(seq[None, :], cfg)["neighborhood_energy_ratio"][0]
    assert met["neighborhood_energy_ratio"].mean() < ref_nbhd


def test_peak_bin_stays_at_tone_bin_for_moderate_sigma():
    cfg = SENSOR_CONFIGS["init0"]
    rng = np.random.default_rng(29)
    n, m = cfg.nchirp, 32
    seq = np.exp(1j * 2 * np.pi * m * np.arange(n) / n)
    phis = rng.normal(0.0, 0.3, size=(50, n))
    met = doppler_metrics_trials(seq[None, :] * np.exp(1j * phis), cfg)
    assert np.mean(met["peak_bin"] == m) > 0.9


def test_single_and_batched_metrics_agree():
    cfg = SENSOR_CONFIGS["init1"]
    rng = np.random.default_rng(30)
    seq = rng.normal(size=cfg.nchirp) + 1j * rng.normal(size=cfg.nchirp)
    one = doppler_metrics(seq, cfg)
    batch = doppler_metrics_trials(seq[None, :], cfg)
    assert np.isclose(one["peak_amplitude"], batch["peak_amplitude"][0])
    assert one["peak_bin"] == int(batch["peak_bin"][0])
    assert np.isclose(one["second_central_moment"],
                      batch["second_central_moment"][0], rtol=1e-12)


def test_circular_stats_of_known_distribution():
    rng = np.random.default_rng(31)
    ang = rng.normal(0.0, 0.5, size=200000)
    stats = circular_stats(ang)
    assert abs(stats["mean_angle_rad"]) < 0.01
    # circular std for a wrapped normal with small sigma ~ linear sigma
    assert abs(stats["circular_std_rad"] - 0.5) < 0.01


def test_demodulate_peak_restores_alignment():
    rng = np.random.default_rng(33)
    n, m = 256, 32
    tone = np.exp(1j * (2 * np.pi * m * np.arange(n) / n
                        + rng.normal(0, 0.2, size=n)))
    # plain sum of a non-zero-bin tone is incoherent; after demodulation the
    # gain follows the DC-tone theory
    assert coherent_gain(tone) < 0.1
    g = coherent_gain(demodulate_peak(tone))
    assert abs(g - theoretical_gain_iid(0.2, n)) < 0.01


def test_weighted_gain_is_window_dependent():
    seq = np.ones(64)  # fully coherent
    win = hann_periodic(64)
    assert np.isclose(coherent_gain_weighted(seq, np.ones(64)), 1.0)
    assert np.isclose(coherent_gain_weighted(seq, win), 1.0)
    # for incoherent sequences the hann floor r = sum w^2 / (sum w)^2 is
    # higher than the rectangular floor 1/N
    rng = np.random.default_rng(32)
    phis = rng.normal(0, 10, size=(400, 64))
    seqs = np.exp(1j * phis)
    g_w = coherent_gain_matrix(seqs, win).mean()
    r = np.sum(win ** 2) / np.sum(win) ** 2
    assert abs(g_w - r) < 0.1 * r
