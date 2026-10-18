"""Effective-sample-size and CUSUM segmentation tests (fixed-seed Monte Carlo)."""

import numpy as np

from temporal_ci.effective_n import (
    CUSUM_THRESHOLD,
    analyze_effective_n,
    autocorr,
    cusum_change_points,
    max_lag_for,
    segments_of,
)


def _ar1(rng, n, phi):
    e = rng.standard_normal(n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1.0 - phi * phi)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + e[t]
    return x


def test_constant_series_is_unusable_not_a_ci():
    r = analyze_effective_n(np.full(200, 7.0))
    assert r["status"] == "zero_variance"
    assert r["neff"] is None
    # constant series also yields no CUSUM split (zero robust scale)
    assert cusum_change_points(np.full(200, 7.0)) == []


def test_iid_neff_close_to_n():
    rng = np.random.default_rng(11)
    r = analyze_effective_n(rng.standard_normal(4000))
    assert r["status"] == "ok"
    assert abs(r["neff_ratio"] - 1.0) < 0.2


def test_ar1_neff_matches_analytic_formula():
    for phi in (0.5, 0.8):
        rng = np.random.default_rng(20261003 + int(10 * phi))
        ests = [analyze_effective_n(_ar1(rng, 4000, phi))["neff"] for _ in range(10)]
        analytic = 4000.0 * (1.0 - phi) / (1.0 + phi)
        assert abs(np.mean(ests) / analytic - 1.0) < 0.15


def test_max_lag_rule():
    assert max_lag_for(1000) == 50
    assert max_lag_for(4000) == 100  # capped
    assert max_lag_for(100) == 5


def test_cusum_detects_known_breakpoint():
    rng = np.random.default_rng(5)
    x = np.concatenate([rng.standard_normal(200), rng.standard_normal(200) + 3.0])
    cps = cusum_change_points(x)
    assert len(cps) == 1
    assert abs(cps[0] - 200) <= 20


def test_cusum_no_false_positive_on_iid():
    rng = np.random.default_rng(7)
    assert cusum_change_points(rng.standard_normal(600)) == []


def test_short_segment_reports_insufficient_samples():
    r = analyze_effective_n(np.random.default_rng(3).standard_normal(25))
    assert r["status"] == "insufficient_samples"
    segs = segments_of(np.arange(40, dtype=float), cps=[35])
    assert segs[-1]["status"] == "insufficient_samples" and segs[-1]["n"] == 5


def test_autocorr_rho0_and_bounds():
    rng = np.random.default_rng(13)
    rho = autocorr(rng.standard_normal(500), max_lag=20)
    assert rho.size == 21 and rho[0] == 1.0
    assert np.all(np.abs(rho) <= 1.0 + 1e-9)


def test_threshold_is_documented_one_percent_value():
    assert CUSUM_THRESHOLD == 1.63
