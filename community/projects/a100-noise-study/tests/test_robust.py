"""Hand-computed checks of the robust estimators and helpers."""

from __future__ import annotations

import numpy as np
import pytest

from noise_study.robust import (
    ESTIMATORS,
    bootstrap_ci,
    gaussian_peak_field,
    inject_gaussian_peak,
    mad,
    mad_noise_floor,
    median,
    official_mean,
    quantile,
    trimmed_mean,
)

X9 = np.arange(1, 10, dtype=float)          # [1..9]
X10 = np.arange(1, 11, dtype=float)         # [1..10]


def test_official_mean_is_plain_mean():
    assert official_mean(X9) == pytest.approx(5.0)


def test_median_and_quantiles_hand_computed():
    assert median(X9) == 5.0
    # np.quantile linear interpolation: Q0.25 of 1..9 -> 3.0, Q0.10 -> 1.8
    assert quantile(X9, 0.25) == pytest.approx(3.0)
    assert quantile(X9, 0.10) == pytest.approx(1.8)
    assert ESTIMATORS["q25"](X9) == pytest.approx(3.0)
    assert ESTIMATORS["q10"](X9) == pytest.approx(1.8)


def test_trimmed_mean_hand_computed():
    # floor(0.1*10)=1 per tail -> mean(2..9) = 5.5
    assert trimmed_mean(X10, 0.10) == pytest.approx(5.5)
    # floor(0.2*10)=2 per tail -> mean(3..8) = 5.5
    assert trimmed_mean(X10, 0.20) == pytest.approx(5.5)
    # asymmetric outlier test: [0, 1, 2, 3, 4, 5, 6, 7, 8, 1000]
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 1000.0])
    assert trimmed_mean(x, 0.10) == pytest.approx(np.mean([1, 2, 3, 4, 5, 6, 7, 8]))
    assert trimmed_mean(X9, 0.0) == pytest.approx(5.0)


def test_trimmed_mean_rejects_bad_fraction():
    with pytest.raises(ValueError):
        trimmed_mean(X9, 0.6)
    with pytest.raises(ValueError):
        trimmed_mean([1.0, 2.0], 0.5)


def test_mad_and_mad_noise_floor_hand_computed():
    # median 5, |x-5| = [4,3,2,1,0,1,2,3,4], median of that = 2
    assert mad(X9) == 2.0
    assert mad_noise_floor(X9) == pytest.approx(5.0 + 1.4826 * 2.0)


def test_estimator_ordering_under_outlier():
    """A single extreme high outlier moves the mean far more than the others."""
    x = np.concatenate([np.full(100, 10.0), [70.0]])
    dev_mean = ESTIMATORS["official_mean"](x) - 10.0
    dev_median = ESTIMATORS["median"](x) - 10.0
    dev_trim = ESTIMATORS["trim10"](x) - 10.0
    dev_mad = ESTIMATORS["mad_floor"](x) - ESTIMATORS["mad_floor"](np.full(100, 10.0))
    assert dev_mean == pytest.approx(60.0 / 101)
    assert dev_median == 0.0
    assert dev_trim == 0.0
    assert abs(dev_mad) < 1e-12


def test_estimators_accept_2d_regions():
    """Regions arrive as 2-D arrays; estimators must flatten them."""
    region = np.arange(1, 19, dtype=float).reshape(3, 6)
    flat = np.arange(1, 19, dtype=float)
    for name in ESTIMATORS:
        assert ESTIMATORS[name](region) == pytest.approx(ESTIMATORS[name](flat)), name


def test_bootstrap_ci_deterministic_and_reasonable():
    x = np.arange(1000, dtype=float)
    lo1, hi1 = bootstrap_ci(x, median, n_boot=500, seed=42)
    lo2, hi2 = bootstrap_ci(x, median, n_boot=500, seed=42)
    assert (lo1, hi1) == (lo2, hi2)          # fixed seed -> reproducible
    assert lo1 < 499.5 < hi1                 # covers the point estimate
    assert 450 < lo1 < 550 and 450 < hi1 < 550


def test_gaussian_peak_field_peak_center_and_tails():
    field = gaussian_peak_field((64, 64), [(30, 40)], amp_db=20.0,
                                sigma_rng=2.0, sigma_vel=2.0)
    assert field[30, 40] == pytest.approx(20.0, rel=1e-12)   # center == amp
    far = field[30 + 20, 40]                                  # 10 sigma away
    assert far < 1e-10
    assert field.min() >= 0.0


def test_inject_gaussian_peak_does_not_mutate_input():
    rv = np.zeros((32, 32))
    out = inject_gaussian_peak(rv, [(16, 16)], 30.0, 1.5, 1.5)
    assert rv.sum() == 0.0
    assert out[16, 16] == pytest.approx(30.0)
    assert out.sum() > 30.0
