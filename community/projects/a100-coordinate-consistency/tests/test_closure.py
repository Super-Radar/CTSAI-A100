"""Closure-residual tests: known-convention recovery, numeric samples,
abstention on unidentifiable input, angle-offset scan."""

import numpy as np
import pytest

from coord_consistency.closure import (
    ABSTAIN,
    evaluate_file,
    is_identifiable,
    residual_stats,
    row_residuals,
    scan_angle_offset,
)
from coord_consistency.hypotheses import predict


def _synthetic_h1_rows(angles_deg, ranges, noise=0.01):
    """Rows that follow H1, quantised to `noise` like logged CSV values."""
    r = np.asarray(ranges, dtype=float)
    a = np.asarray(angles_deg, dtype=float)
    x, y = predict("H1", r, a)
    rng = np.random.default_rng(20261003)
    x = np.round(x + rng.uniform(-noise / 2, noise / 2, x.shape), 3)
    y = np.round(y + rng.uniform(-noise / 2, noise / 2, y.shape), 3)
    return r, a, x, y


def test_residual_formula_numeric_sample():
    # H1 at r=10, theta=0 predicts (0, 10); logged (0.1, 10.2)
    res = row_residuals([10.0], [0.0], [0.1], [10.2], "H1")
    assert res["dx"] == pytest.approx(0.1)
    assert res["dy"] == pytest.approx(0.2)
    assert res["dist"] == pytest.approx(np.hypot(0.1, 0.2))  # 0.2236...


def test_residual_stats_values():
    stats = residual_stats({"dx": [-1.0, 1.0, 2.0, -3.0],
                            "dy": [0.0, 0.0, 0.0, 0.0],
                            "dist": [1.0, 2.0, 3.0, 4.0]})
    assert stats["median_m"] == 2.5
    assert stats["p95_m"] == pytest.approx(np.percentile([1, 2, 3, 4], 95))
    assert stats["max_m"] == 4.0
    assert stats["median_dx_m"] == pytest.approx(1.5)  # |{-1,1,2,-3}| median
    assert stats["median_dy_m"] == 0.0


def test_known_convention_recovers_h1():
    angles = [0.0, 30.78, -45.0, 90.0, 170.0, -170.0, 65.5, -12.25]
    ranges = [12.14, 25.0, 3.5, 40.0, 8.8, 8.8, 55.0, 1.25]
    r, a, x, y = _synthetic_h1_rows(angles, ranges)
    ev = evaluate_file(r, a, x, y)
    assert ev["best_hyp"] == "H1"
    assert ev["second_hyp"] in ("H2", "H3", "H4")
    assert ev["margin_m"] > 1.0                       # clearly separable
    assert ev["best_stats"]["median_m"] < 0.015       # sub-noise-scale fit
    assert ev["best_stats"]["max_m"] < 0.05
    assert ev["identifiable"] is True


def test_four_quadrants_and_wrap_rows():
    # one row per quadrant plus a +-180 wraparound pair
    angles = [10.0, 100.0, -100.0, -10.0, 179.9, -179.9]
    ranges = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    r, a, x, y = _synthetic_h1_rows(angles, ranges)
    ev = evaluate_file(r, a, x, y)
    assert ev["best_hyp"] == "H1"
    assert ev["best_stats"]["median_m"] < 0.015


def test_abstain_on_all_zero_xy():
    r = np.array([5.0, 10.0, 20.0])
    a = np.array([0.0, 45.0, -30.0])
    zero = np.zeros(3)
    assert is_identifiable(zero, zero) is False
    ev = evaluate_file(r, a, zero, zero)
    assert ev["best_hyp"] == ABSTAIN
    assert ev["identifiable"] is False


def test_identifiable_accepts_real_signal():
    r, a, x, y = _synthetic_h1_rows([0.0, 45.0], [5.0, 10.0])
    assert is_identifiable(x, y) is True


def test_offset_scan_detects_known_bias():
    # logged (X, Y) consistent with H1 at (theta + 2 deg)
    r = np.array([5.0, 10.0, 20.0, 7.5])
    a = np.array([0.0, 30.0, -45.0, 175.0])
    x, y = predict("H1", r, a + 2.0)
    best_off, medians = scan_angle_offset(r, a, x, y, "H1")
    assert best_off == pytest.approx(2.0)
    assert float(np.min(medians)) < 1e-9            # biased grid point closes
    assert float(np.median(row_residuals(r, a, x, y, "H1", 0.0)["dist"])) > 0.1


def test_offset_scan_clean_data_prefers_zero():
    r, a, x, y = _synthetic_h1_rows([15.0, 40.0, -25.0], [10.0, 60.0, 120.0])
    best_off, medians = scan_angle_offset(r, a, x, y, "H1")
    assert best_off == 0.0
    ev = evaluate_file(r, a, x, y)
    assert ev["angle_offset_improves"] is False
    assert ev["angle_offset_best_deg"] == 0.0


def test_margin_is_nonnegative():
    r, a, x, y = _synthetic_h1_rows([10.0, 30.0, 55.0, -20.0],
                                    [3.0, 9.0, 33.0, 66.0])
    ev = evaluate_file(r, a, x, y)
    assert ev["margin_m"] >= 0.0
    medians = [ev["hyp_stats"][h]["median_m"] for h in ("H1", "H2", "H3", "H4")]
    assert min(medians) == ev["best_stats"]["median_m"]
