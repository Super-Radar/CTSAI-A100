"""Hypothesis-family tests: reference measurement, quadrants, +-180 wrap."""

import pytest

from coord_consistency.hypotheses import (
    ANGLE_OFFSET_GRID,
    HYPOTHESES,
    HYPOTHESIS_ORDER,
    predict,
    wrap180,
)


def test_h1_reproduces_reference_measurement():
    # session 2026_05_22_13_41_37 TraTarget: r=12.14, angle=30.78 deg
    # logged X=-6.212, Y=10.425
    x, y = predict("H1", 12.14, 30.78)
    assert x == pytest.approx(-6.2126, abs=5e-4)
    assert y == pytest.approx(10.4299, abs=5e-4)
    # closure against the logged pair is sub-centimetre
    assert pytest.approx(x, abs=1e-3) == -6.212
    assert pytest.approx(y, abs=6e-3) == 10.425


def test_h1_quadrant_signs():
    eps = 1e-12
    x, y = predict("H1", 1.0, 0.0)
    assert abs(x) < eps and y == pytest.approx(1.0)
    x, y = predict("H1", 1.0, 90.0)
    assert x == pytest.approx(-1.0) and abs(y) < eps
    x, y = predict("H1", 1.0, -90.0)
    assert x == pytest.approx(1.0) and abs(y) < eps
    x, y = predict("H1", 1.0, 180.0)
    assert abs(x) < eps and y == pytest.approx(-1.0)


def test_wrap180_mapping():
    assert wrap180(180.0) == -180.0
    assert wrap180(-180.0) == -180.0
    assert wrap180(190.0) == -170.0
    assert wrap180(-190.0) == 170.0
    assert wrap180(30.0) == 30.0
    assert wrap180(360.0) == 0.0


def test_pm180_wrap_equivalence():
    # angles 360 deg apart map to the same predicted point
    pairs = [(180.0, -180.0), (179.9, -180.1), (180.1, -179.9),
             (30.0, 390.0), (-170.0, 190.0)]
    for a1, a2 in pairs:
        xa, ya = predict("H1", 7.5, a1)
        xb, yb = predict("H1", 7.5, a2)
        assert xa == pytest.approx(xb, abs=1e-9)
        assert ya == pytest.approx(yb, abs=1e-9)


def test_h2_h3_h4_numeric_values():
    r, th = 2.0, 30.0
    x, y = predict("H2", r, th)
    assert x == pytest.approx(1.0) and y == pytest.approx(3 ** 0.5)
    x, y = predict("H3", r, th)
    assert x == pytest.approx(3 ** 0.5) and y == pytest.approx(1.0)
    x, y = predict("H4", r, th)
    assert x == pytest.approx(-3 ** 0.5) and y == pytest.approx(1.0)


def test_angle_offset_shifts_prediction():
    x, y = predict("H1", 1.0, 0.0, angle_offset_deg=90.0)
    xr, yr = predict("H1", 1.0, 90.0)
    assert x == pytest.approx(xr) and y == pytest.approx(yr)


def test_family_metadata():
    assert HYPOTHESIS_ORDER == ("H1", "H2", "H3", "H4")
    assert set(HYPOTHESES) == set(HYPOTHESIS_ORDER)
    assert ANGLE_OFFSET_GRID[0] == -5.0 and ANGLE_OFFSET_GRID[-1] == 5.0
    assert len(ANGLE_OFFSET_GRID) == 21  # 0.5 deg steps inclusive
    assert float(ANGLE_OFFSET_GRID[10]) == 0.0


def test_unknown_hypothesis_rejected():
    with pytest.raises(KeyError):
        predict("H9", 1.0, 0.0)
