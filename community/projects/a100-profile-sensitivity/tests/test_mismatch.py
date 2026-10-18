"""Deterministic apparent-position checks: correct vs crossed profile."""

import pytest

from profile_sensitivity.mismatch import (predict_apparent, run_synthetic,
                                          synth_adc_words)
from profile_sensitivity.pipeline import dominant_peak, load_cube, rv_image
from profile_sensitivity.profiles import DEFAULT_HXX_DIR, load_profiles

requires_hxx = pytest.mark.skipif(
    not DEFAULT_HXX_DIR.exists(), reason="official hxx configs not in checkout")


@pytest.fixture(scope="module")
def profs():
    if not DEFAULT_HXX_DIR.exists():
        pytest.skip("official hxx configs not in checkout")
    return load_profiles()


def _measured(profs, cap_name, ass_name, target):
    """Synthesize with ``cap`` but reshape/process with ``ass`` (the trap)."""
    cap, ass = profs[cap_name], profs[ass_name]
    words = synth_adc_words(cap, [(*target, 20000.0)], seed=20260929)
    cube = load_cube(words, ass)
    return dominant_peak(rv_image(cube, ass), ass)


@requires_hxx
@pytest.mark.parametrize("cap_name,ass_name", [
    ("init1", "init0"),   # near data read as far
    ("init0", "init1"),   # far data read as near
])
def test_cross_mismatch_range_is_deterministic_ratio(profs, cap_name, ass_name):
    """R_app = R_true * S_cap/S_ass, to within one range bin."""
    target = (8.0, 2.0)
    pk = _measured(profs, cap_name, ass_name, target)
    pred = predict_apparent(profs[cap_name], profs[ass_name], *target)
    assert abs(pk["range_m"] - pred["range_m"]) <= profs[ass_name].rng_res_1dfft_m


@requires_hxx
@pytest.mark.parametrize("cap_name,ass_name", [
    ("init1", "init0"),
    ("init0", "init1"),
])
def test_cross_mismatch_velocity_matches_prediction(profs, cap_name, ass_name):
    """v_app matches the (jump-aware) deterministic prediction, <1 Doppler bin."""
    target = (8.0, 2.0)
    pk = _measured(profs, cap_name, ass_name, target)
    pred = predict_apparent(profs[cap_name], profs[ass_name], *target)
    assert abs(pk["vel_mps"] - pred["vel_mps"]) <= profs[ass_name].v_acu_mps


@requires_hxx
def test_near_as_far_half_frame_jump_is_real(profs):
    """Near data read as far: the chirp split fires a period-2 phase
    modulation, and for this target the Doppler peak jumps by -vel_nfft/2
    (a +2 m/s target is READ as a large negative velocity).  The jump itself
    is deterministic (depends on the beat phase), not noise."""
    target = (8.0, 2.0)
    pred = predict_apparent(profs["init1"], profs["init0"], *target)
    pk = _measured(profs, "init1", "init0", target)
    assert pred["half_frame_jump"] is True
    assert pk["k_doppler"] == pytest.approx(pred["k_doppler"], abs=1.0)
    assert pk["vel_mps"] < 0 < target[1]          # sign catastrophically wrong
    assert abs(pk["vel_mps"] - pred["vel_mps"]) <= profs["init0"].v_acu_mps


@requires_hxx
def test_far_as_near_velocity_sign_is_not_flipped(profs):
    """Far data read as near keeps a clean Doppler peak (mid-record phase
    step leaks but does not move the bin): a receding target stays receding."""
    pk = _measured(profs, "init0", "init1", (8.0, -3.5))
    assert pk["vel_mps"] < 0


@requires_hxx
def test_run_synthetic_table_errors_under_one_bin(profs):
    rows = run_synthetic(profs, target=(8.0, 2.0), seed=20260929)
    assert len(rows) == 4
    for r in rows:
        assert abs(r["R_err_bins"]) < 1.0, r
        assert abs(r["v_err_bins"]) < 1.0, r
        if r["correct"]:
            # correct processing recovers the ground truth to <1 bin as well
            assert abs(r["R_meas_m"] - 8.0) < profs[r["assumed"]].rng_res_1dfft_m
            assert abs(r["v_meas_mps"] - 2.0) < profs[r["assumed"]].v_acu_mps


@requires_hxx
def test_far_data_aliasing_boundary_story(profs):
    """Far-wave content spans +-maxV0/2; read as init1 the *claimed* span is
    +-maxV1/2, so fast far targets alias (README table)."""
    p0, p1 = profs["init0"], profs["init1"]
    ratio = (p0.nchirp / p1.nchirp) * (p0.chirp_period_s / p1.chirp_period_s) \
        * (p1.wavelength_m / p0.wavelength_m)
    assert 18.0 * ratio > p1.max_v_mps / 2     # +18 m/s far target aliases
