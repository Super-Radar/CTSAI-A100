"""Hand-computed physics checks against get_radar_paramete.m parity."""

from pathlib import Path

import pytest

from profile_sensitivity.profiles import (
    DEFAULT_DATA_DIR, DEFAULT_HXX_DIR, Profile, load_profiles,
    mismatch_ratios)

requires_hxx = pytest.mark.skipif(
    not DEFAULT_HXX_DIR.exists(), reason="official hxx configs not in checkout")


@pytest.fixture(scope="module")
def profs():
    if not DEFAULT_HXX_DIR.exists():
        pytest.skip("official hxx configs not in checkout")
    return load_profiles()


@requires_hxx
def test_raw_hxx_fields():
    """The parsed raw fields equal the values printed in the two hxx files."""
    p0 = Profile.from_hxx(DEFAULT_HXX_DIR / "sensor_config_init0.hxx",
                          "init0", "far")
    p1 = Profile.from_hxx(DEFAULT_HXX_DIR / "sensor_config_init1.hxx",
                          "init1", "near")
    assert (p0.fmcw_startfreq, p0.fmcw_bandwidth, p0.fmcw_chirp_rampup) == (76.3, 300.0, 43.0)
    assert (p0.fmcw_chirp_period, p0.nchirp, p0.adc_freq) == (48.0, 256, 25.0)
    assert (p0.adc_sample_start, p0.adc_sample_end, p0.rng_nfft, p0.vel_nfft) == (2.0, 42.8, 1024, 256)
    assert (p1.fmcw_startfreq, p1.fmcw_bandwidth, p1.fmcw_chirp_rampup) == (76.119, 750.0, 104.65)
    assert (p1.fmcw_chirp_period, p1.nchirp, p1.adc_freq) == (108.5, 128, 25.0)
    assert (p1.adc_sample_start, p1.adc_sample_end, p1.rng_nfft, p1.vel_nfft) == (2.0, 104.4, 2048, 128)
    assert p0.rng_win == p0.vel_win == "hanning"
    assert p1.rng_win == p1.vel_win == "hanning"


@requires_hxx
def test_ml_vs_si_parity(profs):
    """Every SI property equals its verbatim MATLAB transcription."""
    for p in profs.values():
        assert p.max_dis_m == pytest.approx(p.ml_max_dis(), rel=1e-12)
        assert p.rng_res_m == pytest.approx(p.ml_rng_res(), rel=1e-12)
        assert p.max_v_mps == pytest.approx(p.ml_max_v(), rel=1e-12)
        assert p.v_res_mps == pytest.approx(p.ml_v_res(), rel=1e-12)
        assert p.fram_time_s == pytest.approx(p.ml_fram_time_ms() * 1e-3, rel=1e-12)
        assert p.v_acu_mps == pytest.approx(p.ml_max_v() / p.vel_nfft, rel=1e-12)


@requires_hxx
def test_derived_physics_hand_computed(profs):
    """Hand-derived reference numbers (see docstring for the arithmetic)."""
    p0, p1 = profs["init0"], profs["init1"]

    # adc_sample_num: init0 round((42.8-2)*25)=1020 (dec==1 -> no +1);
    #                 init1 round((104.4-2)*25)=2560 -> clamped to rng_nfft=2048
    assert p0.adc_sample_num == 1020
    assert p1.adc_sample_num == 2048
    assert p0.dead_slots_per_chirp == 4
    assert p1.dead_slots_per_chirp == 0

    # slopes: 300/43 resp. 750/104.65 MHz/us
    assert p0.slope_hz_per_s == pytest.approx(300 / 43 * 1e12, rel=1e-12)
    assert p1.slope_hz_per_s == pytest.approx(750 / 104.65 * 1e12, rel=1e-12)

    # max_dis = c/2 * (Fs/2) / S: 1.875e15 / S
    assert p0.max_dis_m == pytest.approx(268.75, rel=2e-3)
    assert p1.max_dis_m == pytest.approx(261.618, rel=2e-3)

    # rng_res = c / (2*S*window)
    assert p0.rng_res_m == pytest.approx(0.52696, rel=2e-3)   # window 40.8 us
    assert p1.rng_res_m == pytest.approx(0.20440, rel=2e-3)   # window 102.4 us

    # rng_res_1dfft = max_dis / (rng_nfft/2)
    assert p0.rng_res_1dfft_m == pytest.approx(268.75 / 512, rel=2e-3)
    assert p1.rng_res_1dfft_m == pytest.approx(261.618 / 1024, rel=2e-3)

    # fram_time = Tp * nchirp
    assert p0.fram_time_s == pytest.approx(12.288e-3, rel=1e-9)
    assert p1.fram_time_s == pytest.approx(13.888e-3, rel=1e-9)

    # maxV = lambda / (2*Tp);  V_acu = maxV / vel_nfft
    lam0, lam1 = 3e8 / 76.3e9, 3e8 / 76.119e9
    assert p0.max_v_mps == pytest.approx(lam0 / (2 * 48e-6), rel=2e-3)
    assert p1.max_v_mps == pytest.approx(lam1 / (2 * 108.5e-6), rel=2e-3)
    assert p0.v_acu_mps == pytest.approx(0.159987, rel=2e-3)
    assert p1.v_acu_mps == pytest.approx(0.141889, rel=2e-3)

    # THE TRAP: identical expected payload per channel file
    assert p0.expected_payload_words == 131072
    assert p1.expected_payload_words == 131072

    # single TX in both profiles
    assert p0.nvirtual_chirp == 1 and p1.nvirtual_chirp == 1
    assert p0.nvirtual_array == 4 and p1.nvirtual_array == 4


@requires_hxx
def test_mismatch_ratios_hand_computed(profs):
    """R_app/R_true = S_cap/S_ass;  v_app/v_true = closed form (see module)."""
    p0, p1 = profs["init0"], profs["init1"]
    n2f = mismatch_ratios(p1, p0)   # near data read as far
    f2n = mismatch_ratios(p0, p1)   # far data read as near
    assert n2f["range_ratio"] == pytest.approx((750 * 43) / (104.65 * 300), rel=1e-9)
    assert f2n["range_ratio"] == pytest.approx((104.65 * 300) / (750 * 43), rel=1e-9)
    assert n2f["range_ratio"] == pytest.approx(1.027234, rel=1e-4)
    assert f2n["range_ratio"] == pytest.approx(0.973488, rel=1e-4)
    lam_ratio = (3e8 / 76.3e9) / (3e8 / 76.119e9)
    assert n2f["vel_ratio"] == pytest.approx(
        0.5 * (108.5 / 48) * lam_ratio, rel=1e-4)
    assert f2n["vel_ratio"] == pytest.approx(
        2 * (48 / 108.5) / lam_ratio, rel=1e-4)
    assert n2f["vel_ratio"] == pytest.approx(1.127527, rel=1e-4)
    assert f2n["vel_ratio"] == pytest.approx(0.886896, rel=1e-4)


def test_bitget_nvirtual_chirp_synthetic():
    """tx_groups bitget loop: {0x0000,0x0000,0x0001,0x0000} -> 1 TX."""
    base = dict(name="t", label="t", fmcw_startfreq=76.3, fmcw_bandwidth=300.0,
                fmcw_chirp_rampup=43.0, fmcw_chirp_period=48.0, nchirp=256,
                adc_freq=25.0, dec_factor=1, adc_sample_start=2.0,
                adc_sample_end=42.8, rng_nfft=1024, vel_nfft=256,
                anti_velamb_en=False, anti_velamb_delay=10.0,
                rng_win="hanning", vel_win="hanning")
    p = Profile(tx_groups=(0x0000, 0x0000, 0x0001, 0x0000), **base)
    assert p.nvirtual_chirp == 1
    p2 = Profile(tx_groups=(0x0000, 0x0010, 0x0000, 0x0000), **base)
    assert p2.nvirtual_chirp == 2   # bit 5 -> MATLAB bitget(pos 5) -> iTx=2
