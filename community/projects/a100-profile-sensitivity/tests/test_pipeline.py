"""Unpacking / reshape / FFT-chain unit checks against tiny handcrafted data."""

import numpy as np
import pytest

from profile_sensitivity.pipeline import (
    dominant_peak, load_cube, range_axis_m, rv_image, samples_to_words,
    unpack_words, velocity_axis_mps)
from profile_sensitivity.profiles import DEFAULT_HXX_DIR, load_profiles

requires_hxx = pytest.mark.skipif(
    not DEFAULT_HXX_DIR.exists(), reason="official hxx configs not in checkout")


def test_unpack_word_layout():
    """uint32 = high int16 first, low int16 second (adc_mem_2_real_fixed.m)."""
    words = np.array([0x00010002, 0xFFFF8000, 0x7FFF7FFF], dtype=np.uint32)
    s = unpack_words(words)
    assert s.tolist() == [1, 2, -1, -32768, 32767, 32767]


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(7)
    s = rng.integers(-32768, 32767, size=1000, dtype=np.int64)
    assert unpack_words(samples_to_words(s)).tolist() == s.tolist()


@pytest.fixture(scope="module")
def profs():
    if not DEFAULT_HXX_DIR.exists():
        pytest.skip("official hxx configs not in checkout")
    return load_profiles()


@requires_hxx
def test_load_cube_truncates_and_pads(profs):
    p = profs["init0"]
    expected = p.expected_payload_words
    payload = np.arange(expected, dtype=np.uint32)
    # one trailing pad word (the public files have a trailing 0) -> dropped
    cube = load_cube(np.concatenate([payload, [999]]), p)
    assert cube.shape == (p.rng_nfft, p.nchirp)
    # short input is zero-padded at the end of the flat stream
    cube2 = load_cube(payload[: expected - 5], p)
    assert cube2.shape == cube.shape
    assert np.all(cube2[-10:, -1] == 0)          # last 5 words -> 10 samples
    # word layout: word j of chirp 0 -> samples 2j (high), 2j+1 (low);
    # words 0..41 all have high half 0 and low half j
    assert cube[0, 0] == 0.0 and cube[2 * 41, 0] == 0.0
    assert cube[2 * 41 + 1, 0] == 41.0


@requires_hxx
def test_axes_calibration(profs):
    for p in profs.values():
        r = range_axis_m(p)
        assert r[0] == 0.0
        assert np.allclose(np.diff(r), p.rng_res_1dfft_m)
        v = velocity_axis_mps(p)
        # fftshifted signed bins, monotonic, symmetric span (maxV, no 0 dup)
        assert v[0] == pytest.approx(-p.max_v_mps / 2)
        assert v[-1] == pytest.approx(p.max_v_mps / 2 - p.v_acu_mps)
        assert np.all(np.diff(v) > 0)


@requires_hxx
def test_synthetic_tone_lands_on_predicted_bin(profs):
    """Correct profile: ideal single target peaks at k = f_b*N/Fs (<1 bin)."""
    from profile_sensitivity.mismatch import synth_cube

    for name, p in profs.items():
        r_true, v_true = 8.0, 2.0
        cube = synth_cube(p, [(r_true, v_true, 20000.0)], seed=1)
        img = rv_image(cube, p)
        pk = dominant_peak(img, p)
        f_b = 2.0 * p.slope_hz_per_s * r_true / 3e8
        k_pred = f_b * p.rng_nfft / p.fs_hz
        assert abs(pk["k_range"] - k_pred) <= 1.0
        assert abs(pk["range_m"] - r_true) <= p.rng_res_1dfft_m * 1.0
        f_d = 2.0 * v_true / p.wavelength_m
        kv_pred = f_d * p.chirp_period_s * p.vel_nfft
        assert abs(pk["k_doppler"] - kv_pred) <= 1.0


@requires_hxx
def test_far_synthetic_has_dead_slot_zeros(profs):
    """The synthetic far capture mirrors the firmware's zero-filled slots."""
    from profile_sensitivity.mismatch import synth_adc_words

    p0 = profs["init0"]
    words = synth_adc_words(p0, [(8.0, 2.0, 20000.0)], seed=1)
    s = unpack_words(words)
    m = s.size // 1024
    tail = s[: m * 1024].reshape(m, 1024)[:, 1020:1024]
    assert np.all(tail == 0)

    p1 = profs["init1"]
    words1 = synth_adc_words(p1, [(8.0, 2.0, 20000.0)], seed=1)
    s1 = unpack_words(words1)
    m1 = s1.size // 1024
    tail1 = s1[: m1 * 1024].reshape(m1, 1024)[:, 1020:1024]
    assert (tail1 == 0).mean() < 0.05   # only incidental rounding zeros
