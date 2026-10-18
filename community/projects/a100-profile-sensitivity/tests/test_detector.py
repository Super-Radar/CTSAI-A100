"""Detector regression: synthetic streams + the 8 public captures."""

import numpy as np
import pytest

from profile_sensitivity.detector import detect_profile
from profile_sensitivity.mismatch import synth_adc_words, list_captures
from profile_sensitivity.pipeline import read_raw_words, unpack_words
from profile_sensitivity.profiles import DEFAULT_DATA_DIR, DEFAULT_HXX_DIR, load_profiles

requires_hxx = pytest.mark.skipif(
    not DEFAULT_HXX_DIR.exists(), reason="official hxx configs not in checkout")
requires_data = pytest.mark.skipif(
    not DEFAULT_DATA_DIR.exists(), reason="public ADC captures not in checkout")


@pytest.fixture(scope="module")
def profs():
    if not DEFAULT_HXX_DIR.exists():
        pytest.skip("official hxx configs not in checkout")
    return load_profiles()


@requires_hxx
def test_synthetic_far_empty_scene_detected(profs):
    """Feature A is scene-independent: noise-only far capture still init0."""
    words = synth_adc_words(profs["init0"], [], noise_sigma=300.0, seed=3)
    det = detect_profile(unpack_words(words), profs["init0"], profs["init1"])
    assert det.decision == "init0"
    assert det.zero_tail_fraction == 1.0


@requires_hxx
def test_synthetic_near_detected(profs):
    words = synth_adc_words(profs["init1"], [(8.0, 2.0, 20000.0)], seed=3)
    det = detect_profile(unpack_words(words), profs["init0"], profs["init1"])
    assert det.decision == "init1"
    assert det.zero_tail_fraction < 0.05
    assert det.click_asymmetry < 0.35


def test_pure_noise_abstains():
    """Documented failure mode: no structure at all -> uncertain, not a guess."""
    from profile_sensitivity.profiles import Profile
    # minimal stand-in profiles (only the detector constants matter here)
    base = dict(name="t", label="t", fmcw_startfreq=76.3, fmcw_bandwidth=300.0,
                fmcw_chirp_rampup=43.0, fmcw_chirp_period=48.0, nchirp=256,
                adc_freq=25.0, dec_factor=1, adc_sample_start=2.0,
                adc_sample_end=42.8, rng_nfft=1024, vel_nfft=256,
                anti_velamb_en=False, anti_velamb_delay=10.0,
                tx_groups=(1, 0, 0, 0), rng_win="hanning", vel_win="hanning")
    far = Profile(**base)
    near = Profile(**{**base, "nchirp": 128, "adc_sample_end": 104.4,
                      "rng_nfft": 2048, "vel_nfft": 128,
                      "fmcw_bandwidth": 750.0, "fmcw_chirp_rampup": 104.65,
                      "fmcw_chirp_period": 108.5, "fmcw_startfreq": 76.119})
    rng = np.random.default_rng(11)
    noise = rng.integers(-2000, 2000, size=262144, dtype=np.int64)
    det = detect_profile(noise.astype(np.int16), far, near)
    assert det.decision == "uncertain"


@requires_hxx
@requires_data
def test_public_files_eight_out_of_eight(profs):
    """The detector calls all 8 public captures correctly, high confidence."""
    p0, p1 = profs["init0"], profs["init1"]
    hits = 0
    for cap_name, files in list_captures(DEFAULT_DATA_DIR).items():
        for path in files:
            samples = unpack_words(read_raw_words(path)[: p0.expected_payload_words])
            det = detect_profile(samples, p0, p1)
            assert det.decision == cap_name, (path.name, det)
            assert det.confidence == "high", (path.name, det)
            hits += 1
    assert hits == 8
