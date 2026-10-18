"""Tests for the ADC unpacking / FFT pipeline (MATLAB-equivalent chain)."""

import numpy as np
import pytest

from phase_sens.pipeline import (
    SENSOR_CONFIGS,
    align_length,
    dominant_bin,
    hann_periodic,
    range_fft,
    repack_adc,
    unpack_adc,
)


def test_word_order_high16_first():
    # MATLAB adc_mem_2_real_fixed: H16 = floor(w/2^16) is the FIRST sample,
    # L16 = mod(w, 2^16) the SECOND.
    samples = unpack_adc(np.array([0x00010002], dtype=np.uint32),
                         rng_nfft=2, nchirp=1)
    assert samples.shape == (2, 1)
    assert samples[0, 0] == 1.0
    assert samples[1, 0] == 2.0


def test_two_complement_decode():
    samples = unpack_adc(np.array([0xFFFF8000], dtype=np.uint32),
                         rng_nfft=2, nchirp=1)
    assert samples[0, 0] == -1.0      # 0xFFFF -> -1
    assert samples[1, 0] == -32768.0  # 0x8000 -> -32768


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(7)
    words = rng.integers(0, 2 ** 32, size=(256 // 2) * 4, dtype=np.uint64).astype(np.uint32)
    samples = unpack_adc(words, rng_nfft=256, nchirp=4)
    assert samples.shape == (256, 4)
    back = repack_adc(samples)
    assert np.array_equal(back, words)


def test_align_reports_extra_value():
    values = np.arange(10)
    aligned, report = align_length(values, 8)
    assert aligned.tolist() == [0, 1, 2, 3, 4, 5, 6, 7]
    assert report["extra_count"] == 2
    assert report["first_extra_value"] == 8
    assert report["pad_count"] == 0


def test_align_pads_short_input():
    aligned, report = align_length(np.arange(6), 8)
    assert aligned.tolist() == [0, 1, 2, 3, 4, 5, 0, 0]
    assert report["pad_count"] == 2
    assert report["extra_count"] == 0
    assert report["first_extra_value"] is None


def test_adc_sample_num_derivation():
    # get_radar_paramete.m: round((end-start)*adc_freq/dec) [no +1 since
    # dec_factor==1], capped at rng_nfft.
    assert SENSOR_CONFIGS["init0"].adc_sample_num == 1020  # round(40.8*25)
    assert SENSOR_CONFIGS["init1"].adc_sample_num == 2048  # 2560 capped
    assert SENSOR_CONFIGS["init0"].expected_word_count == 512 * 256


def test_range_fft_places_tone_in_expected_bin():
    cfg = SENSOR_CONFIGS["init0"]
    rng = np.random.default_rng(11)
    m = 37
    k = np.arange(cfg.adc_sample_num)
    tone = np.round(1000 * np.cos(2 * np.pi * m * k / cfg.rng_nfft))
    samples = np.zeros((cfg.rng_nfft, cfg.nchirp))
    samples[: cfg.adc_sample_num, :] = tone[:, None] + \
        rng.normal(0, 0.01, size=(cfg.adc_sample_num, cfg.nchirp))
    r = range_fft(samples, cfg)
    assert dominant_bin(r) == m


def test_hann_periodic_matches_matlab_definition():
    n = 16
    expected = 0.5 * (1 - np.cos(2 * np.pi * np.arange(n) / n))
    assert np.allclose(hann_periodic(n), expected, atol=1e-12)


def test_dominant_bin_excludes_dc():
    matrix = np.zeros((64, 8), dtype=complex)
    matrix[0, :] = 100.0 + 0j   # DC bin, must be ignored
    matrix[5, :] = 2.0 + 1.0j   # weaker but the strongest non-DC bin
    assert dominant_bin(matrix) == 5
