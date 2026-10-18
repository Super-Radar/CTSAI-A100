"""Unit tests for the official-chain reproduction and the synthetic generator."""

from dataclasses import replace

import numpy as np
import pytest

from dropout_study.pipeline import (
    INIT0,
    INIT1,
    RadarConfig,
    load_adc_words,
    pack_words,
    rv_map_average,
    rv_power_map,
    synth_single_target_words,
    unpack_words,
)

# Small config: 4 chirps x 8 words (= 16 samples per chirp), vel_nfft 8.
SMALL = RadarConfig("small-test", 76.3, 300.0, 43.0, 48.0, 4, 25.0, 2.0, 42.8,
                    1, 16, 8, 255, 63)


def test_adc_sample_num_matches_official_script():
    # get_radar_paramete.m, dec_factor == 1 branch, no +1, capped at rng_nfft.
    assert INIT0.adc_sample_num == 1020          # round((42.8 - 2) * 25)
    assert INIT1.adc_sample_num == 2048          # round(102.4 * 25) = 2560 -> capped
    assert SMALL.adc_sample_num == 16            # capped at rng_nfft = 16


def test_expected_word_counts():
    assert INIT0.expected_words == 131072        # 512 words x 256 chirps
    assert INIT1.expected_words == 131072        # 1024 words x 128 chirps


def test_unpack_known_words():
    # 0x0002FFFE -> hi16 = 0x0002 = +2, lo16 = 0xFFFE = -2 (two's complement)
    samples = unpack_words(np.array([[196606]], dtype=np.uint32))
    assert samples.shape == (1, 2)
    assert samples[0].tolist() == [2, -2]
    # 0xFFFFFFFF -> both halves are 0xFFFF = -1
    assert unpack_words(np.array([[4294967295]], dtype=np.uint32))[0].tolist() == [-1, -1]


def test_pack_unpack_roundtrip():
    rng = np.random.default_rng(1)
    samples = rng.integers(-32768, 32768, size=(6, 20), dtype=np.int16)
    assert np.array_equal(unpack_words(pack_words(samples)), samples)


def _write_txt(path, values):
    path.write_text(",".join(str(v) for v in values), encoding="ascii")


def test_flat_stream_maps_chirp_major_like_matlab(tmp_path):
    # MATLAB reshape(col, [wpc, nchirp]) + transpose: flat w -> chirp w//wpc.
    values = [1000 + i for i in range(32)]
    f = tmp_path / "adc.txt"
    _write_txt(f, values)
    words, report = load_adc_words(f, SMALL)
    assert words.shape == (4, 8)
    assert words[2, 3] == 1000 + 2 * 8 + 3
    assert report.n_extra_truncated == 0 and report.n_zero_padded == 0


def test_load_report_truncates_extra_trailing_values(tmp_path):
    f = tmp_path / "adc_long.txt"
    _write_txt(f, list(range(33)))  # like the shipped files: payload + 1 extra value
    words, report = load_adc_words(f, SMALL)
    assert report.n_values == 33
    assert report.n_extra_truncated == 1
    assert words.shape == (4, 8)
    assert any("truncated" in note for note in report.notes)


def test_load_report_zero_pads_short_files(tmp_path):
    f = tmp_path / "adc_short.txt"
    _write_txt(f, list(range(20)))
    words, report = load_adc_words(f, SMALL)
    assert report.n_zero_padded == 12
    assert words.shape == (4, 8)
    assert (words.reshape(-1)[20:] == 0).all()
    assert np.array_equal(words.reshape(-1)[:20], np.arange(20, dtype=np.uint32))
    assert any("zero-padded" in note for note in report.notes)


def test_rv_power_map_shape_and_empty():
    frame = unpack_words(pack_words(np.zeros((4, 16), dtype=np.int16)))
    mp = rv_power_map(frame, SMALL)
    assert mp.shape == (SMALL.rng_nfft // 2, SMALL.vel_nfft)  # (8, 8)
    assert (mp >= 0).all()
    assert rv_power_map(np.zeros((0, 16), dtype=np.int16), SMALL) is None
    assert rv_power_map(None, SMALL) is None


def test_rv_map_average_requires_one_usable_channel():
    frame = unpack_words(pack_words(np.zeros((4, 16), dtype=np.int16)))
    mp = rv_map_average([frame, frame], SMALL)
    assert mp.shape == (SMALL.rng_nfft // 2, SMALL.vel_nfft)
    assert rv_map_average([None, None], SMALL) is None


def test_rv_power_map_supports_shorter_chirp_count():
    # reject strategy: 3 of 4 chirps survive; window covers 3 chirps, FFT stays vel_nfft.
    frame = unpack_words(pack_words(np.zeros((3, 16), dtype=np.int16)))
    mp = rv_power_map(frame, SMALL)
    assert mp.shape == (SMALL.rng_nfft // 2, SMALL.vel_nfft)


@pytest.mark.parametrize("cfg,range_m,doppler_bin", [
    (INIT0, 5.0, 12),
    (INIT0, 21.0, 100),
    (replace(INIT0, nchirp=64), 5.0, 30),
])
def test_synthetic_peak_lands_on_theoretical_bins(cfg, range_m, doppler_bin):
    vel = doppler_bin * cfg.vel_per_bin_mps
    words = synth_single_target_words(cfg, range_m, vel)
    mp = rv_power_map(unpack_words(words), cfg)
    ir, ic = np.unravel_index(int(np.argmax(mp)), mp.shape)
    assert abs(ir - cfg.range_bin_of(range_m)) <= 1.0
    assert abs(ic - (doppler_bin % cfg.vel_nfft)) <= 1.0


def test_real_far_sample_file_zero_perturbation_and_report():
    from dropout_study.study import default_data_dir

    files = sorted(default_data_dir().glob("adc_远波_*_f0_*.txt"))
    if not files:
        pytest.skip("official ADC sample data not found in the repo checkout")
    words, report = load_adc_words(files[0], INIT0)
    assert report.n_values == 131073            # payload 131072 + 1 trailing zero
    assert report.n_extra_truncated == 1
    assert report.expected_words == 131072
    ref = rv_power_map(unpack_words(words), INIT0)
    assert ref.shape == (512, 256)
    # an identity corruption must reproduce the reference map exactly
    from dropout_study.corruption import corrupt_words

    w2, mask = corrupt_words(words, "random_drop", 0, np.random.default_rng(1))
    assert np.array_equal(rv_power_map(unpack_words(w2), INIT0), ref)
