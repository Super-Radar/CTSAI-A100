"""Pipeline checks: .hxx parsing, unpacking, ordering, FFT energy conservation.

Includes Parseval / DC-exactness energy checks of the 2D-FFT chain and a
snapshot regression of the full chain on the repository's public captures.
"""

from __future__ import annotations

import numpy as np
import pytest

from noise_study.official_noise import calculate_global_noise
from noise_study.pipeline import (
    SensorConfig,
    adc_mem_2_real_fixed,
    channel_files,
    do_fft_2d,
    hann_window,
    load_adc_data,
    parse_sensor_config,
    power_to_db,
    rv_power,
)


# ---------------------------------------------------------------------------
# .hxx parsing against the real public profiles
# ---------------------------------------------------------------------------


def test_parse_init0(hxx_dir):
    cfg = parse_sensor_config(hxx_dir / "sensor_config_init0.hxx")
    assert cfg.fmcw_startfreq == pytest.approx(76.3)
    assert cfg.fmcw_bandwidth == pytest.approx(300)
    assert cfg.fmcw_chirp_rampup == pytest.approx(43)
    assert cfg.fmcw_chirp_period == pytest.approx(48)
    assert cfg.nchirp == 256
    assert cfg.adc_freq == pytest.approx(25)
    assert cfg.dec_factor == 1
    assert cfg.rng_nfft == 1024 and cfg.vel_nfft == 256
    # hex literals: 0xff / 0x3f
    assert cfg.rng_fft_scalar == 255 and cfg.vel_fft_scalar == 63
    # "hanning" -> code 2
    assert cfg.rng_win == 2 and cfg.vel_win == 2
    # tx_groups {0x0000,0x0000,0x0001,0x0000} -> 1 TX, 4 virtual arrays
    assert cfg.nvirtual_chirp == 1 and cfg.nvirtual_array == 4
    # get_radar_paramete.m: dec_factor == 1 -> NO +1 -> round(40.8*25) = 1020
    assert cfg.adc_sample_num == 1020
    assert cfg.rng_bin_num == 512


def test_parse_init1(hxx_dir):
    cfg = parse_sensor_config(hxx_dir / "sensor_config_init1.hxx")
    assert cfg.fmcw_startfreq == pytest.approx(76.119)
    assert cfg.fmcw_bandwidth == pytest.approx(750)
    assert cfg.nchirp == 128
    assert cfg.rng_nfft == 2048 and cfg.vel_nfft == 128
    assert cfg.nvirtual_chirp == 1 and cfg.nvirtual_array == 4
    # round((104.4-2)*25) = 2560, clamped to rng_nfft = 2048
    assert cfg.adc_sample_num == 2048
    assert cfg.rng_bin_num == 1024


def test_adc_sample_num_plus_one_only_when_decimated():
    """get_radar_paramete.m adds +1 only for dec_factor > 1."""
    base = dict(
        fmcw_startfreq=76.3, fmcw_bandwidth=300, fmcw_chirp_rampup=43,
        fmcw_chirp_period=48, tx_groups=[0, 0, 1, 0], nchirp=256,
        adc_freq=25, adc_sample_start=2, adc_sample_end=42.8,
        rng_win=2, vel_win=2, rng_nfft=4096, vel_nfft=256,
        rng_fft_scalar=255, vel_fft_scalar=63,
    )
    assert SensorConfig(dec_factor=1, **base).adc_sample_num == 1020
    assert SensorConfig(dec_factor=2, **base).adc_sample_num == 511  # round(1020/2)+1


# ---------------------------------------------------------------------------
# uint32 -> 2 x int16 unpacking (adc_mem_2_real_fixed.m)
# ---------------------------------------------------------------------------


def test_adc_mem_2_real_fixed_word_layout():
    words = np.array([[0x00010002, 0xFFFF8000, 0x7FFF7FFF]], dtype=np.uint32)
    out = adc_mem_2_real_fixed(words)
    # H16 of word0 = 0x0001, L16 = 0x0002
    assert out[0, 0] == 1 and out[0, 1] == 2
    # H16 of word1 = 0xFFFF -> -1, L16 = 0x8000 -> -32768
    assert out[0, 2] == -1 and out[0, 3] == -32768
    # 0x7FFF stays +32767
    assert out[0, 4] == 32767 and out[0, 5] == 32767
    assert out.dtype == np.int16 and out.shape == (1, 6)


def test_adc_mem_2_real_fixed_matches_matlab_formula():
    rng = np.random.default_rng(3)
    words = rng.integers(0, 2**32, size=(5, 7), dtype=np.uint64).astype(np.uint32)
    out = adc_mem_2_real_fixed(words)
    for i in range(5):
        for k in range(7):
            w = int(words[i, k])
            h, l = w // 2**16, w % 2**16
            if h >= 2**15:
                h -= 2**16
            if l >= 2**15:
                l -= 2**16
            assert out[i, 2 * k] == h and out[i, 2 * k + 1] == l


# ---------------------------------------------------------------------------
# load_adc_data: reshape ordering + explicit trailing-value record
# ---------------------------------------------------------------------------


@pytest.fixture()
def tiny_cfg() -> SensorConfig:
    return SensorConfig(
        fmcw_startfreq=76.3, fmcw_bandwidth=300, fmcw_chirp_rampup=43,
        fmcw_chirp_period=48, tx_groups=[0, 0, 1, 0], nchirp=2,
        adc_freq=25, dec_factor=1, adc_sample_start=2, adc_sample_end=42.8,
        rng_win=0, vel_win=0, rng_nfft=4, vel_nfft=2,
        rng_fft_scalar=1, vel_fft_scalar=1,
    )


def test_load_adc_data_ordering_and_tail_record(tmp_path, tiny_cfg):
    """2 chirps x 2 words + 1 trailing value; check MATLAB reshape equivalence."""
    # words: chirp0 = (w00, w01), chirp1 = (w10, w11) after the
    # column-major [rng_nfft/2, nchirp] reshape + transpose.
    words_per_ch = [
        [0x00010002, 0x00030004, 0xFFFF8000, 0x00050006],
        [0x00060007, 0x00080009, 0x000A000B, 0x000C000D],
        [0x000E000F, 0x00100011, 0x00120013, 0x00140015],
        [0x00160017, 0x00180019, 0x001A001B, 0x001C001D],
    ]
    files = []
    for ch, words in enumerate(words_per_ch):
        f = tmp_path / f"ch{ch}.txt"
        f.write_text(",".join(str(w) for w in words) + ",0")  # 5 values, expected 4
        files.append(f)
    adc, reports = load_adc_data(files, tiny_cfg)
    assert adc.shape == (4, 2, 4)                # (samples, chirps, channels)
    for rep in reports:
        assert rep.n_values == 5 and rep.expected_len == 4
        assert rep.action == "truncate" and rep.dropped_tail_values == [0]
    assert adc[:, 0, 0].tolist() == [1, 2, 3, 4]   # chirp 0 samples H,L,H,L
    assert adc[:, 1, 0].tolist() == [-1, -32768, 5, 6]
    assert adc[:, 0, 1].tolist() == [6, 7, 8, 9]
    assert adc[:, 0, 3].tolist() == [0x16, 0x17, 0x18, 0x19]
    assert adc[:, 1, 3].tolist() == [0x1A, 0x1B, 0x1C, 0x1D]


def test_channel_files_resolution(data_dir):
    files = channel_files(data_dir, "far")
    assert len(files) == 4
    for ch, path in enumerate(files):
        assert f"_f0_{ch}_" in path.name
    files_near = channel_files(data_dir, "near")
    assert all(f"_f1_{i}_" in p.name for i, p in enumerate(files_near))


# ---------------------------------------------------------------------------
# Window equivalence
# ---------------------------------------------------------------------------


def test_hann_matches_matlab_formula():
    n = 8
    manual = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(n) / (n - 1)))
    assert np.allclose(hann_window(n), manual)


# ---------------------------------------------------------------------------
# 2D FFT chain: DC exactness + Parseval energy conservation
# ---------------------------------------------------------------------------


def make_chain_cfg(rng_nfft: int, vel_nfft: int, nchirp: int) -> SensorConfig:
    return SensorConfig(
        fmcw_startfreq=76.3, fmcw_bandwidth=300, fmcw_chirp_rampup=43,
        fmcw_chirp_period=48, tx_groups=[0, 0, 1, 0], nchirp=nchirp,
        adc_freq=25, dec_factor=1, adc_sample_start=2, adc_sample_end=42.8,
        rng_win=0, vel_win=0, rng_nfft=rng_nfft, vel_nfft=vel_nfft,
        rng_fft_scalar=1, vel_fft_scalar=1,
    )


def test_do_fft_2d_dc_bin_exact_constant_input():
    """Constant cube, rect windows, scalars 1 -> all energy in DC bins."""
    cfg = make_chain_cfg(rng_nfft=64, vel_nfft=16, nchirp=16)
    c = 2.5
    adc = np.full((cfg.rng_nfft, cfg.nchirp, cfg.nvirtual_array), c, dtype=np.float64)
    rng_fft, vel_fft = do_fft_2d(adc, cfg)
    expected_rng_dc = cfg.rng_nfft * c
    expected_vel_dc = cfg.vel_nfft * expected_rng_dc
    for ch in range(cfg.nvirtual_array):
        np.testing.assert_allclose(rng_fft[0, :, ch], expected_rng_dc, rtol=1e-12)
        assert np.allclose(np.abs(rng_fft[1:, :, ch]), 0.0, atol=1e-9)
        np.testing.assert_allclose(vel_fft[0, 0, ch], expected_vel_dc, rtol=1e-12)
        assert np.allclose(np.abs(vel_fft[0, 1:, ch]), 0.0, atol=1e-9)
        assert np.allclose(np.abs(vel_fft[1:, :, ch]), 0.0, atol=1e-9)


def test_do_fft_2d_range_axis_parseval():
    """Rect window, scalar 1: sum|FFT|^2 == N * sum|x|^2 per chirp/channel."""
    cfg = make_chain_cfg(rng_nfft=32, vel_nfft=8, nchirp=8)
    rng = np.random.default_rng(11)
    adc = rng.normal(size=(cfg.rng_nfft, cfg.nchirp, cfg.nvirtual_array))
    rng_fft, _ = do_fft_2d(adc, cfg)
    energy_in = np.sum(adc.astype(np.float64) ** 2, axis=0)
    energy_out = np.sum(np.abs(rng_fft) ** 2, axis=0)
    np.testing.assert_allclose(energy_out, cfg.rng_nfft * energy_in, rtol=1e-10)


def test_do_fft_2d_velocity_axis_parseval_chirp_constant():
    """Chirp-constant profiles -> all velocity energy in Doppler bin 0.

    For each range row r: sum_j |V[r, j]|^2 == vel_nfft^2 * |R[r]|^2.
    """
    cfg = make_chain_cfg(rng_nfft=16, vel_nfft=8, nchirp=8)
    rng = np.random.default_rng(13)
    profile = rng.normal(size=(cfg.rng_nfft, 1))
    adc = np.repeat(profile[:, None, :], cfg.nchirp, axis=1)
    adc = np.concatenate([adc] * cfg.nvirtual_array, axis=2)  # identical channels
    _, vel_fft = do_fft_2d(adc, cfg)
    # expected: velocity bin 0 == vel_nfft * range-DFT of the profile,
    # restricted to the first rng_nfft/2 rows (scalar 1, rect window)
    ref_rng_dc_free = np.abs(np.fft.fft(profile[:, 0], n=cfg.rng_nfft))
    for ch in range(cfg.nvirtual_array):
        np.testing.assert_allclose(
            np.abs(vel_fft[:, 0, ch]),
            cfg.vel_nfft * ref_rng_dc_free[: cfg.rng_nfft // 2],
            rtol=1e-10,
        )
        assert np.allclose(np.abs(vel_fft[:, 1:, ch]), 0.0, atol=1e-9)


def test_do_fft_2d_matches_direct_matlab_loops():
    """Vectorised chain vs an explicit per-chirp/per-row transcription."""
    cfg = make_chain_cfg(rng_nfft=16, vel_nfft=8, nchirp=8)
    rng = np.random.default_rng(17)
    adc = rng.normal(size=(cfg.rng_nfft, cfg.nchirp, cfg.nvirtual_array))
    rng_fft, vel_fft = do_fft_2d(adc, cfg)
    for ch in range(cfg.nvirtual_array):
        for chirp in range(cfg.nchirp):
            col = adc[: cfg.adc_sample_num, chirp, ch].astype(np.float64)
            ref = np.fft.fft(col, n=cfg.rng_nfft) / cfg.rng_fft_scalar
            np.testing.assert_allclose(rng_fft[:, chirp, ch], ref, rtol=1e-12)
        for row in range(cfg.rng_nfft // 2):
            seg = rng_fft[row, :, ch]
            ref = np.fft.fft(seg, n=cfg.vel_nfft) / cfg.vel_fft_scalar
            np.testing.assert_allclose(vel_fft[row, :, ch], ref, rtol=1e-12)


# ---------------------------------------------------------------------------
# Full-chain snapshot on the repository's public captures
# ---------------------------------------------------------------------------

# Values produced by this exact code base (numpy.fft) on the public example
# captures; guards against accidental chain regressions.  Regenerate by
# running noise_study.cli.
SNAPSHOT_OFFICIAL_NOISE_DB = {
    "far": -7.514094,   # 远波 f0, 4-channel average, this code base on the public capture
    "near": -6.695122,  # 近波 f1, 4-channel average
}


@pytest.fixture(scope="module")
def public_datasets(data_dir, hxx_dir):
    from noise_study.study import load_dataset

    return {kind: load_dataset(kind, data_dir, hxx_dir) for kind in ("far", "near")}


def test_full_chain_official_noise_snapshot(public_datasets):
    for kind, ds in public_datasets.items():
        value = calculate_global_noise(ds.vel_fft, ds.cfg)
        expected = SNAPSHOT_OFFICIAL_NOISE_DB[kind]
        assert value == pytest.approx(expected, abs=0.05), (
            f"{kind}: official noiseValue drifted from snapshot "
            f"{expected} dB -> {value} dB"
        )


def test_full_chain_load_reports(public_datasets):
    for kind, ds in public_datasets.items():
        for rep in ds.load_reports:
            assert rep.n_values == 131073      # payload 131072 + 1 trailing value
            assert rep.expected_len == 131072
            assert rep.action == "truncate"
            assert rep.dropped_tail_values == [0]
