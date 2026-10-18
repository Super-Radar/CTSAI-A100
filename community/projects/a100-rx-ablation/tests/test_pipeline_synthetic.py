# -*- coding: utf-8 -*-
"""官方链路端口的合成数据回归：.hxx 解析、uint32 解包、列优先 reshape、2D-FFT。"""
import numpy as np

from rx_ablation.pipeline import (
    RadarConfig,
    do_fft_2d,
    load_channel,
    unpack_words,
)

HXX_TEMPLATE = """
\tpseudo_header = 1,
    .fmcw_startfreq = {f0},
    .fmcw_bandwidth = {bw},
    .fmcw_chirp_rampup = {ramp},
    .fmcw_chirp_period = {period},
    .tx_groups = {{0x0000, 0x0000, 0x0001, 0x0000}},
    .nchirp = {nchirp},
    .adc_freq = 25,
    .dec_factor = {dec},
    .adc_sample_start = 2,
    .adc_sample_end = {end},
    .rng_win = "hanning",
    .vel_win = "hanning",
    .rng_nfft = 1024,
    .vel_nfft = 256,
    .rng_fft_scalar = 0xff,
    .vel_fft_scalar = 0x3f,
    .anti_velamb_en = false,
"""


def _write_hxx(tmp_path, **kw):
    params = dict(f0=76.3, bw=300, ramp=43, period=48, nchirp=256,
                  dec=1, end=42.8)
    params.update(kw)
    p = tmp_path / "sensor_config_test.hxx"
    p.write_text(HXX_TEMPLATE.format(**params), encoding="utf-8")
    return str(p)


def _tiny_cfg(rng_nfft=8, vel_nfft=8, nchirp=2):
    return RadarConfig(
        source_file="synthetic", fmcw_startfreq=76.3, fmcw_bandwidth=300,
        fmcw_chirp_rampup=43, fmcw_chirp_period=48, nchirp=nchirp,
        adc_freq=25, dec_factor=1, adc_sample_start=2, adc_sample_end=42.8,
        rng_win="hanning", vel_win="hanning", rng_nfft=rng_nfft,
        vel_nfft=vel_nfft, rng_fft_scalar=255, vel_fft_scalar=63,
        tx_groups=[0, 0, 1, 0],
    )


# ---------------------------------------------------------------------------
# .hxx 解析与派生参数
# ---------------------------------------------------------------------------

def test_parse_hxx_and_derived_params(tmp_path):
    cfg = RadarConfig.from_hxx(_write_hxx(tmp_path))
    assert cfg.fmcw_startfreq == 76.3
    assert cfg.tx_groups == [0, 0, 1, 0]
    assert cfg.nvirtual_chirp == 1
    assert cfg.nvirtual_array == 4
    assert cfg.vel_chirp_num == 256
    # 官方 else 分支（dec_factor=1）无 +1：round(40.8*25)=1020
    assert cfg.adc_sample_num == 1020
    assert cfg.rng_bin_num == 512


def test_adc_sample_num_plus_one_branch_and_clamp(tmp_path):
    cfg = RadarConfig.from_hxx(_write_hxx(tmp_path, dec=2))
    # dec_factor>1 分支：round(40.8*25/2)+1 = 511
    assert cfg.adc_sample_num == 511
    cfg2 = RadarConfig.from_hxx(_write_hxx(tmp_path, end=80.0, dec=2))
    # round(78*25/2)+1 = 976 <= 1024，不截断
    assert cfg2.adc_sample_num == 976
    cfg3 = RadarConfig.from_hxx(_write_hxx(tmp_path, end=200.0, dec=2))
    # round(198*25/2)+1 = 2476 > rng_nfft -> 截断到 1024（官方行为）
    assert cfg3.adc_sample_num == 1024


def test_official_resolutions(tmp_path):
    """官方 rng_res / V_res 公式数值（与 get_radar_paramete.m 逐式对应）。"""
    cfg = RadarConfig.from_hxx(_write_hxx(tmp_path))
    assert np.isclose(cfg.rng_res, 1.5e8 / ((300 / 43) * 40.8) / 1e6, rtol=1e-12)
    assert np.isclose(
        cfg.v_res, 0.5 * 3e8 / 76.3e9 * (1.0 / (48 * 256 * 1e-3) / 1e-3), rtol=1e-12
    )
    assert np.isclose(cfg.rng_res, 0.5269608, atol=1e-5)
    assert np.isclose(cfg.v_res, 0.1599873, atol=1e-5)


# ---------------------------------------------------------------------------
# 解包与 reshape 语义
# ---------------------------------------------------------------------------

def test_unpack_words_high_low_interleave_with_sign():
    words = np.array([[0xFFFF8000, 0x00017FFF, 0xFFFFFFFF, 0x00000000]],
                     dtype=np.uint32)
    out = unpack_words(words)
    assert out.dtype == np.int16
    assert out.shape == (1, 8)
    # 每字：高 16 位在前，低 16 位在后；两补码
    assert out[0].tolist() == [
        -1, -32768,          # 0xFFFF8000 -> H=0xFFFF(-1), L=0x8000(-32768)
        1, 32767,            # 0x00017FFF -> H=0x0001,     L=0x7FFF
        -1, -1,              # 0xFFFFFFFF -> H=L=0xFFFF
        0, 0,                # 0x00000000
    ]


def test_load_channel_column_major_and_tail_drop(tmp_path):
    """列优先 reshape：前 rng_nfft/2 个字属于 chirp 0；末尾多余值被丢弃。

    构造 9 个字（8 载荷 + 1 尾），第 k 字 = 0x00010000*k（高 16 位 = k，
    低 16 位 = 0）。rng_nfft=8, nchirp=2 -> expected = 4*2 = 8。
    """
    cfg = _tiny_cfg()
    packed = [0x00010000 * k for k in range(1, 10)]   # k = 1..9
    p = tmp_path / "ch.txt"
    p.write_text(",".join(str(v) for v in packed), encoding="utf-8")
    mat, n_words = load_channel(str(p), cfg)
    assert n_words == 9
    assert mat.shape == (8, 2)
    # chirp0 = 字 1..4 -> 样本 [1,0, 2,0, 3,0, 4,0]
    assert mat[:, 0].tolist() == [1, 0, 2, 0, 3, 0, 4, 0]
    # chirp1 = 字 5..8
    assert mat[0, 1] == 5 and mat[6, 1] == 8
    # 第 9 个字（尾值）被丢弃
    assert not (mat == 9).any()


# ---------------------------------------------------------------------------
# 2D-FFT 功能测试
# ---------------------------------------------------------------------------

def test_do_fft_2d_single_tone_hits_expected_bins():
    cfg = _tiny_cfg(rng_nfft=64, vel_nfft=64, nchirp=64)
    rng = np.random.default_rng(0)
    k_rng = 9
    r = np.arange(64)
    sig = np.cos(2 * np.pi * k_rng * r / 64.0) * 1000.0   # 距离维单音
    adc = np.repeat(sig[:, None, None], 64, axis=1).repeat(4, axis=2)
    adc += rng.normal(0, 1.0, adc.shape)
    _, vel = do_fft_2d(adc, cfg)
    assert vel.shape == (32, 64, 4)
    power = (np.abs(vel) ** 2).mean(axis=2)
    pk = np.unravel_index(np.argmax(power), power.shape)
    assert pk[0] == k_rng          # 距离 bin
    assert pk[1] == 0              # chirp 间无调制 -> Doppler bin 0


def test_do_fft_2d_scalars_applied():
    """官方 /rng_fft_scalar 与 /vel_fft_scalar 必须体现在幅度上。"""
    cfg = _tiny_cfg(rng_nfft=64, vel_nfft=64, nchirp=64)
    rng = np.random.default_rng(1)
    adc = rng.normal(0, 1, (64, 64, 1))
    _, vel = do_fft_2d(adc, cfg)
    # 手工重算（含官方 scalar /255 与 /63）对照
    from scipy.signal.windows import hann
    w = hann(64, sym=True)
    rng_fft = np.fft.fft(adc[:64, :64, 0] * w[:, None], n=64, axis=0) / 255.0
    vel_ref = np.fft.fft(rng_fft[:, :64] * w[None, :], n=64, axis=1)[:32]
    np.testing.assert_allclose(vel[:, :, 0], vel_ref / 63.0, rtol=1e-12)
