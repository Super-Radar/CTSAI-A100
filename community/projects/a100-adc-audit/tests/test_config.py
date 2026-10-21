"""hxx 配置解析测试：合成配置 + 仓内真实配置。"""

from pathlib import Path

import pytest

from adc_audit.config import ConfigError, parse_hxx
from conftest import REPO_ROOT, HXX_TEMPLATE

REAL_CFG_DIR = (
    REPO_ROOT
    / "ADC数据采集"
    / "matlab_signal_processing_platform_231023_for_txt_A100"
    / "cfg"
    / "CTASI-A100配置"
)

have_real = REAL_CFG_DIR.is_dir()
pytestmark = pytest.mark.skipif(not have_real, reason="仓内真实配置不可用")


def test_parse_synthetic(tmp_path):
    p = tmp_path / "sensor_config_init3.hxx"
    p.write_text(
        HXX_TEMPLATE.format(
            startfreq=77.0, bw=500, nchirp=32,
            sample_end=20.0, rng_nfft=512, vel_nfft=128,
        ),
        encoding="utf-8",
    )
    cfg = parse_hxx(p)
    assert cfg.profile == 3
    assert cfg.nchirp == 32
    assert cfg.rng_nfft == 512
    assert cfg.expected_words == 256 * 32
    assert cfg.window_samples == round((20.0 - 2) * 25)


def test_parse_missing_field(tmp_path):
    p = tmp_path / "sensor_config_init0.hxx"
    p.write_text(".nchirp = 8,\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        parse_hxx(p)


def test_real_far_config_init0():
    cfg = parse_hxx(REAL_CFG_DIR / "sensor_config_init0.hxx")
    assert cfg.profile == 0
    assert cfg.start_freq_ghz == pytest.approx(76.3)
    assert cfg.bandwidth_mhz == pytest.approx(300)
    assert cfg.nchirp == 256
    assert cfg.rng_nfft == 1024
    assert cfg.vel_nfft == 256
    assert cfg.adc_freq_mhz == pytest.approx(25)
    assert cfg.sample_start_us == pytest.approx(2)
    assert cfg.sample_end_us == pytest.approx(42.8)
    # (42.8-2)us * 25MHz = 1020 样本 < rng_nfft 1024（每 chirp 尾部 4 个补零样本）
    assert cfg.window_samples == 1020
    assert cfg.expected_words == 512 * 256 == 131072


def test_real_near_config_init1():
    cfg = parse_hxx(REAL_CFG_DIR / "sensor_config_init1.hxx")
    assert cfg.profile == 1
    assert cfg.start_freq_ghz == pytest.approx(76.119)
    assert cfg.bandwidth_mhz == pytest.approx(750)
    assert cfg.nchirp == 128
    assert cfg.rng_nfft == 2048
    assert cfg.vel_nfft == 128
    # (104.4-2)us * 25MHz = 2560 样本 > rng_nfft 2048（平台按 rng_nfft 截取）
    assert cfg.window_samples == 2560
    assert cfg.expected_words == 1024 * 128 == 131072
