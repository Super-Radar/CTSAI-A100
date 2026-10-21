"""对仓内 16 个真实 ADC 文件的回归测试（数据在仓库内，随 checkout 提供）。"""

from pathlib import Path

import pytest

from adc_audit.config import parse_hxx
from adc_audit.loader import load_adc_file
from adc_audit.rules import audit, audit_file
from conftest import REPO_ROOT

BASE = REPO_ROOT / "ADC数据采集"
NEAR_FAR_DIR = BASE / "示例adc数据和结果"
MATLAB_DATA = BASE / "matlab_signal_processing_platform_231023_for_txt_A100" / "data"
CFG_DIR = (
    BASE
    / "matlab_signal_processing_platform_231023_for_txt_A100"
    / "cfg"
    / "CTASI-A100配置"
)

have_real = NEAR_FAR_DIR.is_dir() and MATLAB_DATA.is_dir()
pytestmark = pytest.mark.skipif(not have_real, reason="仓内真实数据不可用")


@pytest.fixture(scope="module")
def real_cfgs():
    return {
        0: parse_hxx(CFG_DIR / "sensor_config_init0.hxx"),
        1: parse_hxx(CFG_DIR / "sensor_config_init1.hxx"),
    }


NEAR_F1_0 = NEAR_FAR_DIR / "adc_近波_20260512141857_f1_0_14_19_05_965.txt"
PF0_RX0 = MATLAB_DATA / "adc_test_20260527115033_Pf0_Rx0.txt"


def test_real_headerless_near_file(real_cfgs):
    loaded = load_adc_file(NEAR_F1_0)
    assert loaded.format_name == "headerless"
    assert loaded.token_count == 131073
    assert loaded.payload_count == 131073  # 期望 131072，恰多 1 个尾零
    assert loaded.bad_tokens == []

    fa = audit_file(loaded, real_cfgs, REPO_ROOT)
    assert fa.filename_profile == 1 and fa.filename_rx == 0
    assert fa.config.profile == 1
    assert fa.finding("FMT-01").status == "pass"

    lf = fa.finding("LEN-01")
    assert lf.status == "warn"
    assert lf.details["expected"] == 131072
    assert lf.details["extra"] == 1
    assert lf.details["extra_all_zero"] is True
    assert lf.details["repair"] == "truncate-tail"
    assert fa.quality is not None
    assert fa.quality.n_real_samples == 262144
    assert fa.quality.n_real_chirps == 128


def test_real_header3_pf0_file(real_cfgs):
    loaded = load_adc_file(PF0_RX0)
    assert loaded.format_name == "header3"
    assert loaded.header == (0, 1024, 256)
    assert loaded.token_count == 131076  # 3 头字段 + 131073 载荷
    assert loaded.payload_count == 131073

    fa = audit_file(loaded, real_cfgs, REPO_ROOT)
    assert fa.filename_profile == 0 and fa.filename_rx == 0
    # 头字段 1024/256 与 init0 的 rng_nfft/nchirp 交叉核对通过
    assert fa.finding("FMT-01").status == "pass"
    lf = fa.finding("LEN-01")
    assert lf.status == "warn" and lf.details["extra"] == 1


def test_real_full_run(real_cfgs):
    """两组目录、两个配置一起审计：16 文件，跨批次重复=4 对，全部仅警告。"""
    report = audit([NEAR_FAR_DIR, MATLAB_DATA], list(real_cfgs.values()))
    assert len(report.files) == 16

    # 每个文件的载荷都比期望恰多 1 个尾零 -> LEN-01 警告且显式说明修复
    for fa in report.files:
        lf = fa.finding("LEN-01")
        assert lf.status == "warn", (fa.repo_rel_path, lf.message)
        assert lf.details["extra"] == 1
        assert lf.details["extra_all_zero"] is True
        assert "截断修复" in lf.message

    # A 组近波与 B 组 f1 为同一批数据（sha256 相同）-> 4 条重复批次警告
    bat01 = [f for f in report.batch_findings if f.rule_id == "BAT-01"]
    assert len(bat01) == 4
    assert all(f.status == "warn" for f in bat01)
    assert all(len(f.details["groups"]) == 2 for f in bat01)

    # 4 个批次（A/f0、A/f1、B/f0(Pf0)、B/f1）通道完整
    bat02 = [f for f in report.batch_findings if f.rule_id == "BAT-02"]
    assert len(bat02) == 4
    assert all(f.status == "pass" for f in bat02), [f.message for f in bat02]

    # 真实数据无任何失败项
    assert report.n_fail == 0
    assert report.exit_code() == 0
