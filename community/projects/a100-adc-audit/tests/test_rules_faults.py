"""规则引擎故障注入回归：每种故障都必须被正确分级报告，完好文件不误报。

覆盖：截断 10% / 删除中间 100 值 / 非数值 token / uint32 越界（>2^32-1 与负数）/
重复通道文件 / 头字段与配置不符 / 头 rx 与文件名不符 / 跨批次重复 / 缺通道。
"""

import numpy as np
import pytest

from adc_audit.rules import audit
from conftest import (
    STD_EXPECTED,
    make_channel_dir,
    make_words,
    write_adc,
)


@pytest.fixture()
def audit_env(tmp_path, std_config):
    """单目录 4 通道完好批次。"""
    d = make_channel_dir(tmp_path / "data", n_words=STD_EXPECTED)
    return d, std_config


def findings_by_rule(fa):
    return {f.rule_id: f for f in fa.findings}


# ---------------------------------------------------------------------------
# 完好文件：不误报
# ---------------------------------------------------------------------------


def test_intact_batch_all_pass(audit_env):
    d, cfg = audit_env
    report = audit([d], [cfg])
    assert len(report.files) == 4
    for fa in report.files:
        assert fa.status == "pass", [f.as_dict() for f in fa.findings]
        assert fa.finding("LEN-01").status == "pass"
        assert fa.finding("VAL-01").status == "pass"
        assert fa.finding("SIG-01").status == "pass"
        assert fa.finding("FMT-01").status == "pass"
        assert fa.quality.n_real_samples == STD_EXPECTED * 2
        assert fa.quality.zero_chirps == 0
        assert fa.quality.sat_ratio == 0.0
    assert report.n_fail == 0 and report.n_warn == 0
    assert report.exit_code() == 0
    # BAT-02 全部通过
    assert all(f.status == "pass" for f in report.batch_findings)


# ---------------------------------------------------------------------------
# LEN-01：截断 / 删除中间值
# ---------------------------------------------------------------------------


def test_truncated_10_percent(tmp_path, std_config):
    words = make_words(STD_EXPECTED, seed=11)
    keep = int(STD_EXPECTED * 0.9)  # 截断 10%
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt", words[:keep])
    report = audit([d], [std_config])
    fa = report.files[0]
    lf = fa.finding("LEN-01")
    assert lf.status == "fail"
    assert lf.details["missing"] == STD_EXPECTED - keep
    assert lf.details["repair"] == "zero-pad-tail"
    assert "补零" in lf.message
    # 补零不参与质量统计（不产生虚假全零 chirp 告警）
    assert fa.finding("SIG-01").status == "pass"
    assert fa.quality.n_real_chirps == keep // 128
    assert report.exit_code() == 1


def test_delete_middle_100_values(tmp_path, std_config):
    words = make_words(STD_EXPECTED, seed=12)
    broken = np.concatenate([words[:1000], words[1100:]])
    assert broken.size == STD_EXPECTED - 100
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt", broken)
    report = audit([d], [std_config])
    lf = report.files[0].finding("LEN-01")
    assert lf.status == "fail"
    assert lf.details["missing"] == 100
    assert lf.details["position"] == "unknown"


def test_extra_trailing_zero_warns(tmp_path, std_config):
    words = list(make_words(STD_EXPECTED, seed=13)) + [0]
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt", words)
    report = audit([d], [std_config])
    lf = report.files[0].finding("LEN-01")
    assert lf.status == "warn"
    assert lf.details["extra"] == 1
    assert lf.details["extra_all_zero"] is True
    assert lf.details["repair"] == "truncate-tail"
    assert report.exit_code() == 0  # 警告不改退出码


def test_extra_nonzero_tail_fails(tmp_path, std_config):
    words = list(make_words(STD_EXPECTED, seed=14)) + [12345]
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt", words)
    report = audit([d], [std_config])
    lf = report.files[0].finding("LEN-01")
    assert lf.status == "fail"
    assert lf.details["extra"] == 1
    assert lf.details["extra_all_zero"] is False


# ---------------------------------------------------------------------------
# VAL-01：非数值 / 越界
# ---------------------------------------------------------------------------


def test_non_numeric_token(tmp_path, std_config):
    vals = [str(w) for w in make_words(STD_EXPECTED, seed=15)]
    vals[500] = "NaN"
    d = tmp_path / "data"
    d.mkdir()
    (d / "adc_test_20260512141857_f1_0_14_19_00_000.txt").write_text(
        ",".join(vals), encoding="utf-8"
    )
    report = audit([d], [std_config])
    vf = report.files[0].finding("VAL-01")
    assert vf.status == "fail"
    assert vf.details["total_bad"] == 1
    assert vf.details["bad_tokens"][0]["index"] == 500
    assert vf.details["bad_tokens"][0]["reason"] == "non-numeric"
    assert report.exit_code() == 1


@pytest.mark.parametrize("bad,reason", [
    ("4294967296", "out-of-range(>2^32-1)"),
    ("-7", "out-of-range(<0)"),
])
def test_uint32_out_of_range(tmp_path, std_config, bad, reason):
    vals = [str(w) for w in make_words(STD_EXPECTED, seed=16)]
    vals[10] = bad
    d = tmp_path / "data"
    d.mkdir()
    (d / "adc_test_20260512141857_f1_0_14_19_00_000.txt").write_text(
        ",".join(vals), encoding="utf-8"
    )
    report = audit([d], [std_config])
    vf = report.files[0].finding("VAL-01")
    assert vf.status == "fail"
    assert vf.details["bad_tokens"][0]["reason"] == reason


# ---------------------------------------------------------------------------
# FMT-01：头字段与配置不符
# ---------------------------------------------------------------------------


def test_header_config_mismatch(tmp_path, std_config):
    d = tmp_path / "data"
    d.mkdir()
    write_adc(
        d / "adc_test_20260512141857_f1_0_14_19_00_000.txt",
        make_words(STD_EXPECTED, seed=17),
        header=(0, 512, 8),  # 与配置 rng_nfft=256/nchirp=16 不符
    )
    report = audit([d], [std_config])
    ff = report.files[0].finding("FMT-01")
    assert ff.status == "fail"
    assert "不一致" in ff.message


def test_header_matches_config(tmp_path, std_config):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(4):
        write_adc(
            d / f"adc_test_20260527115033_Pf1_Rx{i}.txt",
            make_words(STD_EXPECTED, seed=20 + i),
            header=(i, std_config.rng_nfft, std_config.nchirp),
        )
    report = audit([d], [std_config])
    for fa in report.files:
        assert fa.finding("FMT-01").status == "pass"
    assert report.n_fail == 0


# ---------------------------------------------------------------------------
# BAT-01 / BAT-02：批次一致性
# ---------------------------------------------------------------------------


def test_duplicate_channel_file(tmp_path, std_config):
    d = tmp_path / "data"
    d.mkdir()
    words = make_words(STD_EXPECTED, seed=30)
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt", words)
    # Rx1 文件与 Rx0 内容完全相同 -> 同批不同 Rx 内容相同
    write_adc(d / "adc_test_20260512141857_f1_1_14_19_01_000.txt", words)
    write_adc(d / "adc_test_20260512141857_f1_2_14_19_02_000.txt",
              make_words(STD_EXPECTED, seed=32))
    write_adc(d / "adc_test_20260512141857_f1_3_14_19_03_000.txt",
              make_words(STD_EXPECTED, seed=33))
    report = audit([d], [std_config])
    bat02 = [f for f in report.batch_findings if f.rule_id == "BAT-02"]
    assert len(bat02) == 1 and bat02[0].status == "fail"
    assert "内容完全相同" in bat02[0].message
    assert report.exit_code() == 1


def test_header_rx_vs_filename_rx(tmp_path, std_config):
    d = tmp_path / "data"
    d.mkdir()
    for i in range(4):
        write_adc(
            d / f"adc_test_20260527115033_Pf1_Rx{i}.txt",
            make_words(STD_EXPECTED, seed=40 + i),
            header=((2 if i == 0 else i), std_config.rng_nfft, std_config.nchirp),
        )
    report = audit([d], [std_config])
    bat02 = [f for f in report.batch_findings if f.rule_id == "BAT-02"]
    assert bat02[0].status == "fail"
    assert "头部 rx=2 与文件名 Rx0 不一致" in bat02[0].message


def test_duplicate_rx_index(tmp_path, std_config):
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt",
              make_words(STD_EXPECTED, seed=50))
    write_adc(d / "adc_test_20260512141857_f1_0_14_20_00_000.txt",
              make_words(STD_EXPECTED, seed=51))
    write_adc(d / "adc_test_20260512141857_f1_2_14_19_02_000.txt",
              make_words(STD_EXPECTED, seed=52))
    write_adc(d / "adc_test_20260512141857_f1_3_14_19_03_000.txt",
              make_words(STD_EXPECTED, seed=53))
    report = audit([d], [std_config])
    bat02 = [f for f in report.batch_findings if f.rule_id == "BAT-02"]
    assert bat02[0].status == "fail"
    assert "Rx 序号重复" in bat02[0].message


def test_duplicate_batch_across_dirs(tmp_path, std_config):
    d1 = make_channel_dir(tmp_path / "dirA")
    d2 = make_channel_dir(tmp_path / "dirB")
    report = audit([d1, d2], [std_config])
    bat01 = [f for f in report.batch_findings if f.rule_id == "BAT-01"]
    assert len(bat01) == 4  # 4 对内容相同
    assert all(f.status == "warn" for f in bat01)
    assert all("重复批次" in f.message for f in bat01)
    assert report.exit_code() == 0  # 重复批次=警告


def test_missing_channel_warns(tmp_path, std_config):
    d = make_channel_dir(tmp_path / "data", n_rx=2)
    report = audit([d], [std_config])
    bat02 = [f for f in report.batch_findings if f.rule_id == "BAT-02"]
    assert bat02[0].status == "warn"
    assert "缺通道" in bat02[0].message
    assert report.exit_code() == 0


# ---------------------------------------------------------------------------
# SIG-01：信号质量分级
# ---------------------------------------------------------------------------


def test_saturation_graded(tmp_path, std_config):
    # 构造 >=1% 饱和的样本：高/低 16 位同时打满（0x7FFF / 0x8000）
    words = make_words(STD_EXPECTED, seed=60).copy()
    n_sat_words = int(STD_EXPECTED * 0.015)
    sat = np.full(n_sat_words, (0x7FFF << 16) | 0x8000, dtype=np.uint64).astype(np.uint32)
    words[:n_sat_words] = sat
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt", words)
    report = audit([d], [std_config])
    fa = report.files[0]
    sf = fa.finding("SIG-01")
    assert sf.status == "fail"
    # 每个饱和字贡献 hi=+32767 与 lo=-32768 各 1 个样本
    assert fa.quality.sat_ratio == pytest.approx(n_sat_words / STD_EXPECTED)
    assert fa.quality.sat_pos == n_sat_words
    assert fa.quality.sat_neg == n_sat_words


def test_all_zero_file(tmp_path, std_config):
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt",
              np.zeros(STD_EXPECTED, dtype=np.uint32))
    report = audit([d], [std_config])
    fa = report.files[0]
    assert fa.finding("SIG-01").status == "fail"
    assert fa.quality.zero_chirps == std_config.nchirp
    assert fa.quality.dc == 0.0 and fa.quality.rms == 0.0


def test_constant_chirp_long_run(tmp_path, std_config):
    # 一个整 chirp 常值（非零）-> 最长游程 >= rng_nfft -> 警告
    words = make_words(STD_EXPECTED, seed=61).copy()
    wpc = std_config.rng_nfft // 2
    words[:wpc] = (100 << 16) | 100
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt", words)
    report = audit([d], [std_config])
    fa = report.files[0]
    sf = fa.finding("SIG-01")
    assert sf.status == "warn"
    assert fa.quality.longest_run == std_config.rng_nfft
    assert fa.quality.longest_run_value == 100


# ---------------------------------------------------------------------------
# FMT-01：唯一配置与文件名 f0/f1 标记的关系（review 回归）
# ---------------------------------------------------------------------------


def test_single_config_filename_mismatch_fails(tmp_path, std_config):
    # 文件名明确标记 f0，但只提供了 init1 配置：明确不匹配必须判失败，
    # 且不再用错误的配置审计长度/信号
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f0_0_14_19_00_000.txt",
              make_words(STD_EXPECTED, seed=70))
    report = audit([d], [std_config])
    fa = report.files[0]
    ff = fa.finding("FMT-01")
    assert ff.status == "fail"
    assert "不匹配" in ff.message
    assert fa.config is None
    assert fa.finding("LEN-01").status == "fail"
    assert fa.status == "fail"
    assert report.exit_code() == 1


def test_single_config_no_marker_still_warns(tmp_path, std_config):
    # 文件名无任何 f0/f1 标记时维持警告（不能确定 profile，不算明确不匹配）
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_capture_20260512_142200.txt",
              make_words(STD_EXPECTED, seed=71))
    report = audit([d], [std_config])
    fa = report.files[0]
    ff = fa.finding("FMT-01")
    assert ff.status == "warn"
    assert "未含" in ff.message
    assert fa.config is not None
    assert fa.finding("LEN-01").status == "pass"


def test_same_dir_name_different_paths_not_merged(tmp_path, std_config):
    # 两条不同采集路径的末级目录同名（都叫 data）：不得合并为同一批次；
    # 相同内容应报跨批次重复（BAT-01），而不是被同名目录吞掉
    d1 = make_channel_dir(tmp_path / "a" / "data")
    d2 = tmp_path / "b" / "data"
    d2.mkdir(parents=True)
    write_adc(d2 / "adc_test_20260512141857_f1_0_14_19_00_000.txt",
              make_words(STD_EXPECTED, seed=100))  # 与 a/data Rx0 同内容
    report = audit([d1, d2], [std_config])
    labels = {fa.group_label for fa in report.files}
    assert len(labels) == 2  # 完整路径区分两条采集路径
    bat01 = [f for f in report.batch_findings if f.rule_id == "BAT-01"]
    assert len(bat01) == 1
    assert len(bat01[0].details["groups"]) == 2
    assert "重复批次" in bat01[0].message
    assert report.exit_code() == 0  # 跨批次重复仍是警告
