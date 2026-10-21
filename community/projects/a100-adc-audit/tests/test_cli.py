"""CLI 集成测试：退出码约定、JSON/Markdown 报告落盘。"""

import json

import pytest

from adc_audit.cli import main
from conftest import STD_EXPECTED, make_channel_dir, make_words, write_adc


def test_cli_good_batch(tmp_path, std_config):
    d = make_channel_dir(tmp_path / "data")
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    rc = main([
        "--data-dir", str(d),
        "--config", str(std_config.path),
        "--json", str(out_json),
        "--md", str(out_md),
        "--figures-dir", str(tmp_path / "figs"),
        "--quiet",
    ])
    assert rc == 0
    data = json.loads(out_json.read_text(encoding="utf-8"))
    assert data["summary"]["files"] == 4
    assert data["summary"]["file_status"]["pass"] == 4
    assert data["summary"]["exit_code"] == 0
    assert out_md.exists()
    assert (tmp_path / "figs" / "rule_matrix.png").exists()
    assert (tmp_path / "figs" / "quality_metrics.png").exists()
    md = out_md.read_text(encoding="utf-8")
    assert "文件摘要表" in md and "rule_matrix.png" in md


def test_cli_failure_exit_code(tmp_path, std_config):
    d = tmp_path / "data"
    d.mkdir()
    write_adc(d / "adc_test_20260512141857_f1_0_14_19_00_000.txt",
              make_words(STD_EXPECTED, seed=71)[:1800])  # 截断
    rc = main([
        "--data-dir", str(d),
        "--config", str(std_config.path),
        "--json", str(tmp_path / "out.json"),
        "--quiet",
    ])
    assert rc == 1


def test_cli_strict_warns_exit_code(tmp_path, std_config):
    # 缺通道 -> 仅警告：默认退出 0，--strict 退出 1
    d = make_channel_dir(tmp_path / "data", n_rx=1)
    args = ["--data-dir", str(d), "--config", str(std_config.path), "--quiet"]
    assert main(args) == 0
    assert main([*args, "--strict"]) == 1


def test_cli_bad_config_path(tmp_path, std_config):
    d = make_channel_dir(tmp_path / "data")
    rc = main(["--data-dir", str(d), "--config",
               str(tmp_path / "nope.hxx"), "--quiet"])
    assert rc == 2


def test_cli_missing_data_dir(tmp_path, std_config):
    rc = main(["--data-dir", str(tmp_path / "nodir"),
               "--config", str(std_config.path), "--quiet"])
    assert rc == 2


def test_cli_data_dir_without_adc_files(tmp_path, std_config, capsys):
    # 目录存在但没有匹配的 ADC 文件：必须显式报错（退出码 2），
    # 不能"审计 0 个文件"后返回 0 造成未执行审计却成功的假象
    d = tmp_path / "data"
    d.mkdir()
    (d / "notes.txt").write_text("not adc data", encoding="utf-8")
    rc = main(["--data-dir", str(d), "--config", str(std_config.path), "--quiet"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "未找到任何 ADC 文件" in err
