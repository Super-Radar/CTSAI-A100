"""命令行入口。

用法（在本项目目录 community/projects/a100-adc-audit/ 下）::

    PYTHONPATH=src python -m adc_audit \
        --data-dir <dir> [--data-dir <dir> ...] \
        --config <sensor_config_initN.hxx> [--config <...>] \
        [--json out.json] [--md out.md] \
        [--figures-dir DIR] [--num-rx 4] [--strict] [--quiet]

退出码：0 = 无失败（警告不改变退出码）；1 = 存在失败
（`--strict` 时存在警告也返回 1）；2 = 输入/配置错误
（配置解析失败、目录不存在、目录中没有任何 ADC 文件）。
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__
from .config import ConfigError, parse_hxx
from .figures import plot_quality_metrics, plot_rule_matrix
from .report import write_json, write_markdown
from .rules import (
    NUM_RX_DEFAULT,
    AuditReport,
    audit,
    find_repo_root,
)

_GLYPH = {"pass": "✓", "warn": "!", "fail": "✗"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="adc_audit",
        description="CTSAI-A100 ADC 批次完整性与质量审计工具（离线分析）",
    )
    p.add_argument("--data-dir", action="append", required=True,
                   help="ADC 数据目录（可重复传入多个以做跨批次审计）")
    p.add_argument("--config", action="append", required=True,
                   help="sensor_config_initN.hxx 配置（可重复；按文件名 f0/f1 自动匹配）")
    p.add_argument("--json", metavar="PATH", help="JSON 报告输出路径")
    p.add_argument("--md", metavar="PATH", help="Markdown 报告输出路径")
    p.add_argument("--figures-dir", metavar="DIR",
                   help="图表输出目录（默认与 --md 同目录；传空串跳过）")
    p.add_argument("--num-rx", type=int, default=NUM_RX_DEFAULT,
                   help="每批期望的接收通道数（默认 4）")
    p.add_argument("--strict", action="store_true",
                   help="存在警告时也返回退出码 1")
    p.add_argument("--quiet", action="store_true", help="只输出摘要")
    p.add_argument("--version", action="version", version=f"adc_audit {__version__}")
    return p


def run(args: argparse.Namespace) -> int:
    try:
        configs = [parse_hxx(c) for c in args.config]
    except (ConfigError, OSError) as exc:
        print(f"配置解析失败: {exc}", file=sys.stderr)
        return 2

    try:
        report = audit(args.data_dir, configs, num_rx=args.num_rx)
    except FileNotFoundError as exc:
        print(f"错误: {exc}", file=sys.stderr)
        return 2

    _print_report(report, quiet=args.quiet)

    figure_links: list[tuple[str, str]] = []
    if args.figures_dir != "":
        fig_dir = Path(args.figures_dir) if args.figures_dir else (
            Path(args.md).parent if args.md else Path("figures"))
        repo_root = find_repo_root(Path(__file__))
        # 图表始终落盘；报告中以相对路径引用
        f1 = plot_rule_matrix(report, fig_dir / "rule_matrix.png")
        f2 = plot_quality_metrics(report, fig_dir / "quality_metrics.png")
        if args.md:
            md_path = Path(args.md).resolve()
            for title, png in (("规则通过矩阵", f1), ("质量指标", f2)):
                try:
                    rel = png.resolve().relative_to(md_path.parent)
                except ValueError:
                    rel = _rel_or_abs(png.resolve(), md_path.parent, repo_root)
                figure_links.append((title, str(rel).replace("\\", "/")))

    if args.json:
        out = write_json(report, args.json)
        print(f"JSON 报告: {out}")
    if args.md:
        out = write_markdown(report, args.md, figure_links)
        print(f"Markdown 报告: {out}")

    code = report.exit_code(strict=args.strict)
    print(f"退出码 {code}"
          f"（0=无失败，1={'存在失败或警告' if args.strict else '存在失败'}）")
    return code


def _rel_or_abs(target: Path, base: Path, repo_root: Path | None) -> Path:
    return Path(os.path.relpath(target, base))


def _print_report(report: AuditReport, quiet: bool = False) -> None:
    print(f"adc_audit: 审计 {len(report.files)} 个文件，"
          f"{len(report.batch_findings)} 条批次结论")
    if not quiet:
        for f in report.files:
            marks = " ".join(
                f"{fd.rule_id}:{_GLYPH[fd.status]}" for fd in f.findings
            )
            print(f"  [{_GLYPH[f.status]}] {f.short_label:<12} {f.repo_rel_path}  {marks}")
    for fd in report.batch_findings:
        print(f"  [{_GLYPH[fd.status]}] {fd.rule_id} {fd.message}")
    counts = report.rule_counts()
    parts = [
        f"{rid}:{counts[rid]['pass']}✓/{counts[rid]['warn']}!/{counts[rid]['fail']}✗"
        for rid in counts
    ]
    print("规则统计  " + "  ".join(parts))
    print(f"总体: 失败 {report.n_fail}，警告 {report.n_warn}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
