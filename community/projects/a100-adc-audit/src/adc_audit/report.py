"""审计结果输出：JSON（机器可读）与 Markdown（人读摘要）。"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

from . import __version__
from .config import SensorConfig
from .rules import AuditReport, RULES

_STATUS_GLYPH = {"pass": "✓ 通过", "warn": "! 警告", "fail": "✗ 失败"}
_STATUS_MARK = {"pass": "pass", "warn": "warn", "fail": "fail"}


def _now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _display_path(p: str | Path, repo_root: Path | None) -> str:
    """仓内路径显示为仓库相对路径，仓外保持绝对路径（报告可复现、无本机绝对路径）。"""
    try:
        rp = Path(p).resolve()
        if repo_root is not None:
            return str(rp.relative_to(Path(repo_root).resolve()))
    except ValueError:
        pass
    return str(p)


def build_json(report: AuditReport) -> dict:
    return {
        "tool": "adc_audit",
        "version": __version__,
        "generated_at": _now_iso(),
        "data_dirs": [_display_path(d, report.repo_root) for d in report.data_dirs],
        "num_rx": report.num_rx,
        "configs": [
            {**c.as_dict(), "path": _display_path(c.path, report.repo_root)}
            for c in report.configs
        ],
        "rules": {rid: {"title": s.title, "description": s.description, "level": s.level}
                  for rid, s in RULES.items()},
        "summary": {
            "files": len(report.files),
            "file_status": {
                "pass": sum(1 for f in report.files if f.status == "pass"),
                "warn": sum(1 for f in report.files if f.status == "warn"),
                "fail": sum(1 for f in report.files if f.status == "fail"),
            },
            "rule_counts": report.rule_counts(),
            "exit_code": report.exit_code(),
            "exit_code_strict": report.exit_code(strict=True),
        },
        "files": [f.as_dict() for f in report.files],
        "batch_findings": [f.as_dict() for f in report.batch_findings],
    }


def write_json(report: AuditReport, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(build_json(report), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def _fmt_len(fa) -> str:
    f = fa.finding("LEN-01")
    exp = fa.config.expected_words if fa.config else None
    act = fa.loaded.payload_count
    base = f"{act}/{exp}"
    if f is None:
        return base
    if f.status == "pass":
        return f"{base} ="
    if "extra" in f.details:
        return f"{base} +{f.details['extra']}多"
    return f"{base} -{f.details.get('missing', '?')}缺"


def build_markdown(report: AuditReport, figure_links: list[tuple[str, str]] | None = None) -> str:
    lines: list[str] = []
    a = lines.append
    a("# CTSAI-A100 ADC 批次完整性与质量审计报告")
    a("")
    a(f"- 工具：`adc_audit` v{__version__}（离线分析，未做硬件实测）")
    a(f"- 生成时间：{_now_iso()}")
    for d in report.data_dirs:
        a(f"- 数据目录：`{_display_path(d, report.repo_root)}`")
    for c in report.configs:
        a(
            f"- 配置：`{c.path.name}`（init{c.profile}: {c.start_freq_ghz} GHz / "
            f"{c.bandwidth_mhz} MHz / {c.nchirp} chirp / rng_nfft {c.rng_nfft} / "
            f"vel_nfft {c.vel_nfft} / 期望 {c.expected_words} 字/通道）"
        )
    a("")

    a("## 1. 总体结论")
    a("")
    s = report.rule_counts()
    a(f"- 审计文件数：**{len(report.files)}**")
    a(
        f"- 文件级结论：{_STATUS_GLYPH['pass'].split()[0]} 通过 "
        f"{sum(1 for f in report.files if f.status == 'pass')} / "
        f"{_STATUS_GLYPH['warn'].split()[0]} 警告 "
        f"{sum(1 for f in report.files if f.status == 'warn')} / "
        f"{_STATUS_GLYPH['fail'].split()[0]} 失败 "
        f"{sum(1 for f in report.files if f.status == 'fail')}"
    )
    bf = sum(1 for x in report.batch_findings if x.status == "fail")
    bw = sum(1 for x in report.batch_findings if x.status == "warn")
    a(f"- 批次级结论：失败 {bf} / 警告 {bw}")
    a(f"- 退出码：{report.exit_code()}（0=无失败；警告不改变退出码，`--strict` 下警告也返回 1）")
    a("")
    a("| 规则 | 通过 | 警告 | 失败 |")
    a("|---|---:|---:|---:|")
    for rid in RULES:
        a(f"| {rid} {RULES[rid].title} | {s[rid]['pass']} | {s[rid]['warn']} | {s[rid]['fail']} |")
    a("")

    a("## 2. 文件摘要表")
    a("")
    a(
        "| # | 标签 | 文件（仓库相对路径） | 格式 | 载荷/期望 | "
        "饱和率 | 最长游程 | 全零chirp | DC | RMS | 总体 |"
    )
    a("|---:|---|---|---|---|---:|---:|---:|---:|---:|---|")
    for i, f in enumerate(report.files, 1):
        q = f.quality
        fmt = "三字段头" if f.loaded.header else "无头"
        if q:
            sat = f"{q.sat_ratio * 100:.4f}%"
            run = f"{q.longest_run}"
            zc = f"{q.zero_chirps}/{q.n_real_chirps}"
            dc, rms = f"{q.dc:.1f}", f"{q.rms:.1f}"
        else:
            sat = run = zc = dc = rms = "-"
        a(
            f"| {i} | `{f.short_label}` | `{f.repo_rel_path}` | {fmt} | "
            f"{_fmt_len(f)} | {sat} | {run} | {zc} | {dc} | {rms} | "
            f"{_STATUS_MARK[f.status]} |"
        )
    a("")

    a("## 3. 批次级结论")
    a("")
    for f in report.batch_findings:
        a(f"- **{f.rule_id}** [{_STATUS_GLYPH[f.status]}] {f.message}")
    a("")

    a("## 4. 关键发现（全部警告/失败，无静默修补）")
    a("")
    flagged = [f for f in report.batch_findings if f.status != "pass"]
    for f in report.files:
        for finding in f.findings:
            if finding.status != "pass":
                flagged.append(f)
    if not flagged:
        a("- 无（全部规则通过）")
    for f in flagged:
        if hasattr(f, "repo_rel_path"):
            a(f"- `{f.short_label}` {f.repo_rel_path}")
            for finding in f.findings:
                if finding.status != "pass":
                    a(f"  - **{finding.rule_id}** [{_STATUS_GLYPH[finding.status]}] {finding.message}")
        else:
            a(f"- **{f.rule_id}** [{_STATUS_GLYPH[f.status]}] {f.message}")
    a("")

    a("## 5. 各文件明细")
    a("")
    for f in report.files:
        a(f"### {f.short_label} — `{f.repo_rel_path}`")
        a("")
        a(f"- sha256: `{f.loaded.sha256}`")
        a(f"- 格式: {f.loaded.format_name}"
          + (f"，头={list(f.loaded.header)}" if f.loaded.header else ""))
        a(f"- token 数: {f.loaded.token_count}，合法载荷: {f.loaded.payload_count}"
          f"，期望: {f.config.expected_words if f.config else '?'}")
        if f.repair:
            a(f"- **显式修复**: {f.repair}")
        for finding in f.findings:
            a(f"- {finding.rule_id} [{_STATUS_GLYPH[finding.status]}] {finding.message}")
        a("")

    if figure_links:
        a("## 6. 图表")
        a("")
        for title, rel in figure_links:
            a(f"### {title}")
            a("")
            a(f"![{title}]({rel})")
            a("")
    return "\n".join(lines) + "\n"


def write_markdown(
    report: AuditReport, path: str | Path, figure_links: list[tuple[str, str]] | None = None
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build_markdown(report, figure_links), encoding="utf-8")
    return path
