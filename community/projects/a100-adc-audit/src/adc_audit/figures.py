"""审计图表输出（matplotlib，Agg 后端，PNG）。

配色遵循可访问性约定：
- 状态色（通过/警告/失败）仅用于状态，且始终配文字符号，不单靠颜色表意；
- 量值条形图使用单一蓝色渐变（顺序编码），细条幅、隐性网格、无顶右边框；
- 不同量纲的指标拆分为独立子图，绝不使用双 Y 轴。
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from .rules import AuditReport, RULES, worst_status  # noqa: E402

# 调色板（浅色表面）
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
STATUS = {"pass": "#0ca30c", "warn": "#fab219", "fail": "#d03b3b"}
STATUS_GLYPH = {"pass": "✓", "warn": "!", "fail": "×"}
SERIES_BLUE = "#2a78d6"
SERIES_BLUE_DEEP = "#1c5cab"


def _cjk_font() -> str | None:
    for name in ("WenQuanYi Micro Hei", "WenQuanYi Zen Hei", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(name, fallback_to_default=False)
            return name
        except Exception:
            continue
    return None


def _new_axes(fig, rect=111):
    ax = fig.add_subplot(rect)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelsize=8)
    return ax


def _batch_status_for_file(report: AuditReport, fa, rule_id: str) -> str:
    """文件在指定批次级规则中的牵连状态（该规则单独计，不与其它规则混并）。"""
    statuses = []
    for fd in report.batch_findings:
        if fd.rule_id != rule_id:
            continue
        if rule_id == "BAT-01":
            files = fd.details.get("files") or []
            if fa.repo_rel_path in files:
                statuses.append(fd.status)
        elif rule_id == "BAT-02":
            if fd.details.get("group") == fa.group_label:
                statuses.append(fd.status)
    return worst_status(statuses) if statuses else "pass"


def plot_rule_matrix(report: AuditReport, path: str | Path) -> Path:
    """规则通过矩阵：文件 × 规则，状态色 + 文字符号。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    font = _cjk_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rule_ids = list(RULES)
    files = report.files
    n, m = len(files), len(rule_ids)

    fig = plt.figure(figsize=(9.6, 0.42 * n + 2.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax = _new_axes(fig)
    ax.set_facecolor(SURFACE)

    status_by_cell: list[list[str]] = []
    for f in files:
        per_rule = {fd.rule_id: fd.status for fd in f.findings}
        row = []
        for rid in rule_ids:
            if rid in per_rule:
                row.append(per_rule[rid])
            else:
                row.append(_batch_status_for_file(report, f, rid))
        status_by_cell.append(row)

    for i in range(n):
        for j in range(m):
            st = status_by_cell[i][j]
            ax.add_patch(
                plt.Rectangle(
                    (j, n - 1 - i),
                    1,
                    1,
                    facecolor=STATUS[st],
                    edgecolor=SURFACE,
                    linewidth=2,
                )
            )
            ax.text(
                j + 0.5,
                n - 1 - i + 0.5,
                STATUS_GLYPH[st],
                ha="center",
                va="center",
                fontsize=9,
                color=INK,
            )

    ax.set_xlim(0, m)
    ax.set_ylim(0, n)
    ax.set_xticks([j + 0.5 for j in range(m)])
    ax.set_xticklabels(rule_ids, color=INK_2)
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks([n - 1 - i + 0.5 for i in range(n)])
    ax.set_yticklabels(
        [f"{f.short_label}" for f in files], color=INK_2, fontfamily="monospace"
    )
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)

    title = "规则通过矩阵（✓ 通过 / ! 警告 / × 失败）" if font else "Rule outcome matrix (pass / warn / fail)"
    ax.set_title(title, color=INK, fontsize=11, pad=14)

    legend_items = [
        Patch(facecolor=STATUS["pass"], edgecolor=SURFACE, label="通过 pass"),
        Patch(facecolor=STATUS["warn"], edgecolor=SURFACE, label="警告 warn"),
        Patch(facecolor=STATUS["fail"], edgecolor=SURFACE, label="失败 fail"),
    ]
    ax.legend(
        handles=legend_items,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=3,
        frameon=False,
        fontsize=8,
        labelcolor=INK_2,
    )

    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_quality_metrics(report: AuditReport, path: str | Path) -> Path:
    """每文件质量指标：上=最大绝对样本（满量程参考线），下=RMS。单轴双子图。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    font = _cjk_font()
    if font:
        plt.rcParams["font.sans-serif"] = [font, "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    files = [f for f in report.files if f.quality is not None]
    labels = [f.short_label for f in files]
    max_abs = [f.quality.max_abs for f in files]
    rms = [f.quality.rms for f in files]
    sat = [f.quality.sat_ratio for f in files]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.6, 6.4), dpi=150, sharex=True, gridspec_kw={"hspace": 0.32}
    )
    for ax in (ax1, ax2):
        ax.set_facecolor(SURFACE)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(BASELINE)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    fig.patch.set_facecolor(SURFACE)
    x = range(len(files))

    ax1.bar(x, max_abs, width=0.62, color=SERIES_BLUE, edgecolor=SURFACE, linewidth=1)
    ax1.axhline(32768, color=MUTED, linestyle="--", linewidth=1)
    ax1.text(
        len(files) - 0.4,
        32768,
        "满量程 32768" if font else "full scale 32768",
        va="bottom",
        ha="right",
        fontsize=8,
        color=MUTED,
    )
    k = max(range(len(max_abs)), key=lambda i: max_abs[i]) if max_abs else None
    if k is not None:
        ax1.text(k, max_abs[k], f"{max_abs[k]}", ha="center", va="bottom", fontsize=8, color=INK_2)
    t1 = "解包样本最大绝对值 max|x|（越接近满量程越可能饱和）" if font else "max |sample| per file"
    ax1.set_title(t1, color=INK, fontsize=10, loc="left")
    ax1.set_ylabel("LSB", color=INK_2, fontsize=9)

    ax2.bar(x, rms, width=0.62, color=SERIES_BLUE_DEEP, edgecolor=SURFACE, linewidth=1)
    k2 = max(range(len(rms)), key=lambda i: rms[i]) if rms else None
    if k2 is not None:
        ax2.text(k2, rms[k2], f"{rms[k2]:.0f}", ha="center", va="bottom", fontsize=8, color=INK_2)
    t2 = "RMS（信号能量水平，各通道应同量级）" if font else "RMS per file"
    ax2.set_title(t2, color=INK, fontsize=10, loc="left")
    ax2.set_ylabel("LSB", color=INK_2, fontsize=9)
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=45, ha="right", fontfamily="monospace", color=INK_2)

    n_sat = sum(1 for s in sat if s > 0)
    note = (
        f"饱和文件数：{n_sat}/{len(files)}（正负满量程样本计数见报告表）"
        if font
        else f"files with saturation: {n_sat}/{len(files)}"
    )
    fig.text(0.01, 0.012, note, fontsize=8, color=MUTED)

    fig.subplots_adjust(left=0.09, right=0.985, top=0.92, bottom=0.16, hspace=0.38)
    fig.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path
