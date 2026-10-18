"""Reproduce every experiment, figure and result file of the study.

Usage (from the project root, offline, no hardware required)::

    python src/fixed_fft/cli.py                 # default seed 20261003
    python src/fixed_fft/cli.py --seed 20261003 --data-root <dir>

Outputs ``results/*.csv``, ``results/data_sha256.txt``,
``results/summary.md`` and the five figures under ``figures/``.
Only numerical error statistics are produced; no runtime or memory
measurements are reported anywhere (machine-independent study).
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

if __package__ in (None, ""):  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fixed_fft import budget, inputs as fx_inputs
else:
    from . import budget
    from . import inputs as fx_inputs

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (backend must be set first)

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "ADC数据采集" / "示例adc数据和结果"
DEFAULT_SEED = 20261003
N_FFT = 1024

# Validated categorical palette (fixed slot order; never cycled or reordered).
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d9d9d4"
SURFACE = "#fcfcfb"
PLOT_FLOOR = 1e-13  # display floor: exact-zero errors are plotted here

SYNTHETIC_ORDER = list(fx_inputs.SYNTHETIC_INPUTS)
WATERFALL_INPUTS = ("fullscale_multitone", "integer_bin_tone", "dc", "bandlimited_noise")


# ---------------------------------------------------------------------------
# small io helpers
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: Sequence[Dict]) -> None:
    if not rows:
        raise ValueError(f"no rows to write for {path}")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _finish(fig, path: Path) -> None:
    fig.savefig(path, dpi=150, facecolor=SURFACE, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"    wrote {path.name}")


def _series_by_param(rows: Sequence[Dict], param: str) -> Dict[str, List]:
    """input -> sorted list of (param, rel_l2_power) rows."""
    out: Dict[str, List] = {}
    for r in rows:
        out.setdefault(r["input"], []).append((r[param], r["rel_l2_power"]))
    for name in out:
        out[name].sort(key=lambda t: t[0])
    return out


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

def plot_fig1(rows_b: Sequence[Dict], path: Path) -> None:
    """Error vs internal wordlength B, one facet per synthetic input."""
    by_input = _series_by_param(rows_b, "B")
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, sharey=True)
    for ax, name in zip(axes.ravel(), SYNTHETIC_ORDER):
        pts = by_input[name]
        xs = [p[0] for p in pts]
        ys = [max(p[1], PLOT_FLOOR) for p in pts]
        ax.semilogy(xs, ys, color=PALETTE[0], lw=2.0, marker="o", ms=4.5, mfc=SURFACE, mec=PALETTE[0])
        ax.axhline(0.01, color=INK_2, lw=1.0, ls="--")
        ax.set_title(name, color=INK, fontsize=10)
        ax.set_ylim(PLOT_FLOOR / 3, 3.0)
        _style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Internal wordlength B (bit)")
    fig.supylabel("Relative power-spectrum L2 error (dimensionless)", color=INK)
    fig.suptitle(
        f"Fixed-point FFT error vs internal wordlength (N={N_FFT}, Q=16, no scaling)",
        color=INK, fontsize=12,
    )
    fig.text(
        0.5, 0.005,
        "dashed line: 1 % error; curves flat at 1e-13 are exact-zero errors shown at the display floor",
        ha="center", color=INK_2, fontsize=8,
    )
    fig.tight_layout(rect=(0.012, 0.015, 1, 0.97))
    _finish(fig, path)


def plot_fig2(rows_q: Sequence[Dict], path: Path) -> None:
    """Error vs twiddle fraction bits Q, one line per input."""
    b_fixed = rows_q[0]["B"]
    by_input = _series_by_param(rows_q, "Q")
    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    for slot, name in enumerate(SYNTHETIC_ORDER):
        pts = by_input[name]
        ax.semilogy(
            [p[0] for p in pts], [max(p[1], PLOT_FLOOR) for p in pts],
            color=PALETTE[slot], lw=2.0, marker="o", ms=4.5, mfc=SURFACE, mec=PALETTE[slot],
            label=name,
        )
    ax.axhline(0.01, color=INK_2, lw=1.0, ls="--")
    ax.set_xlabel("Twiddle coefficient fraction bits Q (bit)")
    ax.set_ylabel("Relative power-spectrum L2 error (dimensionless)")
    ax.set_ylim(PLOT_FLOOR / 3, 3.0)
    _style_axes(ax)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK)
    ax.set_title(
        f"Fixed-point FFT error vs twiddle precision (N={N_FFT}, B={b_fixed}, no scaling)",
        color=INK, fontsize=11,
    )
    fig.text(
        0.5, 0.005,
        "dashed line: 1 % error; curves flat at 1e-13 are exact-zero errors shown at the display floor",
        ha="center", color=INK_2, fontsize=8,
    )
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    _finish(fig, path)


def plot_fig3(rows_sc: Sequence[Dict], path: Path) -> None:
    """Scaling plan comparison: none vs 1-bit-per-stage, one facet per input."""
    by_input: Dict[str, Dict[str, List]] = {n: {"none": [], "scale-1bit-each-stage": []} for n in SYNTHETIC_ORDER}
    for r in rows_sc:
        by_input[r["input"]][r["scaling"]].append((r["B"], r["rel_l2_power"]))
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.2), sharex=True, sharey=True)
    handles = None
    for ax, name in zip(axes.ravel(), SYNTHETIC_ORDER):
        for scaling, color, ls in (
            ("none", PALETTE[0], "-"),
            ("scale-1bit-each-stage", PALETTE[1], "--"),
        ):
            pts = sorted(by_input[name][scaling])
            line, = ax.semilogy(
                [p[0] for p in pts], [max(p[1], PLOT_FLOOR) for p in pts],
                color=color, ls=ls, lw=2.0, marker="o", ms=4, mfc=SURFACE, mec=color,
                label="no scaling" if scaling == "none" else "scale 1 bit each stage",
            )
            if handles is None:
                handles = [line]
            elif scaling == "scale-1bit-each-stage" and len(handles) == 1:
                handles.append(line)
        ax.axhline(0.01, color=INK_2, lw=1.0, ls="--")
        ax.set_title(name, color=INK, fontsize=10)
        ax.set_ylim(PLOT_FLOOR / 3, 3.0)
        _style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel("Internal wordlength B (bit)")
    fig.supylabel("Relative power-spectrum L2 error (dimensionless)", color=INK)
    fig.legend(handles, ["no scaling", "scale 1 bit each stage"], loc="upper right",
               ncol=2, frameon=False, fontsize=10, labelcolor=INK,
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle(f"Scaling plan comparison (N={N_FFT}, Q=16)", color=INK, fontsize=12)
    fig.text(
        0.5, 0.005,
        "dashed line: 1 % error; curves flat at 1e-13 are exact-zero errors shown at the display floor",
        ha="center", color=INK_2, fontsize=8,
    )
    fig.tight_layout(rect=(0.012, 0.015, 1, 0.96))
    _finish(fig, path)


def plot_fig4(rows_wf: Sequence[Dict], path: Path) -> None:
    """Per-stage growth (in bits, unsaturated reference) and B=16 overflow."""
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.6, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]}
    )
    b_ovf = rows_wf[0]["B_overflow"] if rows_wf else 16
    markers = ("o", "s", "^", "D")
    for slot, name in enumerate(WATERFALL_INPUTS):
        pts = [(r["stage"], r["peak_lsb"], r["overflow_at_B_overflow"])
               for r in rows_wf if r["input"] == name]
        pts.sort()
        stages = [p[0] for p in pts]
        growth = [max(p[1], 1) for p in pts]
        overs = [p[2] for p in pts]
        color = PALETTE[slot]
        ax_top.plot(stages, growth, color=color, lw=2.0, marker=markers[slot], ms=5,
                    mfc=SURFACE, mec=color, label=name)
        ax_bot.plot(stages, overs, color=color, lw=2.0, marker=markers[slot], ms=5,
                    mfc=SURFACE, mec=color, label=name)
    ax_top.set_yscale("log", base=2)
    rail = 1 << (b_ovf - 1)                      # max magnitude a B-bit word stores
    ax_top.axhline(rail, color=INK_2, lw=1.2, ls="--")
    ax_top.text(
        0.985, rail * 0.90, f"B = {b_ovf} saturation rail",
        ha="right", va="top", color=INK_2, fontsize=8,
        transform=ax_top.get_yaxis_transform(),
    )
    ax_top.set_ylabel("Stage peak magnitude (LSB, base-2 scale)")
    ax_bot.set_ylabel(f"Overflow count at B = {b_ovf} (samples)")
    ax_bot.set_xlabel("FFT stage index (1..10)")
    ax_bot.set_ylim(bottom=0)
    ax_top.set_xticks(range(1, 11))
    ax_bot.set_xticks(range(1, 11))
    ax_bot.yaxis.get_major_locator().set_params(integer=True)
    for ax in (ax_top, ax_bot):
        _style_axes(ax)
    ax_top.legend(frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(
        f"Per-stage datapath growth (unsaturated B=32 reference) and B=16 overflow "
        f"(N={N_FFT}, Q=16, no scaling)",
        color=INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.945))
    _finish(fig, path)


def plot_fig5(rows_real: Sequence[Dict], path: Path) -> None:
    """Official ADC data range FFT: fixed-point vs float64 error across B."""
    rows = sorted(rows_real, key=lambda r: r["B"])
    bs = [r["B"] for r in rows]
    mean_err = [max(r["mean_rel_l2_power"], PLOT_FLOOR) for r in rows]
    max_err = [max(r["max_rel_l2_power"], PLOT_FLOOR) for r in rows]
    match = [r["peak_bin_match_fraction"] for r in rows]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(8.6, 7.0), sharex=True, gridspec_kw={"height_ratios": [1.6, 1.0]}
    )
    ax_top.semilogy(bs, mean_err, color=PALETTE[0], lw=2.0, marker="o", ms=4.5,
                    mfc=SURFACE, mec=PALETTE[0], label="mean over chirps")
    ax_top.semilogy(bs, max_err, color=PALETTE[1], lw=2.0, marker="s", ms=4.5,
                    mfc=SURFACE, mec=PALETTE[1], label="worst chirp")
    ax_top.axhline(0.01, color=INK_2, lw=1.0, ls="--")
    ax_top.set_ylabel("Relative power-spectrum L2 error (dimensionless)")
    ax_top.set_ylim(PLOT_FLOOR / 3, 3.0)
    _style_axes(ax_top)
    ax_top.legend(frameon=False, fontsize=9, labelcolor=INK)

    ax_bot.plot(bs, match, color=PALETTE[0], lw=2.0, marker="o", ms=4.5,
                mfc=SURFACE, mec=PALETTE[0])
    ax_bot.set_ylabel("Peak range-bin match fraction (dimensionless)")
    ax_bot.set_xlabel("Internal wordlength B (bit)")
    ax_bot.set_ylim(0.0, 1.05)
    _style_axes(ax_bot)

    fig.suptitle(
        "Official ADC data range FFT, fixed-point vs float64\n"
        "(far-wave f0 profile, RX0; init0: N=1024, hann(1020), Q=16, no scaling)",
        color=INK, fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    _finish(fig, path)


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return f"{x:.3e}"


def render_summary(
    rows_b: Sequence[Dict],
    rows_q: Sequence[Dict],
    rows_sc: Sequence[Dict],
    rows_wf: Sequence[Dict],
    rows_real: Sequence[Dict],
    seed: int,
    adc_name: str,
) -> str:
    min_b_1pct = budget.min_b_below(rows_b, 0.01, scaling="none", Q=16)
    min_q_1pct = budget.min_q_below(rows_q, 0.01)
    b_q_sweep = rows_q[0]["B"] if rows_q else 28
    ovf_boundary = budget.overflow_boundary(rows_sc, "none")
    scale_rows = [r for r in rows_sc if r["scaling"] == "scale-1bit-each-stage"]
    scale_ovf_max_b = max(
        (r["B"] for r in scale_rows if r["total_overflow"] > 0), default=None
    )
    real_1pct = next(
        (r["B"] for r in sorted(rows_real, key=lambda r: r["B"])
         if r["mean_rel_l2_power"] < 0.01),
        None,
    )
    real_best = min(rows_real, key=lambda r: r["B"])
    real_best32 = next(r for r in rows_real if r["B"] == 32)

    lines: List[str] = []
    lines.append("# a100-fixedpoint-fft-budget 结果摘要\n")
    lines.append(f"- 复现命令：`python src/fixed_fft/cli.py --seed {seed}`")
    lines.append(f"- 真实数据文件：`{adc_name}`（sha256 见 `data_sha256.txt`）")
    lines.append(f"- 仿真规模：合成输入 N={N_FFT}；官方链路 init0（rng_nfft=1024，nchirp=256，adc_sample_num=1020）\n")

    lines.append("## 一、达到浮点参考 1% 以内所需的最低位宽\n")
    lines.append("| 条件 | 最低参数 |")
    lines.append("| --- | --- |")
    lines.append(
        f"| 全部合成输入 rel L2 < 1%（Q=16，不缩放） | **B = {min_b_1pct}** |"
    )
    lines.append(
        f"| 全部合成输入 rel L2 < 1%（B={b_q_sweep}，不缩放，高于不缩放溢出边界） | **Q = {min_q_1pct}** |"
    )
    lines.append(
        f"| 官方数据逐 chirp 平均 rel L2 < 1%（Q=16，不缩放） | **B = {real_1pct}** |"
    )
    lines.append("")

    lines.append("## 二、溢出发生边界（saturate 处理，Q=16）\n")
    lines.append("| 输入 | 不缩放时仍有溢出的最大 B | B=32 时误差 |")
    lines.append("| --- | --- | --- |")
    for name in SYNTHETIC_ORDER:
        r32 = next(r for r in rows_b if r["input"] == name and r["B"] == 32)
        b_ovf = ovf_boundary.get(name)
        b_txt = f"B ≤ {b_ovf}" if b_ovf is not None else "全范围无溢出"
        lines.append(f"| {name} | {b_txt} | {_fmt(r32['rel_l2_power'])} |")
    if scale_ovf_max_b is None:
        lines.append("\n- 每级右移 1 bit（scale-1bit-each-stage）：B=8..32 **全范围无溢出**。")
    else:
        lines.append(
            f"\n- 每级右移 1 bit（scale-1bit-each-stage）：仅 B ≤ {scale_ovf_max_b} 出现溢出，"
            f"更大 B 全程无溢出。"
        )
    lines.append("")

    lines.append("## 三、缩放计划对比（Q=16）\n")
    lines.append("| 输入 | B | 不缩放误差 | 每级缩放误差 | 不缩放溢出 | 缩放溢出 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in SYNTHETIC_ORDER:
        for B in (16, 20, 24):
            rn = next(r for r in rows_sc if r["input"] == name and r["B"] == B and r["scaling"] == "none")
            rs = next(r for r in rows_sc if r["input"] == name and r["B"] == B and r["scaling"] == "scale-1bit-each-stage")
            lines.append(
                f"| {name} | {B} | {_fmt(rn['rel_l2_power'])} | {_fmt(rs['rel_l2_power'])} "
                f"| {rn['total_overflow']} | {rs['total_overflow']} |"
            )
    lines.append("")

    lines.append("## 四、真实数据（官方 ADC，init0，单 RX）\n")
    lines.append(f"- B={real_best['B']} 时平均误差 {_fmt(real_best['mean_rel_l2_power'])}，"
                 f"峰值 bin 命中率 {real_best['peak_bin_match_fraction']:.3f}。")
    lines.append(f"- B=32 时平均误差 {_fmt(real_best32['mean_rel_l2_power'])}，"
                 f"峰值 bin 命中率 {real_best32['peak_bin_match_fraction']:.3f}，"
                 f"全程溢出 {real_best32['total_overflow_all_chirps']} 次。")
    lines.append(f"- 平均误差 1% 以内所需最低 B = **{real_1pct}**（Q=16，不缩放）。")
    lines.append(
        "- B≥17 后平均误差稳定在约 3.7e-3 的地板：该数据加窗后谱峰仅占满量程单音的约 0.3%，"
        "误差由整数 LSB 数据通路的逐级截断绝对噪声（约百 LSB 量级）主导，与 Q 基本无关——"
        "这正是定点量程利用率影响精度底噪的体现。\n"
    )

    lines.append("## 五、建议\n")
    if min_b_1pct is not None and min_q_1pct is not None:
        lines.append(
            f"- 预算下限：**B ≥ {max(min_b_1pct, 16)}、Q ≥ {min_q_1pct}** 可让全部合成压力输入的"
            f"功率谱相对误差进入 1% 以内（B 同时需 ≥16 才能无损承载 int16 输入）。"
        )
    lines.append(
        "- 缩放建议：位宽受限（B ≤ 溢出边界）时采用每级右移 1 bit 的缩放计划以消除饱和失真；"
        "位宽充足（B ≥ 24）且输入留有余量时可不缩放，省去增益补偿。"
    )
    lines.append(
        "- 以上均为离线数值仿真结论，仿真算术格式不代表 A100 芯片内部实现，未做任何速度/内存基准。"
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a100-fixedpoint-fft-budget",
        description="Fixed-point FFT wordlength and scaling budget study (offline, machine independent).",
    )
    here = Path(__file__).resolve()
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help=f" RNG seed for the synthetic stress inputs (default {DEFAULT_SEED})")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_DIR,
                        help="directory holding the eight official ADC txt dumps")
    parser.add_argument("--outdir", type=Path, default=PROJECT_DIR,
                        help="project directory to write results/ and figures/ under")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    results_dir = args.outdir / "results"
    figures_dir = args.outdir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_root)

    print(f"[1/7] synthetic stress inputs (seed={args.seed}, N={N_FFT})")
    signals = fx_inputs.make_synthetic_inputs(N_FFT, args.seed)

    print("[2/7] experiment 1: B sweep, Q=16, no scaling")
    rows_b = budget.sweep_B(signals, budget.B_LIST, Q=16, scaling="none")
    write_csv(results_dir / "sweep_B.csv", rows_b)

    print("[3/7] experiment 2: Q sweep, B=28 (above the no-scaling overflow boundary), no scaling")
    rows_q = budget.sweep_Q(signals, budget.Q_LIST, B=28, scaling="none")
    write_csv(results_dir / "sweep_Q.csv", rows_q)

    print("[4/7] experiment 3: scaling plan comparison, Q=16")
    rows_sc = budget.scaling_compare(signals, budget.B_LIST, Q=16)
    write_csv(results_dir / "scaling_compare.csv", rows_sc)

    print("[5/7] experiment 4: per-stage growth waterfall (B=32 reference) and overflow (B=16), Q=16")
    wf_signals = {name: signals[name] for name in WATERFALL_INPUTS}
    rows_wf = budget.stage_waterfall(wf_signals, B_peak=32, B_overflow=16, Q=16, scaling="none")
    write_csv(results_dir / "stage_waterfall.csv", rows_wf)

    adc_name = fx_inputs.DATA_FILES[0]
    adc_path = data_dir / adc_name
    if not adc_path.is_file():
        print(f"ERROR: official ADC dump not found: {adc_path}", file=sys.stderr)
        return 2
    print(f"[6/7] experiment 5: official ADC range FFT ({adc_name}, init0)")
    rows_real = budget.real_data_error(adc_path, budget.B_LIST, Q=16, scaling="none")
    write_csv(results_dir / "real_data_error.csv", rows_real)

    print("[6b/7] data integrity (sha256 of the 8 official dumps)")
    sha_lines = budget.sha256_of_files(data_dir / name for name in fx_inputs.DATA_FILES)
    (results_dir / "data_sha256.txt").write_text("\n".join(sha_lines) + "\n", encoding="utf-8")

    print("[7/7] figures")
    plot_fig1(rows_b, figures_dir / "fig1_error_vs_B.png")
    plot_fig2(rows_q, figures_dir / "fig2_error_vs_Q.png")
    plot_fig3(rows_sc, figures_dir / "fig3_scaling_compare.png")
    plot_fig4(rows_wf, figures_dir / "fig4_stage_waterfall.png")
    plot_fig5(rows_real, figures_dir / "fig5_real_data_range_fft.png")

    summary = render_summary(rows_b, rows_q, rows_sc, rows_wf, rows_real, args.seed, adc_name)
    (results_dir / "summary.md").write_text(summary, encoding="utf-8")
    print("    wrote summary.md")

    min_b = budget.min_b_below(rows_b, 0.01, scaling="none", Q=16)
    min_q = budget.min_q_below(rows_q, 0.01)
    real_1pct = next((r["B"] for r in sorted(rows_real, key=lambda r: r["B"])
                      if r["mean_rel_l2_power"] < 0.01), None)
    print("done.")
    print(f"  min B for <1% on all synthetic inputs (Q=16, no scaling): {min_b}")
    print(f"  min Q for <1% on all synthetic inputs (B={rows_q[0]['B']}, no scaling): {min_q}")
    print(f"  min B for <1% mean error on official data  (Q=16, no scaling): {real_1pct}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
