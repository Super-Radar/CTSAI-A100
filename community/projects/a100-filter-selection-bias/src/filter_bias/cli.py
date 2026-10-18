"""Command-line entry point: reproduce every result artifact offline.

Runs the four core experiments on the pooled 22-file RawTarget corpus and
writes ``results/`` (CSV tables, sha256 provenance, Chinese summary) and
``figures/`` (5 English-labelled PNGs). Fully deterministic apart from the
figure-5 scatter subsample, which is seeded (default seed 20261003).
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless: never require a display
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

# Support both ``python -m filter_bias.cli`` (PYTHONPATH=src) and direct
# script execution ``python src/filter_bias/cli.py``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filter_bias.filters import DIMENSION_META, FilterSpec, single_dim_spec
from filter_bias.parser import parse_files
from filter_bias.sweep import (
    FIELDS,
    QUANTILE_LEVELS,
    deletion_mask,
    distribution_shift,
    joint_grid,
    pairwise_jaccard,
    retention,
    stratified_retention,
    sweep_dimension,
)

DEFAULT_SEED = 20261003

# --- experiment grids -------------------------------------------------------
V_MIN_SWEEP = [round(0.1 * i, 2) for i in range(21)]  # 0.0 .. 2.0 m/s
SPEED_MAX_SWEEP = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 73]
RANGE_MAX_SWEEP = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 267]
ANGLE_MAX_SWEEP = [5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 75]
SNR_MIN_SWEEP = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]

# Reference setting for the deletion-intersection experiment: a plausible
# mid-strength value on every dimension at once, deletions measured per
# single dimension at its reference value.
REFERENCE = {
    "range_max": 150.0,
    "angle_max": 60.0,
    "speed_min": 0.5,
    "speed_max": 30.0,
    "snr_min": 10.0,
}

JOINT_VMIN = V_MIN_SWEEP
JOINT_SMIN = list(range(0, 32, 2))  # 0 .. 30 dB, step 2
JOINT_RMAX = [50, 75, 100, 125, 150, 175, 200, 250]
JOINT_SMIN2 = [0, 5, 10, 15, 20, 25, 30]

DIST_SETTINGS = (
    [("speed_min", v) for v in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0)]
    + [("snr_min", s) for s in (0, 5, 10, 15, 20, 25)]
    + [("range_max", r) for r in (50, 100, 150, 200, 267)]
)

# --- dataviz skill reference palette (light mode) ---------------------------
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID_C, AXIS_C = "#e1e0d9", "#c3c2b7"
BLUE, ORANGE, AQUA, RED = "#2a78d6", "#eb6834", "#1baf7a", "#e34948"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#cde2fb", "#3987e5", "#0d366b"])
DIV_BLUE_RED = LinearSegmentedColormap.from_list("div_blue_red", ["#1c5cab", "#f0efec", "#d03b3b"])

MAIN_DATA_DIR = "Python雷达可视化工具示例"
MAIN_DATA_NAME = "负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv"
EXT_DATA_GLOB = ("RadarTools", "RadarTools_Release", "data")


def project_root() -> Path:
    # <proj>/src/filter_bias/cli.py -> <proj>
    return Path(__file__).resolve().parents[2]


def discover_inputs(repo_root: Path) -> list:
    main_csv = repo_root / MAIN_DATA_DIR / MAIN_DATA_NAME
    if not main_csv.exists():
        cands = sorted((repo_root / MAIN_DATA_DIR).glob("*RawTarget.csv"))
        if not cands:
            raise SystemExit(f"no RawTarget csv found under {repo_root / MAIN_DATA_DIR}")
        main_csv = cands[0]
    ext = sorted((repo_root.joinpath(*EXT_DATA_GLOB)).glob("*_RawTarget.csv"))
    return [main_csv] + ext


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list, fieldnames: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_provenance(path: Path, files: list, repo_root: Path) -> None:
    lines = []
    for f in files:
        rel = f.relative_to(repo_root).as_posix()
        lines.append(f"{sha256_file(f)}  {rel}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 130,
            "savefig.dpi": 130,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Noto Sans", "DejaVu Sans"],
            "font.size": 8.5,
            "text.color": INK,
            "axes.edgecolor": AXIS_C,
            "axes.labelcolor": INK2,
            "axes.titlesize": 9,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": GRID_C,
            "grid.linewidth": 0.6,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "figure.constrained_layout.use": True,
        }
    )


def _sweep_arrays(cloud, rows_all, dimension):
    sel = [r for r in rows_all if r["dimension"] == dimension and r["stratum"] == "ALL"]
    sel.sort(key=lambda r: r["threshold"])
    return (
        np.array([r["threshold"] for r in sel]),
        np.array([r["retention"] for r in sel]),
    )


def fig1_retention_curves(cloud, rows_all, out: Path) -> None:
    dims = ["speed_min", "speed_max", "range_max", "angle_max", "snr_min"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.0))
    for ax, dim in zip(axes.ravel(), dims):
        x, y = _sweep_arrays(cloud, rows_all, dim)
        meta = DIMENSION_META[dim]
        ax.plot(x, y, color=BLUE, lw=1.8, marker="o", ms=3.2, mfc=BLUE, mec=SURFACE)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel(f"{meta.label} threshold ({meta.unit})")
        ax.set_ylabel("Retained fraction")
        ax.set_title(f"{dim}: {meta.description}")
    axes.ravel()[5].axis("off")
    axes.ravel()[5].text(
        0.02, 0.6,
        f"pooled corpus: {len(cloud):,} targets\n22 RawTarget files (CTLRR-300-V2_1)\n"
        "one dimension active per curve",
        va="center", ha="left", fontsize=9, color=INK2,
    )
    fig.suptitle("Fig 1 - Retention curves of single-dimension threshold filters", fontsize=11)
    fig.savefig(out)
    plt.close(fig)


def fig2_jaccard(jrows: list, out: Path) -> None:
    dims = list(REFERENCE)
    n = len(dims)
    mat = np.zeros((n, n))
    np.fill_diagonal(mat, 1.0)
    for r in jrows:
        i, j = dims.index(r["dim_a"]), dims.index(r["dim_b"])
        mat[i, j] = mat[j, i] = r["jaccard"]
    fig, ax = plt.subplots(figsize=(5.6, 5.0))
    im = ax.imshow(mat, cmap=SEQ_BLUE, vmin=0, vmax=1)
    ax.set_xticks(range(n), dims, rotation=30, ha="right")
    ax.set_yticks(range(n), dims)
    ax.grid(False)
    for i in range(n):
        for j in range(n):
            ax.text(
                j, i, f"{mat[i, j]:.2f}", ha="center", va="center",
                fontsize=8, color=(SURFACE if mat[i, j] > 0.6 else INK),
            )
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Jaccard similarity of deletion sets", color=INK2)
    ax.set_title(
        "Fig 2 - Pairwise overlap of per-dimension deletions\n"
        f"(reference: R<=150 m, |a|<=60 deg, |v|>=0.5, |v|<=30 m/s, snr>=10 dB)",
        fontsize=9,
    )
    fig.savefig(out)
    plt.close(fig)


def _heatmap(ax, xv, yv, z, cmap, norm, xlab, ylab, title):
    mesh = ax.pcolormesh(xv, yv, z, cmap=cmap, norm=norm, shading="auto")
    ax.grid(False)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=9)
    return mesh


def fig3_joint_heatmaps(g1: list, g2: list, out: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    s = np.array(sorted({r["value_b"] for r in g1}))
    v = np.array(sorted({r["value_a"] for r in g1}))
    z1 = np.full((v.size, s.size), np.nan)
    for r in g1:
        z1[np.searchsorted(v, r["value_a"]), np.searchsorted(s, r["value_b"])] = r["retention"]
    m1 = _heatmap(
        axes[0], s, v, z1, SEQ_BLUE, None,
        "snr threshold s_min (dB)", "|speed| threshold v_min (m/s)",
        "(a) joint retention, v_min x s_min",
    )
    fig.colorbar(m1, ax=axes[0], shrink=0.9).set_label("Retained fraction", color=INK2)

    s2 = np.array(sorted({r["value_b"] for r in g2}))
    r2 = np.array(sorted({r["value_a"] for r in g2}))
    z2 = np.full((r2.size, s2.size), np.nan)
    for r in g2:
        z2[np.searchsorted(r2, r["value_a"]), np.searchsorted(s2, r["value_b"])] = r["interaction"]
    amp = max(1e-6, float(np.nanmax(np.abs(z2))))
    m2 = _heatmap(
        axes[1], s2, r2, z2, DIV_BLUE_RED, TwoSlopeNorm(vcenter=0.0, vmin=-amp, vmax=amp),
        "snr threshold s_min (dB)", "range threshold R_max (m)",
        "(b) interaction = retention - independence product",
    )
    fig.colorbar(m2, ax=axes[1], shrink=0.9).set_label("Retention minus product", color=INK2)
    fig.suptitle("Fig 3 - Joint threshold grids (blue < 0: shared deletions / double hit)", fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def fig4_distribution_shift(cloud, out: Path) -> None:
    specs = {
        "unfiltered": FilterSpec(),
        "v_min = 0.5 m/s": FilterSpec(speed_min=0.5),
        "s_min = 15 dB": FilterSpec(snr_min=15),
    }
    styles = {
        "unfiltered": dict(color=INK, lw=1.8, ls="-"),
        "v_min = 0.5 m/s": dict(color=BLUE, lw=1.8, ls="--"),
        "s_min = 15 dB": dict(color=ORANGE, lw=1.8, ls="-."),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.9), sharey=False)
    levels_pct = [100 * q for q in QUANTILE_LEVELS]
    for ax, attr in zip(axes, ("range_m", "angle_deg", "snr_db")):
        name, unit = FIELDS[attr]
        x = getattr(cloud, attr)
        for label, spec in specs.items():
            qs = np.quantile(x[spec.kept_mask(cloud)], QUANTILE_LEVELS)
            ax.plot(levels_pct, qs, label=label, **styles[label])
        ax.set_xlabel("Quantile level (%)")
        ax.set_ylabel(f"{name} ({unit})")
        ax.set_title(f"{name} quantile curve")
    axes[0].legend(loc="upper left")
    fig.suptitle("Fig 4 - Distribution quantiles before vs. after threshold filtering", fontsize=10)
    fig.savefig(out)
    plt.close(fig)


def fig5_nearzero_spatial(cloud, out: Path, seed: int, v_min: float = 0.5) -> tuple:
    rng = np.random.default_rng(seed)
    n = len(cloud)
    take = min(40000, n)
    idx = rng.choice(n, size=take, replace=False)
    sub_range = cloud.range_m[idx]
    sub_angle = np.deg2rad(cloud.angle_deg[idx])
    removed = np.abs(cloud.speed_ms[idx]) < v_min
    fig = plt.figure(figsize=(6.8, 6.2))
    ax = fig.add_subplot(projection="polar")
    ax.grid(color=GRID_C, lw=0.6)
    ax.scatter(sub_angle[~removed], sub_range[~removed], s=1.2, c=BLUE, alpha=0.25, lw=0)
    ax.scatter(sub_angle[removed], sub_range[removed], s=1.4, c=RED, alpha=0.35, lw=0)
    frac = float(np.count_nonzero(removed)) / take
    ax.set_rlim(0, float(np.quantile(sub_range, 0.999)))
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_xlabel("")
    fig.suptitle(
        f"Fig 5 - Near-zero speed removal footprint (v_min = {v_min} m/s)\n"
        f"red: |speed| < {v_min} m/s removed ({100 * frac:.1f}% of subsample), "
        "blue: retained; theta = azimuth (deg), r = range (m)",
        fontsize=9,
    )
    fig.savefig(out)
    plt.close(fig)
    return frac


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def build_summary(cloud, stats, rows_all, jrows, g1, g2, nearzero_frac, results_dir) -> str:
    total = len(cloud)

    def all_rows(dim):
        sel = [r for r in rows_all if r["dimension"] == dim and r["stratum"] == "ALL"]
        sel.sort(key=lambda r: r["threshold"])
        return sel

    v_rows = all_rows("speed_min")
    r05 = next(r for r in v_rows if abs(r["threshold"] - 0.5) < 1e-9)

    band_rows = [
        r for r in rows_all
        if r["dimension"] == "speed_min" and abs(r["threshold"] - 0.5) < 1e-9
        and r["stratum"] == "range_band"
    ]
    worst_band = min(band_rows, key=lambda r: r["retention"])

    sect_rows = [
        r for r in rows_all
        if r["dimension"] == "speed_min" and abs(r["threshold"] - 0.5) < 1e-9
        and r["stratum"] == "angle_sector"
    ]
    worst_sect = min(sect_rows, key=lambda r: r["retention"])

    static_of_lowsnr = (
        float(np.count_nonzero((cloud.snr_db < 30.0) & (np.abs(cloud.speed_ms) < 0.5)))
        / np.count_nonzero(cloud.snr_db < 30.0)
        if np.any(cloud.snr_db < 30.0) else float("nan")
    )
    far_share = float(np.count_nonzero(cloud.range_m > 150)) / total

    top_j = max(jrows, key=lambda r: r["jaccard"])
    j_sorted = sorted(jrows, key=lambda r: -r["jaccard"])
    top_cont = max(jrows, key=lambda r: max(r["containment_a"], r["containment_b"]))
    if top_cont["containment_a"] >= top_cont["containment_b"]:
        cont_txt = (f"{_pct(top_cont['containment_a'])} 的 {top_cont['dim_a']} 删除集"
                    f"同时落在 {top_cont['dim_b']} 删除集内")
    else:
        cont_txt = (f"{_pct(top_cont['containment_b'])} 的 {top_cont['dim_b']} 删除集"
                    f"同时落在 {top_cont['dim_a']} 删除集内")

    g1_min = min(g1, key=lambda r: r["interaction"])
    r150 = next(r for r in g2 if r["value_a"] == 150.0 and r["value_b"] == 15.0)
    far_low = float(np.count_nonzero((cloud.range_m > 150) & (cloud.snr_db < 15))) / total

    d_v = distribution_shift(cloud, FilterSpec(speed_min=0.5))
    d_s = distribution_shift(cloud, FilterSpec(snr_min=15))

    def stat(rows, field, stat_name):
        return next(r for r in rows if r["field"] == field and r["stat"] == stat_name)

    range_mean_before = stat(d_v, "range", "mean")["before"]
    range_mean_after_v = stat(d_v, "range", "mean")["after"]
    range_mean_after_s = stat(d_s, "range", "mean")["after"]
    angle_mean_after_v = stat(d_v, "angle", "mean")["after"]

    header_variants = {}
    for s in stats:
        header_variants[s.header_columns] = header_variants.get(s.header_columns, 0) + 1
    variant_txt = "、".join(f"{k} 列 x{v} 个文件" for k, v in sorted(header_variants.items()))
    files_with_motion = int(np.unique(cloud.file_id[np.abs(cloud.speed_ms) > 0]).size)

    lines = []
    lines.append("# a100-filter-selection-bias 关键结论\n")
    lines.append(f"- 生成：`cli.py`（固定种子 {DEFAULT_SEED}）；数据：22 个 RawTarget 文件（CTLRR-300-V2_1），"
                 f"共 {total:,} 条目标行、{sum(s.frames for s in stats):,} 帧；表头列数变体：{variant_txt}；"
                 f"{files_with_motion}/22 个文件含非零 speed。\n")
    lines.append("## 实验① 近零速度过滤的整体隐藏程度\n")
    lines.append(f"- v_min=0.5 m/s 时总体删除 {_pct(1 - r05['retention'])}"
                 f"（保留 {_pct(r05['retention'])}，{r05['n_kept']:,}/{r05['n_total']:,}）；"
                 f"语料中 |speed|<0.5 的观测占比 {_pct(nearzero_frac)}（fig5 子采样口径）。")
    lines.append(f"- 删除并非空间均匀：受创最重的距离带为 {worst_band['band']} m（保留 {_pct(worst_band['retention'])}），"
                 f"受创最重的角度扇区为 |angle| {worst_sect['band']} deg（保留 {_pct(worst_sect['retention'])}）。")
    lines.append(f"- v_min 从 0 扫到 2 m/s，保留率从 100% 降至 {_pct(all_rows('speed_min')[-1]['retention'])}"
                 "（retention_sweep.csv 分层行量化各带损失）。\n")
    lines.append("## 实验② 各筛选维度删除集合的交集\n")
    lines.append(f"- 参考设置（R≤150 m、|angle|≤60 deg、|v|≥0.5、|v|≤30 m/s、snr≥10 dB）下逐维取删除集：")
    for r in j_sorted[:3]:
        lines.append(f"  - {r['dim_a']} x {r['dim_b']}：Jaccard={r['jaccard']:.3f}"
                     f"（交集 {r['intersection']:,} / 并集 {r['union']:,}）")
    lines.append(f"- 两两 Jaccard 全部 ≤ {top_j['jaccard']:.3f}：五个维度的删除集合**基本互不相交**，"
                 "多阈值叠加时的重复计数损失有限。")
    lines.append(f"- 但包含率（交集 / 较小删除集）揭示了单向覆盖：{cont_txt}——"
                 "即近零速度剔除几乎吸收了另一维的全部删除，叠加它对总删除量的边际贡献很小。\n")
    lines.append("## 实验③ snr_min 与 range_max 的联合作用（远距离低 SNR 双重打击）\n")
    lines.append(f"- (R_max=150 m, s_min=15 dB) 联合保留 {_pct(r150['retention'])}，"
                 f"若两维删除独立应为 {_pct(r150['retention_indep'])}，"
                 f"交互量 {r150['interaction']:+.4f}（负=删除共享，双重打击）。")
    lines.append(f"- 本语料中该“双重打击”量级很小：range>150 m 的观测仅 {_pct(far_share)}，"
                 f"其中 snr<15 dB 的仅 {_pct(far_low)}——远距低 SNR 点几乎不存在，两维删除难有交集。")
    lines.append(f"- 交互最强的组合是 v_min x s_min 的高 SNR 端：最负交互在 (v={g1_min['value_a']}, s={g1_min['value_b']})，"
                 f"交互 {g1_min['interaction']:+.3f}；snr<30 dB 的观测中 {_pct(static_of_lowsnr)} 同时 |speed|<0.5 m/s"
                 "（低 SNR 观测多为静态杂波，两个阈值在删同一批点，joint_grid.csv、fig3）。\n")
    lines.append("## 实验④ 筛选前后分布偏移\n")

    def _drift(delta, tol=0.05):
        if delta < -tol:
            return "分布向小值收缩"
        if delta > tol:
            return "分布向大值偏移"
        return "分布几乎不变"

    dr_v = range_mean_after_v - range_mean_before
    dr_s = range_mean_after_s - range_mean_before
    lines.append(f"- v_min=0.5：range 均值 {range_mean_before:.1f} -> {range_mean_after_v:.1f} m"
                 f"（Δ{dr_v:+.1f} m，{_drift(dr_v)}），angle 均值 -> {angle_mean_after_v:.2f} deg"
                 "（近零删除集中于特定距离带/方位，见 fig5）。")
    lines.append(f"- s_min=15：range 均值 {range_mean_before:.1f} -> {range_mean_after_s:.1f} m"
                 f"（Δ{dr_s:+.1f} m，{_drift(dr_s)}；分位数曲线见 fig4、distribution_shift.csv）。\n")
    lines.append("## 局限\n")
    lines.append("- 无真值标注：本报告只评价保留率与分布变化，不给出准确率/召回率，不评选“最佳检测阈值”。")
    lines.append("- 数据为 RadarTools 示例目标日志（CTLRR-300-V2_1 标注，非 A100 硬件实测）；全部为离线分析。")
    lines.append("- 结果可由 `python -m filter_bias.cli`（或 `python src/filter_bias/cli.py`）确定性重跑。\n")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="filter-bias",
        description="Threshold selection-bias study on CTSAI-A100 RawTarget logs (offline).",
    )
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="CTSAI-A100 checkout root (default: auto-detect relative to this file)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"RNG seed for the fig5 scatter subsample (default {DEFAULT_SEED})")
    ap.add_argument("--skip-figures", action="store_true", help="only write results/, no figures/")
    args = ap.parse_args(argv)

    root = project_root()
    repo_root = args.repo_root if args.repo_root is not None else root.parents[2]
    results_dir = root / "results"
    figures_dir = root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    files = discover_inputs(repo_root)
    print(f"[data] {len(files)} RawTarget files under {repo_root}")
    cloud, stats = parse_files(files)
    total = len(cloud)
    print(f"[data] pooled {total:,} target rows, {sum(s.frames for s in stats):,} frames")
    if total == 0:
        print("ERROR: empty corpus", file=sys.stderr)
        return 2
    write_provenance(results_dir / "data_sha256.txt", files, repo_root)

    # Experiment 1: single-dimension sweeps (near-zero experiment stratified).
    rows_all = []
    rows_all += sweep_dimension(cloud, "speed_min", V_MIN_SWEEP,
                                strata=("range_band", "angle_sector"))
    rows_all += sweep_dimension(cloud, "speed_max", SPEED_MAX_SWEEP)
    rows_all += sweep_dimension(cloud, "range_max", RANGE_MAX_SWEEP)
    rows_all += sweep_dimension(cloud, "angle_max", ANGLE_MAX_SWEEP)
    rows_all += sweep_dimension(cloud, "snr_min", SNR_MIN_SWEEP, strata=("range_band",))
    write_csv(
        results_dir / "retention_sweep.csv", rows_all,
        ["dimension", "threshold", "unit", "stratum", "band", "n_total", "n_kept", "retention"],
    )
    print(f"[exp1] retention sweep rows: {len(rows_all)}")

    # Experiment 2: pairwise Jaccard of per-dimension deletion sets.
    deletions = {dim: deletion_mask(cloud, single_dim_spec(dim, val))
                 for dim, val in REFERENCE.items()}
    jrows = pairwise_jaccard(deletions)
    write_csv(
        results_dir / "jaccard.csv", jrows,
        ["dim_a", "dim_b", "n_del_a", "n_del_b", "intersection", "union", "jaccard",
         "containment_a", "containment_b"],
    )
    top = max(jrows, key=lambda r: r["jaccard"])
    print(f"[exp2] top deletion overlap: {top['dim_a']} x {top['dim_b']} J={top['jaccard']:.3f}")

    # Experiment 3: joint grids (v_min x s_min and range_max x snr_min).
    g1 = joint_grid(cloud, "speed_min", JOINT_VMIN, "snr_min", JOINT_SMIN)
    g2 = joint_grid(cloud, "range_max", JOINT_RMAX, "snr_min", JOINT_SMIN2)
    for r in g1:
        r["grid"] = "vmin_smin"
    for r in g2:
        r["grid"] = "rmax_smin"
    write_csv(
        results_dir / "joint_grid.csv", g1 + g2,
        ["grid", "dim_a", "value_a", "dim_b", "value_b", "n_kept",
         "retention", "retention_indep", "interaction"],
    )
    r150 = next(r for r in g2 if r["value_a"] == 150.0 and r["value_b"] == 15.0)
    print(f"[exp3] (R<=150m, snr>=15dB): retention={r150['retention']:.4f} "
          f"indep={r150['retention_indep']:.4f} interaction={r150['interaction']:+.4f}")

    # Experiment 4: distribution quantile shifts.
    drows = []
    for dim, val in DIST_SETTINGS:
        for r in distribution_shift(cloud, single_dim_spec(dim, val)):
            r["dimension"] = dim
            r["threshold"] = float(val)
            drows.append(r)
    write_csv(
        results_dir / "distribution_shift.csv", drows,
        ["dimension", "threshold", "field", "unit", "stat", "before", "after", "delta"],
    )
    print(f"[exp4] distribution shift rows: {len(drows)}")

    # Figures.
    if not args.skip_figures:
        _apply_style()
        fig1_retention_curves(cloud, rows_all, figures_dir / "fig1_retention_curves.png")
        fig2_jaccard(jrows, figures_dir / "fig2_jaccard_matrix.png")
        fig3_joint_heatmaps(g1, g2, figures_dir / "fig3_joint_heatmap.png")
        fig4_distribution_shift(cloud, figures_dir / "fig4_distribution_shift.png")
        nearzero_frac = fig5_nearzero_spatial(
            cloud, figures_dir / "fig5_nearzero_spatial.png", seed=args.seed)
        print(f"[figs] 5 figures written to {figures_dir} (fig5 removed {_pct(nearzero_frac)})")
    else:
        nearzero_frac = float(np.count_nonzero(np.abs(cloud.speed_ms) < 0.5)) / total

    (results_dir / "summary.md").write_text(
        build_summary(cloud, stats, rows_all, jrows, g1, g2, nearzero_frac, results_dir),
        encoding="utf-8",
    )
    print(f"[done] results in {results_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
