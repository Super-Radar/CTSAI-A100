"""Experiments of the noise-floor estimation sensitivity study.

Four experiments (all on the repository's public example captures, offline,
no hardware in the loop, no empty-scene ground truth -- conclusions are
phrased as deviations relative to each estimator's uncontaminated value):

1. ``experiment_reproduction`` -- official noiseValue for near/far captures
   per RX channel and for the official 4-channel average, with percentile
   bootstrap CIs (resampled region cells).
2. ``experiment_contamination`` -- synthetic Gaussian peaks injected into the
   official noise region of a *real* RV image (single-half vs both-halves
   injection) quantifying how much the official mean moves while
   median/MAD-based estimates barely change.
3. ``experiment_region_scan`` -- official-style mean over a grid of noise
   regions (range start x width, two velocity-band choices) on real data,
   showing the spread caused by region choice alone.
4. ``experiment_bias_curves`` -- estimator deviation vs injected peak power
   (0..40 dB) for all estimators.

Figures use English labels (matplotlib default fonts lack CJK glyphs); the
README explains every figure in Chinese.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from . import robust
from .official_noise import (
    NoiseRegionBounds,
    calculate_noise_from_rv_db,
    noise_region_bounds,
    official_region,
    region_mean,
)
from .pipeline import (
    SensorConfig,
    channel_files,
    default_data_dir,
    default_hxx_dir,
    do_fft_2d,
    load_adc_data,
    parse_sensor_config,
    power_to_db,
    rv_power,
)

# Validated categorical palette (fixed order, see dataviz reference palette).
SERIES_COLORS = {
    "official_mean": "#2a78d6",  # blue
    "median": "#eb6834",  # orange
    "q25": "#1baf7a",  # aqua
    "q10": "#eda100",  # yellow
    "trim10": "#e87ba4",  # magenta
    "trim20": "#008300",  # green
    "mad_floor": "#4a3aa7",  # violet
}
ESTIMATOR_ORDER = list(SERIES_COLORS)

KIND_LABEL = {"far": "far wave (init0, f0)", "near": "near wave (init1, f1)"}
HXX_NAME = {"far": "sensor_config_init0.hxx", "near": "sensor_config_init1.hxx"}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


@dataclass
class Dataset:
    kind: str
    cfg: SensorConfig
    files: List[Path]
    vel_fft: np.ndarray  # (rng_nfft/2, vel_nfft, 4) complex Radar_cube
    rv_db: np.ndarray  # 4-channel-averaged power image, dB
    rv_db_channels: Dict[int, np.ndarray]
    load_reports: List[object]
    bounds: NoiseRegionBounds

    @property
    def region(self) -> np.ndarray:
        """Official min-of-halves noise region values (dB)."""
        return official_region(self.rv_db, self.cfg)


def load_dataset(kind: str, data_dir: Path, hxx_dir: Path) -> Dataset:
    cfg = parse_sensor_config(Path(hxx_dir) / HXX_NAME[kind])
    files = channel_files(Path(data_dir), kind)
    adc, reports = load_adc_data(files, cfg)
    _, vel_fft = do_fft_2d(adc, cfg)
    rv_db = power_to_db(rv_power(vel_fft))
    per_ch = {ch: power_to_db(np.abs(vel_fft[:, :, ch]) ** 2) for ch in range(cfg.nvirtual_array)}
    return Dataset(
        kind=kind,
        cfg=cfg,
        files=files,
        vel_fft=vel_fft,
        rv_db=rv_db,
        rv_db_channels=per_ch,
        load_reports=reports,
        bounds=noise_region_bounds(cfg),
    )


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------


def write_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def fmt(value: object, nd: int = 3) -> str:
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


def md_table(header: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    out = ["| " + " | ".join(str(h) for h in header) + " |"]
    out.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        out.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(out)


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Experiment 1: reproduction table
# ---------------------------------------------------------------------------


def experiment_reproduction(
    datasets: Dict[str, Dataset],
    out_dir: Path,
    seed: int,
    n_boot: int = 2000,
) -> Tuple[List[List[object]], List[List[object]]]:
    """Official + robust noise estimates for every dataset/channel set."""
    header = [
        "dataset", "channel_set", "region_cells",
        *[f"{name}_dB" for name in ESTIMATOR_ORDER],
        "boot_ci95_official_mean_low_dB", "boot_ci95_official_mean_high_dB",
        "boot_ci95_median_low_dB", "boot_ci95_median_high_dB",
    ]
    rows: List[List[object]] = []
    md_rows: List[List[object]] = []
    for ds in (datasets["far"], datasets["near"]):
        channel_sets: List[Tuple[str, np.ndarray]] = [
            (f"ch{ch}", ds.rv_db_channels[ch]) for ch in range(ds.cfg.nvirtual_array)
        ] + [("4ch_avg_official", ds.rv_db)]
        for ci, (label, img) in enumerate(channel_sets):
            region = official_region(img, ds.cfg)
            values = [robust.ESTIMATORS[name](region) for name in ESTIMATOR_ORDER]
            mean_lo, mean_hi = robust.bootstrap_ci(
                region, robust.official_mean, n_boot=n_boot, seed=seed + 100 + ci
            )
            med_lo, med_hi = robust.bootstrap_ci(
                region, robust.median, n_boot=n_boot, seed=seed + 200 + ci
            )
            rows.append(
                [ds.kind, label, region.size, *values, mean_lo, mean_hi, med_lo, med_hi]
            )
            md_rows.append(
                [
                    ds.kind, label, region.size, values[0],
                    f"[{mean_lo:.2f}, {mean_hi:.2f}]",
                    values[1], f"[{med_lo:.2f}, {med_hi:.2f}]",
                ]
            )
    write_csv(out_dir / "results" / "reproduction_table.csv", header, rows)

    # ---- figure 1: RV images with the official noise regions ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), constrained_layout=True)
    for ax, ds in zip(axes, (datasets["far"], datasets["near"])):
        im = ax.imshow(
            ds.rv_db,
            origin="lower", aspect="auto", cmap="viridis",
            vmin=float(np.percentile(ds.rv_db, 5)),
            vmax=float(np.percentile(ds.rv_db, 99.5)),
            extent=(0, ds.cfg.vel_nfft, 0, ds.cfg.rng_bin_num),
        )
        b = ds.bounds
        for vel_slice, tag in ((b.vel1_slice, "half 1"), (b.vel2_slice, "half 2")):
            ax.add_patch(
                Rectangle(
                    (vel_slice.start, b.rng_slice.start),
                    vel_slice.stop - vel_slice.start,
                    b.rng_slice.stop - b.rng_slice.start,
                    fill=False, edgecolor="#ffffff", linewidth=1.2,
                )
            )
        noise_value = calculate_noise_from_rv_db(ds.rv_db, ds.cfg)
        ax.axhline(b.rng_slice.start, color="#dddddd", lw=0.6, alpha=0.6)
        ax.set_title(
            f"{KIND_LABEL[ds.kind]}: official noiseValue = {noise_value:.2f} dB\n"
            f"white boxes = official noise region "
            f"(range {b.rng_start}-{b.rng_end}, vel {b.vel1_start}-{b.vel1_end} & "
            f"{b.vel2_start}-{b.vel2_end}, 1-based)",
            fontsize=9,
        )
        ax.set_xlabel("velocity bin")
        ax.set_ylabel("range bin")
        fig.colorbar(im, ax=ax, label="RV power (dB)", shrink=0.9)
    _save(fig, out_dir / "figures" / "fig1_reproduction_rvimage.png")
    return rows, md_rows


# ---------------------------------------------------------------------------
# Experiment 2: contamination demo
# ---------------------------------------------------------------------------


def _region_centers(ds: Dataset) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """0-based (range, velocity) centers of the two half-band regions."""
    b = ds.bounds
    r_center = (b.rng_slice.start + b.rng_slice.stop - 1) // 2
    v1_center = (b.vel1_slice.start + b.vel1_slice.stop - 1) // 2
    v2_center = (b.vel2_slice.start + b.vel2_slice.stop - 1) // 2
    return (r_center, v1_center), (r_center, v2_center)


def experiment_contamination(
    ds: Dataset, out_dir: Path
) -> List[List[object]]:
    """Synthetic Gaussian peaks inside the official region of a real RV image.

    Scenarios (``sN`` = Gaussian sigma of N bins in range and velocity):

    * ``single_half_20dB_s2`` -- peak only in velocity half 1: the official
      element-wise min with the clean half 2 suppresses it (a robustness
      feature of the official design worth documenting);
    * ``both_halves_*`` -- peaks injected at the *aligned* offsets of both
      half bands (worst case for the min step): every contaminated cell of
      the combined region is hit;
    * ``s6`` variants model a *wide* contamination (e.g. leakage pedestal /
      extended clutter) rather than a compact point target.
    """
    center1, center2 = _region_centers(ds)
    scenarios: List[Tuple[str, float, List[Tuple[int, int]], float]] = [
        ("single_half_20dB_s2", 20.0, [center1], 2.0),
        ("both_halves_20dB_s2", 20.0, [center1, center2], 2.0),
        ("both_halves_40dB_s2", 40.0, [center1, center2], 2.0),
        ("both_halves_40dB_s6", 40.0, [center1, center2], 6.0),
    ]
    clean_region = ds.region
    clean = {name: robust.ESTIMATORS[name](clean_region) for name in ESTIMATOR_ORDER}

    header = ["scenario", "peak_amp_dB", "sigma_bins", "n_peak_cells", "estimator",
              "value_dB", "deviation_vs_clean_dB"]
    rows: List[List[object]] = []
    for name in ESTIMATOR_ORDER:
        rows.append(["clean", 0.0, 0.0, 0, name, clean[name], 0.0])

    for label, amp, centers, sigma in scenarios:
        img = robust.inject_gaussian_peak(ds.rv_db, centers, amp, sigma, sigma)
        region = official_region(img, ds.cfg)
        # count cells raised by > 3 dB inside the *combined* region
        n_cells = int(np.sum(region - clean_region > 3.0))
        for name in ESTIMATOR_ORDER:
            value = robust.ESTIMATORS[name](region)
            rows.append([label, amp, sigma, n_cells, name, value, value - clean[name]])
    write_csv(out_dir / "results" / "contamination_demo.csv", header, rows)

    # ---- figure 2 ----
    fig = plt.figure(figsize=(13.0, 4.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.6])
    b = ds.bounds
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax2 = fig.add_subplot(gs[2])
    region_s6 = None
    for ax, region, title in (
        (ax0, clean_region, "clean region (real data)"),
        (ax1, official_region(
            robust.inject_gaussian_peak(
                ds.rv_db, [center1, center2], 40.0, 6.0, 6.0), ds.cfg),
         "both-halves injected peak, +40 dB, sigma=6 bins (synthetic)"),
    ):
        vmax = float(np.percentile(region, 99.9))
        im = ax.imshow(region, origin="lower", aspect="auto", cmap="viridis",
                       vmin=float(np.percentile(clean_region, 1)), vmax=vmax)
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("region velocity column")
        ax.set_ylabel("region range row")
        fig.colorbar(im, ax=ax, shrink=0.85, label="dB")
    # deviation bars
    labels = [s[0] for s in scenarios]
    x = np.arange(len(ESTIMATOR_ORDER))
    width = 0.2
    scenario_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
    for si, label in enumerate(labels):
        devs = []
        for name in ESTIMATOR_ORDER:
            match = [r for r in rows if r[0] == label and r[4] == name]
            devs.append(match[0][6])
        ax2.bar(x + (si - 1.5) * width, devs, width, label=label,
                color=scenario_colors[si], edgecolor="#fcfcfb", linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(ESTIMATOR_ORDER, rotation=30, ha="right", fontsize=8)
    ax2.set_xlim(-0.8, len(ESTIMATOR_ORDER) - 0.2)
    ax2.set_ylabel("estimator deviation vs clean (dB)")
    ax2.set_title(f"{KIND_LABEL[ds.kind]}: estimator shift under\n"
                  f"synthetic Gaussian peaks (marked synthetic)", fontsize=9)
    ax2.legend(fontsize=7)
    ax2.grid(axis="y", color="#dddddd", lw=0.5)
    ax2.axhline(0.0, color="#52514e", lw=0.8)
    _save(fig, out_dir / "figures" / "fig2_contamination_demo.png")
    return rows


# ---------------------------------------------------------------------------
# Experiment 3: region scan
# ---------------------------------------------------------------------------


SCAN_START_FRACTIONS = np.linspace(0.60, 0.90, 13)
SCAN_WIDTHS = [16, 20, 24, 32, 40, 48, 64, 80, 96, 112, 128]
SCAN_VEL_MODES = ("two_half_min", "first_half")


def experiment_region_scan(datasets: Dict[str, Dataset], out_dir: Path) -> List[List[object]]:
    """Official-style region mean over a grid of alternative noise regions."""
    header = ["dataset", "vel_mode", "start_bin0", "width_bins", "mean_dB"]
    rows: List[List[object]] = []
    for ds in (datasets["far"], datasets["near"]):
        b = ds.bounds
        vel_modes = {
            "two_half_min": (b.vel1_slice, b.vel2_slice),
            "first_half": b.vel1_slice,
        }
        for mode, vel in vel_modes.items():
            for frac in SCAN_START_FRACTIONS:
                start = int(round(frac * ds.cfg.rng_bin_num))
                for width in SCAN_WIDTHS:
                    if start + width > ds.cfg.rng_bin_num:
                        continue
                    value = region_mean(ds.rv_db, slice(start, start + width), vel)
                    rows.append([ds.kind, mode, start, width, value])
    write_csv(out_dir / "results" / "region_scan.csv", header, rows)

    # ---- figure 3: heatmaps ----
    fig, axes = plt.subplots(
        2, 2, figsize=(11, 7.6), constrained_layout=True, sharex=True
    )
    for row_i, ds in enumerate((datasets["far"], datasets["near"])):
        for col_i, mode in enumerate(SCAN_VEL_MODES):
            ax = axes[row_i][col_i]
            grid = np.full((len(SCAN_START_FRACTIONS), len(SCAN_WIDTHS)), np.nan)
            for i_f, frac in enumerate(SCAN_START_FRACTIONS):
                start = int(round(frac * ds.cfg.rng_bin_num))
                for i_w, width in enumerate(SCAN_WIDTHS):
                    if start + width > ds.cfg.rng_bin_num:
                        continue
                    match = [r[4] for r in rows
                             if r[0] == ds.kind and r[1] == mode
                             and r[2] == start and r[3] == width]
                    grid[i_f, i_w] = match[0]
            im = ax.imshow(grid, origin="lower", aspect="auto", cmap="Blues")
            ax.set_xticks(range(len(SCAN_WIDTHS)))
            ax.set_xticklabels(SCAN_WIDTHS, fontsize=8)
            ax.set_yticks(range(0, len(SCAN_START_FRACTIONS), 2))
            ax.set_yticklabels(
                [int(round(f * ds.cfg.rng_bin_num)) for f in SCAN_START_FRACTIONS[::2]],
                fontsize=8,
            )
            spread = float(np.nanmax(grid) - np.nanmin(grid))
            ax.set_title(
                f"{KIND_LABEL[ds.kind]} | vel mode: {mode}\n"
                f"official-style mean (dB), spread max-min = {spread:.2f} dB",
                fontsize=9,
            )
            if row_i == 1:
                ax.set_xlabel("region width (bins)")
            ax.set_ylabel("region start (bin, 0-based)")
            fig.colorbar(im, ax=ax, shrink=0.85, label="mean dB")
    _save(fig, out_dir / "figures" / "fig3_region_scan_heatmap.png")

    # spread summary per dataset/mode
    summary_rows = []
    for ds_kind in ("far", "near"):
        for mode in SCAN_VEL_MODES:
            vals = [r[4] for r in rows if r[0] == ds_kind and r[1] == mode]
            summary_rows.append(
                [ds_kind, mode, len(vals), float(np.min(vals)), float(np.max(vals)),
                 float(np.max(vals) - np.min(vals)), float(np.std(vals))]
            )
    write_csv(
        out_dir / "results" / "region_scan_spread.csv",
        ["dataset", "vel_mode", "n_regions", "min_dB", "max_dB", "spread_dB", "std_dB"],
        summary_rows,
    )
    return summary_rows


# ---------------------------------------------------------------------------
# Experiment 4: bias curves vs injected peak power
# ---------------------------------------------------------------------------


def experiment_bias_curves(
    datasets: Dict[str, Dataset], out_dir: Path,
    sigmas: Tuple[float, float] = (2.0, 6.0),
    max_amp: float = 40.0, step: float = 2.0,
) -> List[List[object]]:
    """Deviation of every estimator vs injected both-halves peak power.

    Injection is the worst case for the official min-of-halves step (peaks at
    aligned offsets in both velocity half bands), clearly marked synthetic.
    Two peak widths are scanned: a compact target (sigma=2 bins) and a wide
    contamination pedestal (sigma=6 bins).
    """
    amps = np.arange(0.0, max_amp + 0.5 * step, step)
    header = ["dataset", "sigma_bins", "peak_amp_dB", "estimator",
              "value_dB", "deviation_dB"]
    rows: List[List[object]] = []
    for ds in (datasets["far"], datasets["near"]):
        center1, center2 = _region_centers(ds)
        centers = [center1, center2]
        clean_region = ds.region
        for sigma in sigmas:
            for amp in amps:
                img = robust.inject_gaussian_peak(ds.rv_db, centers, float(amp),
                                                  sigma, sigma)
                region = official_region(img, ds.cfg)
                for name in ESTIMATOR_ORDER:
                    clean_val = robust.ESTIMATORS[name](clean_region)
                    value = robust.ESTIMATORS[name](region)
                    rows.append([ds.kind, float(sigma), float(amp), name,
                                 value, value - clean_val])
    write_csv(out_dir / "results" / "bias_curves.csv", header, rows)

    # ---- figure 4: rows = sigma, cols = dataset ----
    fig, axes = plt.subplots(len(sigmas), 2, figsize=(11, 4.4 * len(sigmas)),
                             constrained_layout=True)
    for row_i, sigma in enumerate(sigmas):
        for col_i, ds in enumerate((datasets["far"], datasets["near"])):
            ax = axes[row_i][col_i]
            for name in ESTIMATOR_ORDER:
                pts = [(r[2], r[5]) for r in rows
                       if r[0] == ds.kind and r[1] == float(sigma) and r[3] == name]
                xs = [p[0] for p in pts]
                ys = [p[1] for p in pts]
                lw = 2.4 if name == "official_mean" else 1.6
                ax.plot(xs, ys, color=SERIES_COLORS[name], lw=lw, label=name)
            ax.set_xlabel("injected peak amplitude above local noise (dB, synthetic)")
            ax.set_ylabel("estimator deviation vs clean (dB)")
            ax.set_title(f"{KIND_LABEL[ds.kind]}, sigma={sigma:.0f} bins "
                         f"(both-halves injection)", fontsize=9)
            ax.grid(color="#dddddd", lw=0.5)
            ax.axhline(0.0, color="#52514e", lw=0.8)
    axes[0][0].legend(fontsize=8, loc="upper left")
    _save(fig, out_dir / "figures" / "fig4_bias_curves.png")
    return rows


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_all(
    data_dir: Optional[Path] = None,
    hxx_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    seed: int = 20260929,
    n_boot: int = 2000,
) -> Dict[str, object]:
    """Run every experiment; write figures/ and results/ under ``out_dir``."""
    data_dir = Path(data_dir) if data_dir else default_data_dir()
    hxx_dir = Path(hxx_dir) if hxx_dir else default_hxx_dir()
    out_dir = Path(out_dir) if out_dir else Path(__file__).resolve().parents[2]
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = {kind: load_dataset(kind, data_dir, hxx_dir) for kind in ("far", "near")}

    repro_rows, repro_md = experiment_reproduction(datasets, out_dir, seed, n_boot)
    contam_rows = experiment_contamination(datasets["far"], out_dir)
    scan_summary = experiment_region_scan(datasets, out_dir)
    bias_rows = experiment_bias_curves(datasets, out_dir)

    summary = _build_summary(datasets, repro_md, contam_rows, scan_summary,
                             bias_rows, data_dir, seed, n_boot)
    (out_dir / "results" / "summary.md").write_text(summary, encoding="utf-8")
    return {
        "datasets": datasets,
        "reproduction": repro_rows,
        "contamination": contam_rows,
        "region_scan": scan_summary,
        "bias_curves": bias_rows,
        "summary_path": out_dir / "results" / "summary.md",
    }


def _build_summary(datasets, repro_md, contam_rows, scan_summary, bias_rows,
                   data_dir, seed, n_boot) -> str:
    ds_far, ds_near = datasets["far"], datasets["near"]
    lines: List[str] = []
    add = lines.append

    add("# a100-noise-study 运行结果摘要（自动生成）")
    add("")
    add(f"- 数据目录：`{data_dir}`")
    add(f"- 随机种子：{seed}；bootstrap 次数：{n_boot}")
    add("- 诚实性声明：公开示例数据无空场真值，本文一切数值均为"
        "“相对官方方法/相对无注入基线的偏差”，不构成绝对底噪/SNR 断言。")
    add("")
    add("## 1. 配置与数据装载")
    for ds in (ds_far, ds_near):
        add(f"### {KIND_LABEL[ds.kind]}（{HXX_NAME[ds.kind]}）")
        add("")
        cfg_rows = [[k, v] for k, v in ds.cfg.summary().items()]
        add(md_table(["参数", "值"], cfg_rows))
        add("")
        add(f"官方噪声区域（MATLAB 1-based 含端点）："
            f"range {ds.bounds.rng_start}..{ds.bounds.rng_end}，"
            f"速度半区 {ds.bounds.vel1_start}..{ds.bounds.vel1_end} 与 "
            f"{ds.bounds.vel2_start}..{ds.bounds.vel2_end}，"
            f"共 {ds.bounds.region_cells} 个 cell（min 合成后）。")
        add("")
        load_rows = []
        for rep in ds.load_reports:
            tail = rep.dropped_tail_values
            load_rows.append(
                [rep.file, rep.n_values, rep.expected_len, rep.action,
                 tail if tail else "-"]
            )
        add(md_table(["文件", "实际值数", "期望值数", "处理", "被丢弃尾值"], load_rows))
        add("")
    add("## 2. 复现表（官方 noiseValue + 稳健估计量，dB）")
    add("")
    add(md_table(
        ["dataset", "channel_set", "cells", "official_mean_dB",
         "boot95(official)", "median_dB", "boot95(median)"],
        repro_md,
    ))
    add("")
    add("## 3. 合成注入污染演示（远波 4 通道，已标注为合成注入）")
    add("")
    contam_md = [[r[0], r[1], r[2], r[3], r[4], fmt(r[5], 2), fmt(r[6], 2)]
                 for r in contam_rows
                 if r[4] in ("official_mean", "median", "mad_floor")]
    add(md_table(["scenario", "peak_amp_dB", "sigma_bins", "n_hit_cells",
                  "estimator", "value_dB", "deviation_dB"], contam_md))
    add("")
    add("## 4. 区域扫描散布")
    add("")
    add(md_table(
        ["dataset", "vel_mode", "n_regions", "min_dB", "max_dB",
         "spread(max-min)_dB", "std_dB"],
        scan_summary,
    ))
    add("")
    add("## 5. 偏差曲线关键点（both-halves 注入，合成）")
    add("")
    key_md = []
    for ds_kind in ("far", "near"):
        for sigma in (2.0, 6.0):
            for amp in (20.0, 40.0):
                for name in ("official_mean", "median", "mad_floor"):
                    match = [r for r in bias_rows
                             if r[0] == ds_kind and r[1] == sigma
                             and r[2] == amp and r[3] == name]
                    key_md.append([ds_kind, sigma, amp, name,
                                   fmt(match[0][4], 2), fmt(match[0][5], 2)])
    add(md_table(["dataset", "sigma_bins", "peak_amp_dB", "estimator",
                  "value_dB", "deviation_dB"], key_md))
    add("")
    return "\n".join(lines)
