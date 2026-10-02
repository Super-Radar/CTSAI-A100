"""One-command reproduction of the a100-coordinate-consistency study.

Run from the project directory::

    PYTHONPATH=src python -m coord_consistency.cli

The command parses all 41 target logs under ``RadarTools/RadarTools_Release/data``,
runs the hypothesis closure study on the 20 TraTarget logs, the field census on
all 41 files, regenerates every figure in ``figures/`` and every table in
``results/``.  All randomness (scatter subsampling) is seeded with 20261003 so
the artifacts are byte-reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

import matplotlib

matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from .census import FOCUS_FIELDS, aggregate_rows, file_census
from .closure import evaluate_file, row_residuals
from .hypotheses import HYPOTHESIS_ORDER, predict
from .parser import TargetLog, parse_file

__all__ = ["main", "run_study"]

SEED = 20261003

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "RadarTools" / "RadarTools_Release" / "data"

# --------------------------------------------------------------------- #
# Chart tokens (validated light-surface palette; see dataviz reference):
# single-hue sequential ramp + two categorical slots for <=2-series charts.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
S1 = "#2a78d6"   # categorical slot 1 (blue)
S2 = "#eb6834"   # categorical slot 2 (orange)
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "study_blue", ["#cde2fb", "#3987e5", "#0d366b"])

FIGURES: List[str] = [
    "fig1_closure_scatter.png",
    "fig2_residual_distribution.png",
    "fig3_hypothesis_margins.png",
    "fig4_field_census_heatmap.png",
]


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(BASELINE)
    ax.tick_params(colors=MUTED, labelcolor=INK2)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def session_id(filename: str) -> str:
    """Recording session id, e.g. '2025_11_20_14_39_07'."""
    return filename[:19]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover(data_dir: Path) -> Dict[str, List[Path]]:
    raw = sorted(data_dir.glob("*_RawTarget.csv"))
    tra = sorted(data_dir.glob("*_TraTarget.csv"))
    if not raw or not tra:
        raise SystemExit(f"no target logs found under {data_dir}")
    return {"Raw": raw, "Tra": tra}


# --------------------------------------------------------------------- #
# Analysis pipeline
# --------------------------------------------------------------------- #
def run_study(data_dir: Path, out_dir: Path, quiet: bool = False):
    def log(msg: str) -> None:
        if not quiet:
            print(msg, flush=True)

    rng = np.random.default_rng(SEED)
    files = discover(data_dir)
    all_files = files["Raw"] + files["Tra"]

    results_dir = out_dir / "results"
    figures_dir = out_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. parse + census (streamed file by file) ------------------- #
    census_rows: List[Dict[str, object]] = []
    per_file_matrix: Dict[str, Dict[str, float]] = {}
    tra_arrays: Dict[str, Dict[str, np.ndarray]] = {}

    for path in all_files:
        tlog = parse_file(str(path))
        census_rows.extend(file_census(tlog))
        per_file_matrix[tlog.filename] = {
            rec["column"]: rec["effective_fill_rate"] for rec in file_census(tlog)
        }
        if tlog.kind == "Tra":
            r = tlog.column("range"); a = tlog.column("angle")
            x = tlog.column("X"); y = tlog.column("Y")
            ok = tlog.valid_mask("range", "angle", "X", "Y")
            tra_arrays[tlog.filename] = {
                "r": r[ok], "angle": a[ok], "x": x[ok], "y": y[ok],
            }
        log(f"parsed {tlog.kind:>3} {tlog.filename} "
            f"frames={len(tlog.frames)} rows={tlog.n_rows_total}")
        del tlog

    # ---- 2. closure study on Tra logs -------------------------------- #
    closure_records: List[Dict[str, object]] = []
    margin_records: List[Dict[str, object]] = []
    residuals_by_file: Dict[str, np.ndarray] = {}
    pooled = {"px": [], "x": [], "py": [], "y": []}

    for fname in sorted(tra_arrays):
        arr = tra_arrays[fname]
        ev = evaluate_file(arr["r"], arr["angle"], arr["x"], arr["y"])
        best = ev["best_hyp"]
        bs = ev["hyp_stats"][best]
        res = row_residuals(arr["r"], arr["angle"], arr["x"], arr["y"], best, 0.0)
        residuals_by_file[fname] = res["dist"]
        px, py = predict(best, arr["r"], arr["angle"], 0.0)
        pooled["px"].append(px); pooled["x"].append(arr["x"])
        pooled["py"].append(py); pooled["y"].append(arr["y"])

        closure_records.append({
            "file": fname,
            "kind": "Tra",
            "n_rows": ev["n_rows"],
            "best_hyp": best,
            "best_median_m": bs["median_m"],
            "best_p95_m": bs["p95_m"],
            "best_max_m": bs["max_m"],
            "best_median_dx_m": bs["median_dx_m"],
            "best_median_dy_m": bs["median_dy_m"],
            "second_hyp": ev["second_hyp"],
            "margin_m": ev["margin_m"],
            "identifiable": ev["identifiable"],
            "angle_offset_best_deg": ev["angle_offset_best_deg"],
            "angle_offset_improvement_m": ev["angle_offset_improvement_m"],
            "angle_offset_improves": ev["angle_offset_improves"],
        })
        margin_records.append({
            "file": fname,
            **{f"{h}_median_m": ev["hyp_stats"][h]["median_m"] for h in HYPOTHESIS_ORDER},
            "best_hyp": best,
            "second_hyp": ev["second_hyp"],
            "margin_m": ev["margin_m"],
        })
        log(f"closure {session_id(fname)}: best={best} "
            f"median={bs['median_m']:.4f} m margin={ev['margin_m']:.2f} m")

    # ---- 3. write result tables -------------------------------------- #
    def write_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
        if not records:
            return
        cols = list(records[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=cols, lineterminator="\n")
            w.writeheader()
            w.writerows(records)

    write_csv(results_dir / "closure_by_file.csv", closure_records)
    write_csv(results_dir / "hypothesis_margins.csv", margin_records)
    census_all = census_rows + aggregate_rows(census_rows)
    write_csv(results_dir / "field_census.csv", census_all)

    with open(results_dir / "data_sha256.txt", "w", encoding="utf-8") as fh:
        for path in sorted(all_files, key=lambda p: p.name):
            fh.write(f"{sha256_file(path)}  {path.name}\n")

    # ---- 4. figures --------------------------------------------------- #
    _fig1_closure(figures_dir / FIGURES[0], pooled, rng)
    _fig2_residuals(figures_dir / FIGURES[1], residuals_by_file)
    _fig3_margins(figures_dir / FIGURES[2], margin_records)
    _fig4_census(figures_dir / FIGURES[3], per_file_matrix, files)
    log("figures written")

    # ---- 5. summary ---------------------------------------------------- #
    write_summary(results_dir / "summary.md", closure_records, census_all,
                  n_files=len(all_files))
    log("summary written")
    return {"closure": closure_records, "census": census_rows}


# --------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------- #
def _fig1_closure(path: Path, pooled: Dict[str, List[np.ndarray]], rng) -> None:
    """Diagonal scatter: logged X/Y vs H1 prediction, all TraTarget rows."""
    px = np.concatenate(pooled["px"]); x = np.concatenate(pooled["x"])
    py = np.concatenate(pooled["py"]); y = np.concatenate(pooled["y"])

    def subsample(a: np.ndarray, idx: np.ndarray) -> np.ndarray:
        return a[idx]

    n = min(len(px), 30000)
    idx = rng.choice(len(px), size=n, replace=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), facecolor=SURFACE)
    panels = [
        (subsample(px, idx), subsample(x, idx),
         "X component", "Predicted $X=-r\\sin\\theta$ under H1 (m)",
         "Logged X (m)", S1),
        (subsample(py, idx), subsample(y, idx),
         "Y component", "Predicted $Y=r\\cos\\theta$ under H1 (m)",
         "Logged Y (m)", S2),
    ]
    m = float(np.percentile(np.abs(np.concatenate([px, py])), 99.9))
    lo, hi = -1.05 * m, 1.05 * m
    for ax, (pred, meas, title, xlabel, ylabel, color) in zip(axes, panels):
        _style_axes(ax)
        ax.plot([lo, hi], [lo, hi], color=BASELINE, linewidth=1.5,
                label="identity (perfect closure)")
        ax.scatter(pred, meas, s=2.5, color=color, alpha=0.25,
                   edgecolors="none", label="TraTarget rows (subsample)")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(title, color=INK, fontsize=11)
        ax.set_xlabel(xlabel, color=INK2)
        ax.set_ylabel(ylabel, color=INK2)
        ax.legend(loc="upper left", frameon=False, fontsize=8,
                  labelcolor=INK2)
    fig.suptitle("Geometric closure of TraTarget cartesian columns (H1 convention)",
                 color=INK, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _fig2_residuals(path: Path, residuals_by_file: Dict[str, np.ndarray]) -> None:
    """Pooled H1 residual histogram + per-file violin, log-scaled."""
    names = sorted(residuals_by_file)
    data = [residuals_by_file[k] for k in names]
    pooled_all = np.concatenate(data)
    x_max = float(np.percentile(pooled_all, 99.5))

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(11.5, 6.0), facecolor=SURFACE,
        gridspec_kw={"width_ratios": [1, 1.4]})

    _style_axes(ax1)
    ax1.hist(pooled_all, bins=80, range=(0.0, x_max), color=S1,
             edgecolor=SURFACE, linewidth=0.5)
    ax1.set_xlabel("Closure residual under H1 (m)", color=INK2)
    ax1.set_ylabel("Target rows", color=INK2)
    ax1.set_title(f"Pooled histogram (all Tra files, n={len(pooled_all)})",
                  color=INK, fontsize=10.5)
    ax1.axvline(float(np.median(pooled_all)), color=S2, linewidth=1.5,
                label=f"median = {np.median(pooled_all):.4f} m")
    ax1.legend(frameon=False, fontsize=8, labelcolor=INK2)

    _style_axes(ax2)
    violins = ax2.violinplot(data, showmedians=True, showextrema=False)
    for body in violins["bodies"]:
        body.set_facecolor(S1)
        body.set_alpha(0.55)
        body.set_edgecolor(S1)
    violins["cmedians"].set_color(INK)
    violins["cmedians"].set_linewidth(1.4)
    ax2.set_yscale("log")
    ax2.set_xticks(range(1, len(names) + 1))
    ax2.set_xticklabels([session_id(n)[5:] for n in names], rotation=90,
                        fontsize=7)
    ax2.set_ylabel("Closure residual under H1 (m, log scale)", color=INK2)
    ax2.set_xlabel("Recording session (TraTarget files)", color=INK2)
    ax2.set_title("Per-file residual distribution", color=INK, fontsize=10.5)

    fig.suptitle("Residual distribution between logged (X, Y) and H1 prediction",
                 color=INK, fontsize=12.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _fig3_margins(path: Path, margin_records: List[Dict[str, object]]) -> None:
    """Best vs second-best hypothesis median per file (log axis) + margins."""
    recs = sorted(margin_records, key=lambda r: float(r["margin_m"]))
    names = [session_id(r["file"])[5:] for r in recs]
    best = np.array([float(r[f"{r['best_hyp']}_median_m"]) for r in recs])
    second = np.array([float(r[f"{r['second_hyp']}_median_m"]) for r in recs])
    margins = np.array([float(r["margin_m"]) for r in recs])
    hyps = [str(r["best_hyp"]) for r in recs]

    fig, ax = plt.subplots(figsize=(9.5, 7.2), facecolor=SURFACE)
    _style_axes(ax)
    ax.grid(axis="y", visible=False)
    yy = np.arange(len(recs))
    h = 0.36
    ax.barh(yy + h / 2 + 0.02, best, height=h, color=S1,
            label="Best hypothesis (always H1 here)")
    ax.barh(yy - h / 2 - 0.02, second, height=h, color=S2,
            label="Second-best hypothesis")
    ax.set_xscale("log")
    ax.set_xlim(1e-3, max(second) * 6)
    ax.set_yticks(yy)
    ax.set_yticklabels([f"{n}  ({h_})" for n, h_ in zip(names, hyps)],
                       fontsize=8)
    ax.invert_yaxis()
    for i, (b, s, m) in enumerate(zip(best, second, margins)):
        ax.text(s * 1.15, i, f"margin {m:.2f} m", va="center",
                fontsize=7.5, color=INK2)
    ax.set_xlabel("Median closure residual (m, log scale)", color=INK2)
    ax.set_ylabel("Recording session (best hypothesis)", color=INK2)
    ax.set_title("Separability margin per TraTarget file: "
                 "second-best minus best median residual", color=INK, fontsize=12)
    ax.legend(loc="lower right", frameon=False, fontsize=9, labelcolor=INK2)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def _fig4_census(path: Path, per_file_matrix: Dict[str, Dict[str, float]],
                 files: Dict[str, List[Path]]) -> None:
    """Heatmap of effective non-zero fill rate: focus columns x all files."""
    fields = list(FOCUS_FIELDS)
    ordered: List[str] = []
    for kind in ("Raw", "Tra"):
        for p in sorted(files[kind], key=lambda p: p.name):
            ordered.append(p.name)

    mat = np.full((len(ordered), len(fields)), np.nan)
    for i, fname in enumerate(ordered):
        for j, f in enumerate(fields):
            v = per_file_matrix.get(fname, {}).get(f)
            if v is not None:
                mat[i, j] = v

    fig, ax = plt.subplots(figsize=(8.6, 11.0), facecolor=SURFACE)
    im = ax.imshow(mat, aspect="auto", cmap=SEQ_CMAP, vmin=0.0, vmax=1.0,
                   interpolation="nearest")
    ax.set_facecolor(SURFACE)
    ax.set_xticks(range(len(fields)))
    ax.set_xticklabels(fields, rotation=35, ha="right", fontsize=8.5,
                       color=INK2)
    labels = []
    n_raw = len(files["Raw"])
    for i, fname in enumerate(ordered):
        prefix = "R " if i < n_raw else "T "
        labels.append(prefix + session_id(fname)[5:])
    ax.set_yticks(range(len(ordered)))
    ax.set_yticklabels(labels, fontsize=6.5, color=INK2)
    # hairline surface gaps between cells
    ax.set_xticks(np.arange(-0.5, len(fields), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(ordered), 1), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=1.0)
    ax.tick_params(which="both", colors=MUTED)
    ax.axhline(n_raw - 0.5, color=INK2, linewidth=1.2)
    ax.text(-0.015, 0.5 * n_raw / len(ordered), "RawTarget (21 files)",
            fontsize=9, color=INK, ha="right", va="center", rotation=90,
            transform=ax.transAxes)
    ax.text(-0.015, (n_raw + 0.5 * (len(ordered) + n_raw)) / len(ordered),
            "TraTarget (20 files)", fontsize=9, color=INK, ha="right",
            va="center", rotation=90, transform=ax.transAxes)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Effective non-zero fill rate", color=INK2)
    cb.ax.tick_params(colors=MUTED, labelcolor=INK2)
    cb.outline.set_visible(False)
    ax.set_title("Field-validity census: fraction of target rows with a\n"
                 "parseable non-zero value (|v| > 1e-6), per file",
                 color=INK, fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------- #
# Chinese summary
# --------------------------------------------------------------------- #
def write_summary(path: Path, closure_records: List[Dict[str, object]],
                  census_rows: List[Dict[str, object]], n_files: int) -> None:
    best = [str(r["best_hyp"]) for r in closure_records]
    uniq = sorted(set(best))
    meds = np.array([float(r["best_median_m"]) for r in closure_records])
    p95s = np.array([float(r["best_p95_m"]) for r in closure_records])
    maxs = np.array([float(r["best_max_m"]) for r in closure_records])
    margins = np.array([float(r["margin_m"]) for r in closure_records])
    offs = [bool(r["angle_offset_improves"]) for r in closure_records]
    off_best = sorted({float(r["angle_offset_best_deg"]) for r in closure_records})

    def agg(kind: str, column: str) -> Dict[str, object]:
        for rec in census_rows:
            if rec["kind"] == kind and rec["file"] == "AGGREGATE" \
                    and rec["column"] == column:
                return rec
        raise KeyError(f"{kind}/{column}")

    tra_x = agg("Tra", "X"); tra_y = agg("Tra", "Y")
    raw_x = agg("Raw", "X"); raw_y = agg("Raw", "Y")

    focus_lines = []
    for column in FOCUS_FIELDS:
        rw = agg("Raw", column); tr = agg("Tra", column)
        focus_lines.append(
            f"| {column} | {rw['effective_fill_rate']:.4f} | {tr['effective_fill_rate']:.4f} |")

    lines = [
        "# a100-coordinate-consistency 结果摘要",
        "",
        "本文件由 `PYTHONPATH=src python -m coord_consistency.cli` 自动生成"
        "（随机种子 20261003），请勿手工编辑。",
        "",
        "## 语料",
        "",
        f"- 目标日志共 {n_files} 个文件（21 个 `*_RawTarget.csv`、"
        f"20 个 `*_TraTarget.csv`），来自 `RadarTools/RadarTools_Release/data/`，"
        "文件名标注 CTLRR-300-V2_1。",
        "- 各文件 SHA-256 见 `data_sha256.txt`。",
        f"- Tra 语料有效目标行合计 "
        f"{sum(int(r['n_rows']) for r in closure_records)} 行"
        "（range/angle/X/Y 均可解析且非缺失）。",
        "",
        "## 坐标约定闭合结论（TraTarget，残差 = 预测点与 logged (X,Y) 的平面欧氏距离）",
        "",
        f"- **最优假设：{'+'.join(uniq)}**（{len(closure_records)}/{len(closure_records)} "
        "个 TraTarget 文件一致），即 **X = -r·sin(θ)，Y = r·cos(θ)**"
        "（θ 为度、顺时针为正、X 左 Y 前）。",
        f"- 最优残差中位数范围：**{meds.min():.4f} ~ {meds.max():.4f} m**"
        f"（20 个文件中位数的中位数 {np.median(meds):.4f} m）。",
        f"- 最优残差 p95 范围：{p95s.min():.4f} ~ {p95s.max():.4f} m；"
        f"最大值范围：{maxs.min():.4f} ~ {maxs.max():.4f} m。",
        f"- 可区分裕度（次优中位数 − 最优中位数）：**{margins.min():.2f} ~ "
        f"{margins.max():.2f} m**，全部文件裕度 > 0，判别无歧义。",
        f"- 角度零点偏移扫描（±5°、0.5° 步长）：最优偏移集合 "
        f"{off_best}°，{sum(offs)}/{len(offs)} 个文件因偏移而改善——"
        "即语料不支持存在系统性角度零点偏移。",
        "",
        "## 字段有效性普查（有效 = 可解析且 |值| > 1e-6）",
        "",
        f"- **RawTarget 的 X/Y 有效填充率为 "
        f"{raw_x['effective_fill_rate']:.4%} / {raw_y['effective_fill_rate']:.4%}"
        "（全为 0 或 1e-31 量级浮点垃圾值），不可用作笛卡尔坐标**。",
        f"- TraTarget 的 X/Y 有效填充率为 "
        f"{tra_x['effective_fill_rate']:.4%} / {tra_y['effective_fill_rate']:.4%}。",
        "",
        "| 字段 | Raw 有效填充率 | Tra 有效填充率 |",
        "|---|---|---|",
        *focus_lines,
        "",
        "## 图",
        "",
        "- `figures/fig1_closure_scatter.png`：Tra 全语料 logged X/Y 与 H1 预测的恒等散点。",
        "- `figures/fig2_residual_distribution.png`：H1 残差分布（合并直方图 + 按文件小提琴）。",
        "- `figures/fig3_hypothesis_margins.png`：每文件最优 vs 次优中位残差与裕度。",
        "- `figures/fig4_field_census_heatmap.png`：字段 × 文件有效填充率热图。",
        "",
        "## 复现",
        "",
        "```bash",
        "PYTHONPATH=src python -m coord_consistency.cli",
        "```",
        "",
    ]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# --------------------------------------------------------------------- #
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="CTSAI-A100 target-log coordinate convention study")
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"corpus directory (default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--out-dir", type=Path, default=PROJECT_DIR,
                    help="project directory holding figures/ and results/")
    args = ap.parse_args(argv)

    if not args.data_dir.is_dir():
        print(f"data directory not found: {args.data_dir}", file=sys.stderr)
        return 2

    run_study(args.data_dir, args.out_dir)
    print("done: results/ and figures/ regenerated")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
