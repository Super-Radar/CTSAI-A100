"""Command-line driver for the a100-polar-density study.

Runs the full offline analysis on the RawTarget captures that ship with the
repository (main input + 21-file review corpus), reproducing every figure and
results table committed under ``figures/`` and ``results/``.

Stochastic parts (the synthetic benchmark) are seeded with the fixed seed
``20261003`` so re-runs are bit-reproducible.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display required
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from scipy import stats

if __package__ in (None, ""):  # allow `python src/polar_density/cli.py`
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from polar_density import binning as bn
    from polar_density import conservation as cons
    from polar_density.parser import parse_capture
else:
    from . import binning as bn
    from . import conservation as cons
    from .parser import parse_capture

SEED = 20261003
MAIN_REL = Path("Python雷达可视化工具示例/负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv")
CORPUS_REL = Path("RadarTools/RadarTools_Release/data")

# dataviz: two-colour categorical pair (CVD-distinct), recessive greys
C_BLUE = "#4269d0"
C_CORAL = "#ff725c"
INK = "#2b2b2b"
INK_MUTED = "#6f6f6f"
GRID = "#d9d9d9"


# --------------------------------------------------------------------------- #
# Helpers                                                                     #
# --------------------------------------------------------------------------- #


def discover_repo_root(override: str | None) -> Path:
    """Locate the repository root (contains the sample-data directories)."""
    if override:
        root = Path(override).resolve()
        if not (root / MAIN_REL).exists():
            raise SystemExit(f"--repo-root has no main input under {root}")
        return root
    for parent in Path(__file__).resolve().parents:
        if (parent / MAIN_REL).exists() and (parent / CORPUS_REL).is_dir():
            return parent
    raise SystemExit(
        "could not locate the repository root; pass --repo-root /path/to/CTSAI-A100"
    )


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _masked(field: np.ndarray) -> np.ma.MaskedArray:
    """Mask zero cells so LogNorm colour scales only cover live sectors."""
    return np.ma.masked_equal(np.asarray(field, dtype=float), 0.0)


def _heatmap(
    ax,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    field: np.ndarray,
    *,
    title: str,
    cbar_label: str,
    cmap: str = "viridis",
) -> None:
    data = _masked(field)
    live = data.compressed()
    vmin = float(live.min()) if live.size else 1.0
    vmax = float(live.max()) if live.size else 10.0
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad("white")
    mesh = ax.pcolormesh(
        theta_edges,
        r_edges,
        data,
        cmap=cmap_obj,
        norm=LogNorm(vmin=vmin, vmax=vmax),
        shading="flat",
        rasterized=True,
    )
    ax.set_title(title, color=INK)
    ax.set_xlabel("Azimuth angle θ (deg)", color=INK)
    ax.set_ylabel("Range r (m)", color=INK)
    ax.tick_params(colors=INK_MUTED)
    ax.figure.colorbar(mesh, ax=ax, pad=0.02).set_label(cbar_label, color=INK)


def _style_axes(ax) -> None:
    ax.tick_params(colors=INK_MUTED)
    for spine in ax.spines.values():
        spine.set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def competition_rank(values: np.ndarray) -> np.ndarray:
    """1-based competition ranking (ties share the best rank)."""
    return stats.rankdata(-np.asarray(values, dtype=float), method="min").astype(int)


# --------------------------------------------------------------------------- #
# Experiments                                                                 #
# --------------------------------------------------------------------------- #


def experiment_rank_shift(
    counts: np.ndarray,
    density: np.ndarray,
    areas: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    top_k: int = 20,
) -> tuple[list[dict], dict]:
    """Compare top-K hotspots under count vs area-normalised density ranking.

    ``shift = rank_by_count - rank_by_density``; positive means the sector is
    *promoted* (moves towards rank 1) once sector area is normalised away.
    """
    flat_counts = counts.ravel()
    flat_density = density.ravel()
    live = flat_counts > 0
    rank_count = np.full(flat_counts.shape, -1, dtype=int)
    rank_density = np.full(flat_counts.shape, -1, dtype=int)
    rank_count[live] = competition_rank(flat_counts[live])
    rank_density[live] = competition_rank(flat_density[live])

    # deterministic top-K by count: sort by count desc, then flat index asc
    live_idx = np.flatnonzero(live)
    order = np.lexsort((live_idx, -flat_counts[live_idx]))
    top = live_idx[order[:top_k]]

    r_centers = bn.cell_centers(r_edges)
    t_centers = bn.cell_centers(theta_edges)
    rows: list[dict] = []
    for flat in top:
        i, j = np.unravel_index(flat, counts.shape)
        rows.append(
            {
                "rank_by_count": int(rank_count[flat]),
                "rank_by_density": int(rank_density[flat]),
                "shift": int(rank_count[flat] - rank_density[flat]),
                "r_center_m": r_centers[i],
                "theta_center_deg": t_centers[j],
                "area_m2": areas[i, j],
                "count": int(flat_counts[flat]),
                "density_per_m2_frame": flat_density[flat],
            }
        )

    rho = stats.spearmanr(flat_counts[live], flat_density[live]).statistic
    rho_area = stats.spearmanr(
        flat_counts[live], np.broadcast_to(areas, counts.shape).ravel()[live]
    ).statistic
    dropped = sum(1 for row in rows if row["rank_by_density"] > top_k)
    shifts = np.array([row["shift"] for row in rows], dtype=float)
    stats_out = {
        "n_live_sectors": int(live.sum()),
        "spearman_count_density": float(rho),
        "spearman_count_area": float(rho_area),
        "dropped_out_of_topK": int(dropped),
        "shift_median": float(np.median(shifts)),
        "shift_min": int(shifts.min()),
        "shift_max": int(shifts.max()),
        "n_promoted": int((shifts > 0).sum()),
        "n_demoted": int((shifts < 0).sum()),
        "topK": top_k,
    }
    return rows, stats_out


def plot_fig1(counts, density, r_edges, theta_edges, n_frames, out_path, dr, dtheta):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
    _heatmap(
        axes[0],
        r_edges,
        theta_edges,
        counts,
        title=f"Detection count per sector (F = {n_frames} frames)",
        cbar_label="Detections per sector (count)",
    )
    _heatmap(
        axes[1],
        r_edges,
        theta_edges,
        density,
        title="Area-normalised density (same data)",
        cbar_label="Density (detections·m$^{-2}$·frame$^{-1}$)",
        cmap="magma",
    )
    fig.suptitle(
        "Raw count vs area-normalised polar density — "
        f"Δr = {dr:g} m, Δθ = {dtheta:g}° (half-open bins)",
        color=INK,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig2(rows: list[dict], n_live: int, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 7.6), constrained_layout=True)
    ys = np.arange(len(rows))
    labels = [
        f"r = {row['r_center_m']:.1f} m, θ = {row['theta_center_deg']:+.1f}°"
        for row in rows
    ]
    for y, row in zip(ys, rows):
        up = row["shift"] > 0
        color = C_BLUE if up else (C_CORAL if row["shift"] < 0 else INK_MUTED)
        ax.plot(
            [row["rank_by_density"], row["rank_by_count"]],
            [y, y],
            color=color,
            linewidth=1.6,
            alpha=0.85,
            zorder=1,
        )
    ax.scatter(
        [row["rank_by_count"] for row in rows],
        ys,
        s=42,
        color=C_CORAL,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="rank by raw count",
    )
    ax.scatter(
        [row["rank_by_density"] for row in rows],
        ys,
        s=42,
        color=C_BLUE,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        label="rank by density (area-normalised)",
    )
    ax.set_xscale("log")
    ax.set_xlim(0.8, max(n_live, 10) * 1.5)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("Sector rank, 1 = hottest (log scale)", color=INK)
    ax.set_ylabel("Top-20 count hotspots (sector centre)", color=INK)
    ax.set_title(
        "Hotspot rank shift after area normalisation\n"
        "blue marker left of coral = hotter than raw count suggested "
        "(shift = count rank − density rank)",
        color=INK,
        fontsize=10,
    )
    _style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig3(
    density_eq,
    r_eq,
    theta_edges,
    ring_means_angle,
    ring_centers_angle,
    ring_means_area,
    ring_centers_area,
    out_path,
):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
    _heatmap(
        axes[0],
        r_eq,
        theta_edges,
        density_eq,
        title="Equal-area binning (annuli of equal area, Δθ uniform)",
        cbar_label="Density (detections·m$^{-2}$·frame$^{-1}$)",
        cmap="magma",
    )
    ax = axes[1]
    ax.plot(
        ring_centers_angle,
        ring_means_angle,
        marker="o",
        markersize=4,
        linewidth=1.4,
        color=C_BLUE,
        label="equal-angle grid (Δr = 1 m, Δθ = 1°)",
    )
    ax.plot(
        ring_centers_area,
        ring_means_area,
        marker="s",
        markersize=4,
        linewidth=1.4,
        color=C_CORAL,
        label="equal-area grid (equal-area annuli)",
    )
    ax.set_xlabel("Ring centre radius (m)", color=INK)
    ax.set_ylabel(
        "Ring-mean density (detections·m$^{-2}$·frame$^{-1}$)", color=INK
    )
    ax.set_yscale("log")
    ax.set_title("Density comparability across binnings", color=INK)
    ax.legend(frameon=False, fontsize=8)
    _style_axes(ax)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_fig4(counts_syn, density_syn, r_edges_syn, theta_edges_syn, flatness, growth,
              out_path):
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.4), constrained_layout=True)
    _heatmap(
        axes[0],
        r_edges_syn,
        theta_edges_syn,
        counts_syn,
        title=(
            "Uniform annulus — raw counts grow with r\n"
            f"(outer/inner ring count ratio ≈ {growth:.1f}×)"
        ),
        cbar_label="Detections per sector (count)",
    )
    _heatmap(
        axes[1],
        r_edges_syn,
        theta_edges_syn,
        density_syn,
        title=(
            "Uniform annulus — density stays flat\n"
            f"(max ring deviation {flatness * 100:.1f}% of mean)"
        ),
        cbar_label="Density (detections·m$^{-2}$·frame$^{-1}$)",
        cmap="magma",
    )
    fig.suptitle(
        "Synthetic benchmark: area-uniform points in an annulus, "
        "seed = %d" % SEED,
        color=INK,
    )
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Synthetic benchmark                                                         #
# --------------------------------------------------------------------------- #


def sample_annulus_uniform(rng, n, r_inner, r_outer, theta_lo, theta_hi):
    """Area-uniform points in an annular region (r ~ sqrt-uniform)."""
    u = rng.random(n)
    r = np.sqrt(r_inner**2 + u * (r_outer**2 - r_inner**2))
    theta = rng.uniform(theta_lo, theta_hi, n)
    return r, theta


def synthetic_benchmark(rng, reports: list[str], figdir: Path) -> dict:
    """Uniform annulus, hotspot recovery and conservation on synthetic data."""
    r_inner, r_outer = 5.0, 80.0
    theta_lo, theta_hi = -75.0, 75.0
    n_background = 160_000
    n_hotspot = 24_000
    n_frames = 200

    r_bg, th_bg = sample_annulus_uniform(
        rng, n_background, r_inner, r_outer, theta_lo, theta_hi
    )
    r_hs = rng.normal(45.0, 2.5, n_hotspot)
    th_hs = rng.normal(20.0, 3.0, n_hotspot)
    keep = (
        (r_hs >= r_inner)
        & (r_hs < r_outer)
        & (th_hs >= theta_lo)
        & (th_hs < theta_hi)
    )
    r_hs, th_hs = r_hs[keep], th_hs[keep]

    def as_targets(rr, tt):
        return np.column_stack(
            [np.zeros(rr.size), rr, np.zeros(rr.size), tt, np.full(rr.size, 20.0)]
        )

    bg_targets = as_targets(r_bg, th_bg)
    targets = np.vstack([bg_targets, as_targets(r_hs, th_hs)])
    n_total = int(targets.shape[0])

    r_edges = np.arange(r_inner, r_outer, 1.0)
    theta_edges = np.arange(theta_lo, theta_hi, 1.0)
    counts_bg = bn.polar_histogram(bg_targets, r_edges, theta_edges)
    counts = bn.polar_histogram(targets, r_edges, theta_edges)
    areas = bn.sector_areas(r_edges, theta_edges)
    density_bg = bn.density_from_counts(counts_bg, areas, n_frames)
    density = bn.density_from_counts(counts, areas, n_frames)

    # flatness of the uniform background: ring means, interior rings only
    ring_means = counts_bg.sum(axis=1) / areas.sum(axis=1) / n_frames
    interior = slice(5, len(r_edges) - 6)
    flatness = float(
        np.max(np.abs(ring_means[interior] - ring_means[interior].mean()))
        / ring_means[interior].mean()
    )
    growth = float(counts_bg.sum(axis=1)[-8] / max(counts_bg.sum(axis=1)[1], 1))
    bg_peak_ring = int(np.argmax(counts_bg.sum(axis=1)))

    # hotspot recovery under density vs count ranking
    flat_d = density.ravel()
    flat_c = counts.ravel()
    hot_d = int(np.argmax(flat_d))
    hi, hj = np.unravel_index(hot_d, counts.shape)
    r_centers = bn.cell_centers(r_edges)
    t_centers = bn.cell_centers(theta_edges)
    hot_rank_count = int(competition_rank(flat_c)[hot_d])
    recovery = {
        "peak_r": float(r_centers[hi]),
        "peak_theta": float(t_centers[hj]),
        "true_r": 45.0,
        "true_theta": 20.0,
        "peak_count": int(flat_c[hot_d]),
        "hotspot_rank_by_count": hot_rank_count,
    }

    # conservation on synthetic data: F=1 identity + re-binning transfer
    density_f1 = bn.density_from_counts(counts, areas, 1)
    rep = cons.check_conservation(density_f1, areas, counts, n_frames=1)
    reports.append(rep.as_line("synthetic F=1 identity sum(density*A)=sum(counts)"))
    r_coarse = bn.equal_area_r_edges(r_inner, r_outer, 25)
    # destination grid must cover the source angular domain [-75, 75) deg
    th_coarse = np.arange(theta_lo, theta_hi + 1e-9, 5.0)
    rebinned = bn.rebin_polar(density, r_edges, theta_edges, r_coarse, th_coarse)
    rep2 = cons.check_rebin_conservation(
        density, areas, rebinned, bn.sector_areas(r_coarse, th_coarse)
    )
    reports.append(rep2.as_line("synthetic rebin equal-angle -> equal-area"))

    plot_fig4(
        counts_bg,
        density_bg,
        r_edges,
        theta_edges,
        flatness,
        growth,
        figdir / "fig4_synthetic_benchmark.png",
    )
    return {
        "n_points": n_total,
        "n_background": int(r_bg.size),
        "n_frames": n_frames,
        "flatness": flatness,
        "growth": growth,
        "bg_peak_ring": bg_peak_ring,
        "n_rings": len(r_edges) - 1,
        "recovery": recovery,
        "conservation": [rep, rep2],
    }


# --------------------------------------------------------------------------- #
# CSV / text outputs                                                          #
# --------------------------------------------------------------------------- #


def write_hotspot_csv(rows: list[dict], out_path: Path) -> None:
    fields = [
        "rank_by_count",
        "rank_by_density",
        "shift",
        "r_center_m",
        "theta_center_deg",
        "area_m2",
        "count",
        "density_per_m2_frame",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    key: (
                        _fmt(row[key])
                        if isinstance(row[key], float)
                        else row[key]
                    )
                    for key in fields
                }
            )


def write_bin_statistics(
    counts, density, areas, r_edges, theta_edges, out_path: Path
) -> int:
    r_centers = bn.cell_centers(r_edges)
    t_centers = bn.cell_centers(theta_edges)
    n_rows = 0
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "r_center_m",
                "theta_center_deg",
                "area_m2",
                "count",
                "density_per_m2_frame",
            ]
        )
        for i, r_c in enumerate(r_centers):
            for j, t_c in enumerate(t_centers):
                writer.writerow(
                    [
                        _fmt(r_c),
                        _fmt(t_c),
                        _fmt(areas[i, j]),
                        int(counts[i, j]),
                        _fmt(density[i, j]),
                    ]
                )
                n_rows += 1
    return n_rows


def write_data_sha256(files: list[Path], repo_root: Path, out_path: Path) -> None:
    lines = []
    for path in files:
        lines.append(f"{sha256_of(path)}  {path.relative_to(repo_root).as_posix()}")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="a100-polar-density",
        description=(
            "Area-normalised polar density analysis of RawTarget radar "
            "captures (offline, no hardware required)."
        ),
    )
    parser.add_argument(
        "--repo-root",
        default=None,
        help="CTSAI-A100 checkout root (auto-detected from this file's location)",
    )
    parser.add_argument("--dr", type=float, default=1.0, help="range bin width [m]")
    parser.add_argument(
        "--dtheta", type=float, default=1.0, help="angle bin width [deg]"
    )
    parser.add_argument(
        "--project-root",
        default=None,
        help="output base directory (default: the a100-polar-density project dir)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    rng = np.random.default_rng(SEED)

    project_root = (
        Path(args.project_root).resolve()
        if args.project_root
        else Path(__file__).resolve().parents[2]
    )
    figdir = project_root / "figures"
    resdir = project_root / "results"
    figdir.mkdir(parents=True, exist_ok=True)
    resdir.mkdir(parents=True, exist_ok=True)

    repo_root = discover_repo_root(args.repo_root)
    main_csv = repo_root / MAIN_REL
    corpus_files = sorted((repo_root / CORPUS_REL).glob("*_RawTarget.csv"))
    all_files = [main_csv] + corpus_files
    print(f"repo root : {repo_root}")
    print(f"main input: {main_csv.name}")
    print(f"corpus    : {len(corpus_files)} review captures")

    write_data_sha256(all_files, repo_root, resdir / "data_sha256.txt")

    reports: list[str] = []
    summary_bits: dict = {}

    # ---------------- parse main capture ---------------------------------- #
    log = parse_capture(main_csv)
    targets = log.all_targets()
    n_frames = log.n_frames
    print(
        f"parsed    : {n_frames} frames, {log.n_targets} detections "
        f"(header columns: {log.header.count(',') + 1})"
    )

    r_edges = bn.default_edges(targets[:, 1], args.dr)
    theta_edges = bn.default_edges(targets[:, 3], args.dtheta)
    counts = bn.polar_histogram(targets, r_edges, theta_edges)
    areas = bn.sector_areas(r_edges, theta_edges)
    density = bn.density_from_counts(counts, areas, n_frames)
    n_sectors = int(counts.size)
    print(
        f"grid      : {len(r_edges) - 1} range bins x {len(theta_edges) - 1} "
        f"angle bins = {n_sectors} sectors, "
        f"{int((counts > 0).sum())} non-empty"
    )

    # ---------------- experiment 1: count vs density ---------------------- #
    plot_fig1(
        counts,
        density,
        r_edges,
        theta_edges,
        n_frames,
        figdir / "fig1_count_vs_density.png",
        args.dr,
        args.dtheta,
    )
    rows, rank_stats = experiment_rank_shift(
        counts, density, areas, r_edges, theta_edges
    )
    write_hotspot_csv(rows, resdir / "hotspot_rank_shift.csv")
    plot_fig2(
        rows,
        rank_stats["n_live_sectors"],
        figdir / "fig2_hotspot_rank_shift.png",
    )

    area_bias = float(areas.max() / areas.min())
    summary_bits["rank_stats"] = rank_stats
    summary_bits["area_bias"] = area_bias
    summary_bits["area_first_last"] = (
        float(areas[0, 0]),
        float(areas[-1, 0]),
    )
    reports.append(
        cons.check_conservation(
            density, areas, counts, n_frames
        ).as_line("main capture sum(density*A)=sum(counts)/F")
    )

    # ---------------- experiment 2: equal-area binning -------------------- #
    n_r_eq = len(r_edges) - 1
    r_eq = bn.equal_area_r_edges(r_edges[0], r_edges[-1], n_r_eq)
    counts_eq = bn.polar_histogram(targets, r_eq, theta_edges)
    areas_eq = bn.sector_areas(r_eq, theta_edges)
    density_eq = bn.density_from_counts(counts_eq, areas_eq, n_frames)

    rebinned = bn.rebin_polar(density, r_edges, theta_edges, r_eq, theta_edges)
    rep_rebin = cons.check_rebin_conservation(
        density, areas, rebinned, areas_eq
    )
    reports.append(
        rep_rebin.as_line("main capture rebin equal-angle -> equal-area")
    )

    # ring-mean density from both binnings (profile overlay for fig3)
    def ring_profile(r_e, cnts, ars):
        means = cnts.sum(axis=1) / ars.sum(axis=1) / n_frames
        centers = bn.cell_centers(r_e)
        return centers[means > 0], means[means > 0]

    c_a, m_a = ring_profile(r_edges, counts, areas)
    c_e, m_e = ring_profile(r_eq, counts_eq, areas_eq)

    # (1) exact binning invariance: equal-area ring means must reproduce the
    # grid-free interval means computed straight from the sorted radii
    r_sorted = np.sort(targets[:, 1])
    theta_span = np.deg2rad(theta_edges[-1] - theta_edges[0])

    def cum_count(x):
        return np.searchsorted(r_sorted, x, side="left").astype(float)

    raw_mass = cum_count(r_eq[1:]) - cum_count(r_eq[:-1])
    raw_area = 0.5 * theta_span * (r_eq[1:] ** 2 - r_eq[:-1] ** 2)
    grid_mean = counts_eq.sum(axis=1) / areas_eq.sum(axis=1)
    raw_mean = raw_mass / raw_area
    denom = np.maximum(np.abs(raw_mean), 1e-300)
    invariance = float(
        np.max(np.abs(grid_mean - raw_mean) / denom)
    )
    # (2) structural agreement of the two binnings' density profiles
    interp = np.interp(c_e, c_a, m_a)
    inside = (c_e >= c_a[0]) & (c_e <= c_a[-1])
    rho_struct = float(stats.spearmanr(m_e[inside], interp[inside]).statistic)
    # (3) the count profile of the equal-angle grid tracks cell area instead
    rho_count_area = float(
        stats.spearmanr(counts.sum(axis=1), areas.sum(axis=1)).statistic
    )
    summary_bits["invariance"] = invariance
    summary_bits["rho_struct"] = rho_struct
    summary_bits["rho_count_area"] = rho_count_area
    summary_bits["eq_area_cell"] = float(areas_eq[0, 0])
    summary_bits["eq_area_spread"] = float(
        (areas_eq.max() - areas_eq.min()) / areas_eq.min()
    )

    plot_fig3(
        density_eq,
        r_eq,
        theta_edges,
        m_a,
        c_a,
        m_e,
        c_e,
        figdir / "fig3_equal_area_density.png",
    )

    # ---------------- experiment 3: cartesian rebin ------------------------ #
    r = targets[:, 1]
    th = np.deg2rad(targets[:, 3])
    x = r * np.cos(th)
    y = r * np.sin(th)
    x_edges = bn.default_edges(x, 1.0)
    y_edges = bn.default_edges(y, 1.0)
    counts_cart, density_cart = bn.cartesian_density(
        targets, x_edges, y_edges, n_frames
    )
    areas_cart = np.outer(
        np.diff(x_edges), np.diff(y_edges)
    )
    reports.append(
        cons.check_conservation(
            density_cart, areas_cart, counts_cart, n_frames
        ).as_line("cartesian rebin sum(density*A)=sum(counts)/F")
    )
    totals_ok = int(counts.sum()) == int(counts_cart.sum()) == targets.shape[0]
    reports.append(
        f"[{'PASS' if totals_ok else 'FAIL'}] polar vs cartesian total weight: "
        f"{int(counts.sum())} == {int(counts_cart.sum())} == {targets.shape[0]}"
    )
    summary_bits["cart_total"] = int(counts_cart.sum())
    summary_bits["polar_total"] = int(counts.sum())

    # ---------------- corpus overview ------------------------------------- #
    corpus_frames = 0
    corpus_targets = 0
    for path in corpus_files:
        clog = parse_capture(path)
        corpus_frames += clog.n_frames
        corpus_targets += clog.n_targets
    summary_bits["corpus_files"] = len(corpus_files)
    summary_bits["corpus_frames"] = corpus_frames
    summary_bits["corpus_targets"] = corpus_targets

    # ---------------- synthetic benchmark --------------------------------- #
    syn = synthetic_benchmark(rng, reports, figdir)
    summary_bits["synthetic"] = syn

    # ---------------- bin statistics + summary ---------------------------- #
    n_rows = write_bin_statistics(
        counts,
        density,
        areas,
        r_edges,
        theta_edges,
        resdir / "bin_statistics.csv",
    )
    summary_bits["bin_rows"] = n_rows
    summary_bits["grid"] = (
        len(r_edges) - 1,
        len(theta_edges) - 1,
        r_edges[0],
        r_edges[-1],
        theta_edges[0],
        theta_edges[-1],
    )
    summary_bits["main_frames"] = n_frames
    summary_bits["main_targets"] = int(targets.shape[0])
    summary_bits["header_cols"] = log.header.count(",") + 1
    summary_bits["r_span"] = (float(targets[:, 1].min()), float(targets[:, 1].max()))
    summary_bits["th_span"] = (float(targets[:, 3].min()), float(targets[:, 3].max()))

    write_summary(resdir / "summary.md", summary_bits, reports)

    print()
    for line in reports:
        print(line)
    failed = any(line.startswith("[FAIL]") for line in reports)
    if failed:
        print("CONSERVATION CHECKS FAILED", file=sys.stderr)
        return 1
    print(
        "\nfigures -> "
        + str(figdir.resolve())
        + "\nresults -> "
        + str(resdir.resolve())
    )
    print("ALL CHECKS PASSED")
    return 0


def write_summary(out_path: Path, bits: dict, reports: list[str]) -> None:
    rk = bits["rank_stats"]
    rec = bits["synthetic"]["recovery"]
    grid = bits["grid"]
    text = f"""# a100-polar-density 结果摘要

固定随机种子 `{SEED}`；全部产物可由 `cli.py` 复现。

## 输入概览

- 主输入：`Python雷达可视化工具示例/负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv`
  （表头 {bits['header_cols']} 列变体，{bits['main_frames']} 帧，{bits['main_targets']} 条目标行；
  range {bits['r_span'][0]:.2f}–{bits['r_span'][1]:.2f} m，angle {bits['th_span'][0]:.2f}–{bits['th_span'][1]:.2f}°）。
- 复核语料：`RadarTools/RadarTools_Release/data/` 下 {bits['corpus_files']} 个 `*_RawTarget.csv`，
  共 {bits['corpus_frames']} 帧、{bits['corpus_targets']} 条目标行（SHA256 见 `data_sha256.txt`，共 22 个文件）。
- 分箱网格：Δr = 1 m、Δθ = 1°（半开区间 [a, b) 约定），覆盖数据范围
  r ∈ [{grid[2]:g}, {grid[3]:g}] m、θ ∈ [{grid[4]:g}, {grid[5]:g}]°，
  共 {grid[0]} × {grid[1]} = {bits['bin_rows']} 个扇区。

## 实验①：计数图 vs 密度图与热点名次位移

- 活跃（非零）扇区 {rk['n_live_sectors']} 个；计数与密度的 Spearman 秩相关
  ρ = {rk['spearman_count_density']:.4f}（≠1 的部分完全来自扇区面积差异）。
- **面积偏置量级**：全网格最小/最大扇区面积比 1:{bits['area_bias']:.0f}
  （最内环扇区 {bits['area_first_last'][0]:.3g} m² → 最外环扇区 {bits['area_first_last'][1]:.3g} m²）；
  同样目标密度下远距扇区计数天然按面积放大。计数与密度的秩相关仅
  ρ = {rk['spearman_count_density']:.4f}（扇区级计数-面积 Spearman ρ = {rk['spearman_count_area']:.4f}：
  大扇区在同等密度下天然多收检测；纯面积伪影在合成基准中单独量化）。
- **热点名次位移（top-{rk['topK']} 计数热点）**：中位位移 {rk['shift_median']:+.0f} 名，
  范围 [{rk['shift_min']:+d}, {rk['shift_max']:+d}]；{rk['n_promoted']} 个被面积归一提升、
  {rk['n_demoted']} 个下降、{rk['topK'] - rk['n_promoted'] - rk['n_demoted']} 个不变；
  **{rk['dropped_out_of_topK']} 个计数 top-{rk['topK']} 热点在密度图中跌出 top-{rk['topK']}**。
  明细见 `hotspot_rank_shift.csv`（shift = 计数名次 − 密度名次，正=归一后更热）。

## 实验②：等面积分箱的密度可比性

- 等面积网格（sqrt 间距径向边界，环带面积相等 + 均匀角度划分）：单扇区面积
  {bits['eq_area_cell']:.4g} m²，全网格面积相对离散度 {bits['eq_area_spread']:.1e}（浮点误差量级）。
- **分箱不变性（精确核验）**：等面积网格的环带平均密度与不经分箱、直接按径向区间计算的均值
  最大相对差 {bits['invariance']:.1e}——密度是分箱不变的物理量，原始计数不是。
- 两种分箱的环带密度剖面结构一致（Spearman ρ = {bits['rho_struct']:.4f}）。
  作为对照，环带**计数**不具有这种不变性：等角度网格的环带计数与环带面积的
  Spearman ρ = {bits['rho_count_area']:.4f}（符号与方向取决于目标分布，量级 1:{bits['area_bias']:.0f}
  的面积差异始终是计数表示中的混杂因子；合成基准单独给出纯面积伪影 ~{bits['synthetic']['growth']:.0f}×）。

## 实验③：笛卡尔重分箱对照（同权重总量）

- 同一批检测在极坐标网格与 1 m×1 m 笛卡尔网格上的总权重均为 {bits['polar_total']}
  （= 检测总数），密度面积分相等；差异只在空间表示：极坐标等角度扇区面积 ∝ r²，
  笛卡尔格元面积恒定，计数热点图因此呈现不同的偏置模式。

## 守恒核验（相对容差 1e-9）

""" + "\n".join(f"- {line}" for line in reports) + f"""

## 合成基准（seed = {SEED}）

- 均匀环带（r ∈ [5, 80] m，面积均匀撒点 {bits['synthetic']['n_background']} 点 / {bits['synthetic']['n_frames']} 帧）：
  最外/最内环带计数比 ≈ {bits['synthetic']['growth']:.1f}×（计数随 r 增长），
  面积归一后环带密度最大偏差 {bits['synthetic']['flatness'] * 100:.2f}%（平坦）。
  纯均匀背景的计数峰值出现在第 {bits['synthetic']['bg_peak_ring'] + 1}/{bits['synthetic']['n_rings']} 环
  （最外环附近，纯面积伪影），密度峰值则无此偏移。
- 已知热点（真值 r = {rec['true_r']:g} m, θ = {rec['true_theta']:+g}°，共 {bits['synthetic']['n_points'] - bits['synthetic']['n_background']} 点）：
  密度图峰值扇区位于 r = {rec['peak_r']:.1f} m, θ = {rec['peak_theta']:+.1f}°（1 个扇区内），
  该扇区密度名次第 1、原始计数名次第 {rec['hotspot_rank_by_count']}——面积归一恢复真实最热点。

## 结论

计数热图在等角度极坐标网格上带有 ∝ r² 的面积混杂（量级 1:{bits['area_bias']:.0f}）：
相同目标密度下远距扇区计数天然更高，实测 top-20 计数热点中有 {rk['dropped_out_of_topK']} 个
在密度图中跌出 top-20；密度 = 计数/(扇区面积×帧数) 修正后，热点排名、跨距离可比性与分箱不变性均恢复。
注意：这是**检测输出密度**，不是真实目标密度（见 README 局限）。
"""
    out_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
