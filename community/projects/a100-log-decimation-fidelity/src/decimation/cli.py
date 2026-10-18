"""Reproducible experiment driver for the log-decimation fidelity study.

Everything is deterministic: the only randomness is time-stratified sampling,
whose seeds are derived from the fixed master seed (default 20261003).

Run from the project directory::

    PYTHONPATH=src python3 src/decimation/cli.py [--options]

Outputs (all regenerable by re-running):
    results/data_sha256.txt
    results/fidelity_curves.csv      per-(method, k, phase/replicate) metrics
    results/worst_phase.csv          per-k worst / median / best phase table
    results/short_response_retention.csv
    results/summary.md               Chinese summary with the key numbers
    figures/*.png                    5 figures (English labels, units)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

if __package__ in (None, ""):  # executed as a plain script
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from decimation.fidelity import (
        CellGrid,
        analytic_missing_phases,
        frame_time_weights,
        merged_time_weights,
        short_response_metrics,
        synthetic_irregular_session,
        synthetic_pulse_session,
        tv_l1,
        l2_distance,
        count_stat_errors,
    )
    from decimation.parser import parse_target_log
    from decimation.samplers import (
        byte_budget_k,
        stratified_keep_indices,
        uniform_keep_indices,
    )
else:
    from .fidelity import (
        CellGrid,
        analytic_missing_phases,
        frame_time_weights,
        merged_time_weights,
        short_response_metrics,
        synthetic_irregular_session,
        synthetic_pulse_session,
        tv_l1,
        l2_distance,
        count_stat_errors,
    )
    from .parser import parse_target_log
    from .samplers import (
        byte_budget_k,
        stratified_keep_indices,
        uniform_keep_indices,
    )

#: validated categorical palette (dataviz reference, light mode)
C_BLUE = "#2a78d6"      # slot 1 - equal-interval decimation
C_ORANGE = "#eb6834"    # slot 2 - time-stratified sampling
C_BLUE_D = "#104281"    # ordinal blue 650 - "worst"
C_BLUE_M = "#2a78d6"    # ordinal blue 450 - "median"
C_BLUE_L = "#86b6ef"    # ordinal blue 250 - "best"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

MASTER_SEED = 20261003

_HERE = Path(__file__).resolve()
PROJECT_DIR_DEFAULT = _HERE.parents[2]
REPO_ROOT_DEFAULT = _HERE.parents[5]
DATA1_DEFAULT = (
    REPO_ROOT_DEFAULT
    / "RadarTools"
    / "RadarTools_Release"
    / "data"
    / "2026_05_20_16_24_32_Tatget_CTLRR-300-V2_1_RawTarget.csv"
)
DATA2_DEFAULT = (
    REPO_ROOT_DEFAULT
    / "Python雷达可视化工具示例"
    / "负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv"
)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": AXIS,
            "axes.labelcolor": INK2,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "axes.grid": True,
            "axes.grid.axis": "y",
            "axes.axisbelow": True,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.0,
            "lines.markersize": 6,
            "legend.frameon": False,
            "legend.fontsize": 9,
            "font.family": "DejaVu Sans",
        }
    )


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def byte_model(log) -> dict:
    """Average-bytes-per-frame model for the byte-budget view."""
    with open(log.source, "rb") as fh:
        header_bytes = len(fh.readline())
    per_frame = (log.file_size_bytes - header_bytes) / max(1, log.n_frames)
    return {
        "header_bytes": header_bytes,
        "per_frame_bytes": per_frame,
        "total_mb": log.file_size_bytes / 1e6,
    }


def evaluate_sample(
    frames, times_ms, keep_idx, grid, full_density, full_exposure_s, T
):
    """All fidelity metrics for one retained-frame set."""
    keep = np.asarray(keep_idx, dtype=np.int64)
    w_merged = merged_time_weights(times_ms, keep)
    kept_frames = [frames[j] for j in keep]
    density, _ = grid.histogram(kept_frames, w_merged)
    sampled_exposure = grid.exposure(kept_frames, w_merged / 1000.0)
    sr = short_response_metrics(full_exposure_s, sampled_exposure, T)
    ce = count_stat_errors(frames, keep)
    return {
        "m": int(len(keep)),
        "l1": tv_l1(full_density, density),
        "l2": l2_distance(full_density, density),
        "retention": sr["retention_rate"],
        "n_missed_short": sr["n_missed"],
        "n_short": sr["n_short"],
        **ce,
    }


def run_experiments(args) -> int:
    proj = Path(args.project_dir)
    results_dir = proj / "results"
    figures_dir = proj / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    _style()

    # ----------------------------------------------------------- parse data
    log1 = parse_target_log(args.data)
    log2 = parse_target_log(args.data2)
    frames, times = log1.frames, log1.times_ms
    n = log1.n_frames
    print(
        f"[data] primary:   {log1.n_frames} frames, {log1.duration_s:.2f} s, "
        f"{int(log1.counts.sum())} target rows, header cols {log1.header_columns}"
    )
    print(
        f"[data] secondary: {log2.n_frames} frames, {log2.duration_s:.2f} s, "
        f"{int(log2.counts.sum())} target rows, header cols {log2.header_columns}"
    )

    with open(results_dir / "data_sha256.txt", "w", encoding="utf-8") as fh:
        for tag, log in (("primary", log1), ("secondary", log2)):
            fh.write(
                f"{tag}: {log.source}\\n"
                f"  bytes={log.file_size_bytes} frames={log.n_frames} "
                f"duration_s={log.duration_s:.3f} "
                f"header_columns={log.header_columns}\\n"
                f"  sha256={sha256_of(log.source)}\\n"
            )

    # ------------------------------------------------------- reference state
    r_max = max(float(fr.targets[:, 0].max()) for fr in frames if fr.n_targets)
    grid = CellGrid(dr=args.dr, dtheta=args.dtheta, r_max=math.ceil(r_max))
    w_full = frame_time_weights(times)
    full_density, _ = grid.histogram(frames, w_full)
    full_exposure_s = grid.exposure(frames, w_full / 1000.0)
    T = args.short_t
    bm1 = byte_model(log1)
    print(
        f"[grid] dr={args.dr} m, dtheta={args.dtheta} deg, "
        f"r_max={math.ceil(r_max)} m, shape={grid.shape}; T={T} s; "
        f"short cells={int(((full_exposure_s > 0) & (full_exposure_s < T)).sum())}"
    )

    # frame-parity diagnostic: the session content itself alternates by frame
    # index parity, which even-k equal-interval decimation aliases onto
    dens_even, _ = grid.histogram(frames[0::2], w_full[0::2])
    dens_odd, _ = grid.histogram(frames[1::2], w_full[1::2])
    parity_tv = tv_l1(dens_even, dens_odd)
    print(f"[diag] TV(even-index frames, odd-index frames) = {parity_tv:.4f}")

    # --------------------------------------------------- experiment 1 + 2
    ks = list(range(args.kmin, args.kmax + 1))
    rows = []
    per_k_uniform = {}

    identity = evaluate_sample(frames, times, np.arange(n), grid,
                               full_density, full_exposure_s, T)
    rows.append(
        {
            "method": "identity",
            "k": 1,
            "draw_id": 0,
            "m": n,
            "compression_ratio": 1.0,
            "l1": identity["l1"],
            "l2": identity["l2"],
            "short_retention": identity["retention"],
            "count_mean_rel_err": identity["mean_rel_err"],
            "count_p50_abs_err": identity["p50_abs_err"],
            "count_p90_abs_err": identity["p90_abs_err"],
            "est_kept_bytes": log1.file_size_bytes,
        }
    )

    for k in ks:
        per_k_uniform[k] = []
        for phi in range(k):
            keep = uniform_keep_indices(n, k, phi)
            ev = evaluate_sample(
                frames, times, keep, grid, full_density, full_exposure_s, T
            )
            per_k_uniform[k].append((phi, ev))
            rows.append(
                {
                    "method": "uniform",
                    "k": k,
                    "draw_id": phi,
                    "m": ev["m"],
                    "compression_ratio": n / ev["m"],
                    "l1": ev["l1"],
                    "l2": ev["l2"],
                    "short_retention": ev["retention"],
                    "count_mean_rel_err": ev["mean_rel_err"],
                    "count_p50_abs_err": ev["p50_abs_err"],
                    "count_p90_abs_err": ev["p90_abs_err"],
                    "est_kept_bytes": int(
                        bm1["header_bytes"] + bm1["per_frame_bytes"] * ev["m"]
                    ),
                }
            )

    master = np.random.default_rng(args.seed)
    child_seeds = master.integers(
        0, 2**63 - 1, size=(len(ks), args.replicates)
    )
    per_k_strat = {}
    for i, k in enumerate(ks):
        per_k_strat[k] = []
        for rep in range(args.replicates):
            keep = stratified_keep_indices(
                times, n // k, seed=int(child_seeds[i, rep])
            )
            ev = evaluate_sample(
                frames, times, keep, grid, full_density, full_exposure_s, T
            )
            per_k_strat[k].append((rep, ev))
            rows.append(
                {
                    "method": "stratified",
                    "k": k,
                    "draw_id": rep,
                    "m": ev["m"],
                    "compression_ratio": n / ev["m"],
                    "l1": ev["l1"],
                    "l2": ev["l2"],
                    "short_retention": ev["retention"],
                    "count_mean_rel_err": ev["mean_rel_err"],
                    "count_p50_abs_err": ev["p50_abs_err"],
                    "count_p90_abs_err": ev["p90_abs_err"],
                    "est_kept_bytes": int(
                        bm1["header_bytes"] + bm1["per_frame_bytes"] * ev["m"]
                    ),
                }
            )

    fields = list(rows[0].keys())
    with open(results_dir / "fidelity_curves.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"[exp1] fidelity_curves.csv: {len(rows)} rows")

    # --------------------------------------------------- worst-phase table
    wp_rows = []
    for k in ks:
        items = sorted(per_k_uniform[k], key=lambda t: t[1]["l1"])
        phi_best, ev_best = items[0]
        phi_med, ev_med = items[len(items) // 2]
        phi_worst, ev_worst = items[-1]
        wp_rows.append(
            {
                "k": k,
                "worst_phase": phi_worst,
                "l1_worst": ev_worst["l1"],
                "retention_worst": ev_worst["retention"],
                "missed_short_worst": ev_worst["n_missed_short"],
                "median_phase": phi_med,
                "l1_median": ev_med["l1"],
                "retention_median": ev_med["retention"],
                "best_phase": phi_best,
                "l1_best": ev_best["l1"],
                "l1_gap_worst_over_best": ev_worst["l1"] / ev_best["l1"]
                if ev_best["l1"] > 0
                else float("inf"),
                "m_mean": float(np.mean([ev["m"] for _, ev in per_k_uniform[k]])),
            }
        )
    with open(results_dir / "worst_phase.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(wp_rows[0].keys()))
        w.writeheader()
        w.writerows(wp_rows)

    # ------------------------------------- short-response retention buckets
    bucket_edges = np.arange(0.0, T + 1e-9, T / 5.0)
    sel_ks = [k for k in (2, 4, 8, 16) if args.kmin <= k <= args.kmax] or [ks[0]]
    sr_rows = []
    full_flat = full_exposure_s.ravel()
    for k in sel_ks:
        items = sorted(per_k_uniform[k], key=lambda t: t[1]["l1"])
        phi_med, _ = items[len(items) // 2]
        keep = uniform_keep_indices(n, k, phi_med)
        w_merged = merged_time_weights(times, keep)
        samp_exp = grid.exposure([frames[j] for j in keep], w_merged / 1000.0).ravel()
        for b in range(len(bucket_edges) - 1):
            lo, hi = bucket_edges[b], bucket_edges[b + 1]
            # short cells only: strictly positive exposure, capped at T
            mask = (full_flat > lo) & (full_flat <= hi)
            if mask.sum() == 0:
                continue
            sr_rows.append(
                {
                    "method": f"uniform_k{k}_medianphase{phi_med}",
                    "k": k,
                    "bucket_lo_s": round(lo, 3),
                    "bucket_hi_s": round(hi, 3),
                    "n_cells": int(mask.sum()),
                    "retention_rate": float((samp_exp[mask] > 0).mean()),
                }
            )
        rep0, _ = per_k_strat[k][0]
        keep_s = stratified_keep_indices(times, n // k, seed=int(child_seeds[ks.index(k), 0]))
        w_m_s = merged_time_weights(times, keep_s)
        samp_exp_s = grid.exposure(
            [frames[j] for j in keep_s], w_m_s / 1000.0
        ).ravel()
        for b in range(len(bucket_edges) - 1):
            lo, hi = bucket_edges[b], bucket_edges[b + 1]
            # short cells only: strictly positive exposure, capped at T
            mask = (full_flat > lo) & (full_flat <= hi)
            if mask.sum() == 0:
                continue
            sr_rows.append(
                {
                    "method": f"stratified_k{k}_rep{rep0}",
                    "k": k,
                    "bucket_lo_s": round(lo, 3),
                    "bucket_hi_s": round(hi, 3),
                    "n_cells": int(mask.sum()),
                    "retention_rate": float((samp_exp_s[mask] > 0).mean()),
                }
            )
    with open(results_dir / "short_response_retention.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(sr_rows[0].keys()))
        w.writeheader()
        w.writerows(sr_rows)

    # ============================================================ figures
    _fig_curves(per_k_uniform, per_k_strat, ks, n, figures_dir)
    _fig_worst_phase(wp_rows, ks, figures_dir)
    _fig_miss_example(frames, times, grid, full_exposure_s, T, figures_dir)
    _fig_byte_budget(log1, log2, n, figures_dir)
    _fig_synthetic(args, figures_dir)

    # ============================================================ summary
    _write_summary(
        results_dir, log1, log2, per_k_uniform, per_k_strat, wp_rows, sr_rows,
        ks, n, T, args, bm1, parity_tv,
    )
    print("[done] results/ and figures/ refreshed")
    return 0


# ------------------------------------------------------------------ figure 1
def _fig_curves(per_k_uniform, per_k_strat, ks, n, figures_dir):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True,
        gridspec_kw={"hspace": 0.28},
    )

    def series(per_k):
        ratio, l1_m, l1_lo, l1_hi, ret_m, ret_lo, ret_hi = [], [], [], [], [], [], []
        for k in ks:
            evs = [ev for _, ev in per_k[k]]
            ratios = np.array([n / ev["m"] for ev in evs])
            l1s = np.array([ev["l1"] for ev in evs])
            rets = np.array([ev["retention"] for ev in evs])
            ratio.append(ratios.mean())
            l1_m.append(l1s.mean()); l1_lo.append(l1s.min()); l1_hi.append(l1s.max())
            ret_m.append(rets.mean()); ret_lo.append(rets.min()); ret_hi.append(rets.max())
        return (np.array(ratio), np.array(l1_m), np.array(l1_lo),
                np.array(l1_hi), np.array(ret_m), np.array(ret_lo),
                np.array(ret_hi))

    ru = series(per_k_uniform)
    rs = series(per_k_strat)
    common = dict(alpha=0.18, linewidth=0)

    ax1.plot(ru[0], ru[1], color=C_BLUE, marker="o", label="Equal-interval (mean over phases)")
    ax1.fill_between(ru[0], ru[2], ru[3], color=C_BLUE, **common)
    ax1.plot(rs[0], rs[1], color=C_ORANGE, marker="s", label="Time-stratified (mean over 10 draws)")
    ax1.fill_between(rs[0], rs[2], rs[3], color=C_ORANGE, **common)
    ax1.plot([1.0], [0.0], marker="D", color=INK2, linestyle="none", label="k = 1 (identity)")
    ax1.set_ylabel("L1 occupancy distance (total variation)")
    ax1.set_title("Time-weighted occupancy error vs compression ratio")
    ax1.set_ylim(bottom=0)

    ax2.plot(ru[0], ru[4], color=C_BLUE, marker="o")
    ax2.fill_between(ru[0], ru[5], ru[6], color=C_BLUE, **common)
    ax2.plot(rs[0], rs[4], color=C_ORANGE, marker="s")
    ax2.fill_between(rs[0], rs[5], rs[6], color=C_ORANGE, **common)
    ax2.plot([1.0], [1.0], marker="D", color=INK2, linestyle="none")
    ax2.set_ylabel(f"Short-lived cell retention (T = 1 s)")
    ax2.set_xlabel("Compression ratio (n frames kept / m frames retained)")
    ax2.set_title("Short-lived cell retention vs compression ratio")
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(left=0.8)
    ax2.set_xticks([1, 2, 4, 6, 8, 10, 12, 14, 16])

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8)
    fig.subplots_adjust(bottom=0.17, hspace=0.32)

    fig.savefig(figures_dir / "fig1_fidelity_curves.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def _fig_worst_phase(wp_rows, ks, figures_dir):
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(8.0, 6.4), sharex=True,
        gridspec_kw={"hspace": 0.28},
    )
    k_arr = np.array([r["k"] for r in wp_rows], dtype=float)

    ax1.plot(k_arr, [r["l1_worst"] for r in wp_rows], color=C_BLUE_D,
             marker="o", label="Worst phase")
    ax1.plot(k_arr, [r["l1_median"] for r in wp_rows], color=C_BLUE_M,
             marker="s", label="Median phase")
    ax1.plot(k_arr, [r["l1_best"] for r in wp_rows], color=C_BLUE_L,
             marker="^", label="Best phase")
    for r in wp_rows:
        ax1.annotate(
            f"phi={r['worst_phase']}",
            (r["k"], r["l1_worst"]),
            textcoords="offset points", xytext=(0, 8),
            ha="center", fontsize=7, color=INK2,
            bbox=dict(boxstyle="round,pad=0.15", facecolor=SURFACE,
                      edgecolor="none"),
        )
    ax1.set_ylabel("L1 occupancy distance (total variation)")
    ax1.set_title("Equal-interval decimation: worst / median / best phase, by k")
    ax1.set_ylim(0, max(r["l1_worst"] for r in wp_rows) * 1.30)
    ax1.legend(loc="lower left", bbox_to_anchor=(0.0, 1.16), ncol=3,
               fontsize=8)

    width = 0.38
    offset = 0.24
    ax2.bar(k_arr - offset, [r["retention_worst"] for r in wp_rows],
            width=width, color=C_BLUE, edgecolor=SURFACE, linewidth=2.0,
            label="Worst-phase retention")
    ax2.bar(k_arr + offset, [r["retention_median"] for r in wp_rows],
            width=width, color=C_BLUE_L, edgecolor=SURFACE, linewidth=2.0,
            label="Median-phase retention")
    ax2.set_xlabel("Decimation step k (frames)")
    ax2.set_ylabel("Short-lived cell retention (T = 1 s)")
    ax2.set_title("Short-lived cell retention: worst vs median phase")
    ax2.set_ylim(0, 1.05)
    ax2.set_xticks(k_arr.astype(int))
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2,
               fontsize=8)
    fig.subplots_adjust(top=0.88, bottom=0.17, hspace=0.38)

    fig.savefig(figures_dir / "fig2_worst_phase.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def _fig_miss_example(frames, times, grid, full_exposure_s, T, figures_dir,
                      max_k_search=10):
    short_rc = np.argwhere((full_exposure_s > 0) & (full_exposure_s < T))
    order = np.argsort(full_exposure_s[short_rc[:, 0], short_rc[:, 1]])
    short_rc = short_rc[order]

    chosen = None
    for max_span_s in (2.0, 6.0, 20.0):
        for rc in short_rc:
            active = grid.cell_active_frames(frames, (int(rc[0]), int(rc[1])))
            span_s = (times[active[-1]] - times[active[0]]) / 1000.0
            if len(active) < 2 or span_s > max_span_s:
                continue
            active_set = set(int(a) for a in active)
            for k in range(2, max_k_search + 1):
                missing, hitting = [], []
                for phi in range(k):
                    keep = set(int(x) for x in uniform_keep_indices(len(frames), k, phi))
                    (missing if not (keep & active_set) else hitting).append(phi)
                if missing and hitting:
                    chosen = (rc, active, k, missing[0], hitting[0], span_s)
                    break
            if chosen:
                break
        if chosen:
            break
    if chosen is None:
        print("[fig3] no fully-missable short cell found; skipping figure")
        return

    rc, active, k, phi_miss, phi_hit, span_s = chosen
    t0 = times[0]
    rel = (times - t0) / 1000.0
    act_rel = rel[active]
    lo, hi = act_rel[0], act_rel[-1]
    pad = max(1.5, 1.5 * span_s)
    xlo, xhi = max(0.0, lo - pad), min(rel[-1], hi + pad)
    in_win = (rel >= xlo) & (rel <= xhi)

    fig, axes = plt.subplots(
        2, 1, figsize=(8.0, 4.8), sharex=True,
        gridspec_kw={"hspace": 0.3},
    )
    for ax, phi, missed in (
        (axes[0], phi_miss, True),
        (axes[1], phi_hit, False),
    ):
        keep = uniform_keep_indices(len(frames), k, phi)
        ax.axvspan(lo, hi, color=C_ORANGE, alpha=0.12, linewidth=0)
        ax.eventplot(
            rel[in_win], lineoffsets=1.0, linelengths=0.25,
            color=AXIS, linewidths=0.8,
        )
        ax.eventplot(
            rel[active], lineoffsets=2.0, linelengths=0.35,
            color=C_ORANGE, linewidths=1.6,
        )
        ax.eventplot(
            rel[keep], lineoffsets=3.0, linelengths=0.35,
            color=C_BLUE, linewidths=1.6,
        )
        for y, lab, col in ((1.0, "all frames", MUTED),
                            (2.0, "short cell active", C_ORANGE),
                            (3.0, "retained frames", C_BLUE)):
            ax.text(xhi + 0.01 * (xhi - xlo + 1e-9) + 0.05, y, lab,
                    va="center", fontsize=8, color=col)
        status = "MISSED" if missed else "retained"
        ax.set_title(
            f"k = {k}, phase = {phi}: short response {status} "
            f"(active span {span_s:.2f} s)",
            fontsize=10,
            color=INK,
        )
        ax.set_ylim(0.4, 3.6)
        ax.set_yticks([])
        ax.grid(False)
        for s in ("left",):
            ax.spines[s].set_visible(False)
    axes[1].set_xlabel("Time since session start (s)")
    axes[1].set_xlim(xlo, xhi)
    fig.savefig(figures_dir / "fig3_short_response_miss.png", dpi=150,
                bbox_inches="tight")
    plt.close(fig)
    print(
        f"[fig3] cell (r={rc[0]},c={rc[1]}), {len(active)} active frames, "
        f"k={k}: phase {phi_miss} misses, phase {phi_hit} retains"
    )


# ------------------------------------------------------------------ figure 4
def _fig_byte_budget(log1, log2, n, figures_dir, c_max=16):
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    ratios = np.arange(1, c_max + 1, dtype=float)
    for log, color, label in (
        (log1, C_BLUE, f"Primary session 2026-05-20 ({log1.n_frames} frames)"),
        (log2, C_ORANGE, f"Secondary session -30 deg 2026-03-10 ({log2.n_frames} frames)"),
    ):
        bm = byte_model(log)
        mb = []
        for c in ratios:
            k = byte_budget_k(c)
            m = log.n_frames // k
            mb.append((bm["header_bytes"] + bm["per_frame_bytes"] * m) / 1e6)
        ax.plot(ratios, mb, color=color, marker="o", label=label)
        ax.axhline(bm["total_mb"], color=color, linestyle=":", linewidth=1.2,
                   alpha=0.7)
        ax.annotate(
            f"{bm['total_mb']:.1f} MB",
            (1.0, bm["total_mb"]),
            textcoords="offset points", xytext=(6, 6),
            fontsize=8, color=color,
        )
    ax.set_xlabel("Target compression ratio C  (keep-every-k with k = ceil(C))")
    ax.set_ylabel("Estimated retained log size (MB)")
    ax.set_title("Byte-budget view: log size after keep-every-k decimation")
    ax.set_xticks(ratios.astype(int))
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 1.14), ncol=2,
              fontsize=8)
    fig.subplots_adjust(top=0.80)
    fig.savefig(figures_dir / "fig4_byte_budget.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def _fig_synthetic(args, figures_dir):
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(10.0, 4.4), gridspec_kw={"wspace": 0.3},
    )

    # (a) pulse train: analytic lines with direct end labels, enumerated markers
    ks_small = np.arange(2, 17)
    for L, color, marker in ((1, C_BLUE_D, "o"), (3, C_BLUE_M, "s"),
                             (5, C_BLUE_L, "^")):
        analytic = np.array(
            [len(analytic_missing_phases(40, L, k)) / k for k in ks_small]
        )
        simulated = []
        frames_p, times_p = synthetic_pulse_session(
            pulses=((40, L, (6.25, 15.0)),)
        )
        n_p = len(frames_p)
        for k in ks_small:
            misses = 0
            for phi in range(k):
                keep = set(int(x) for x in uniform_keep_indices(n_p, k, phi))
                hits = any(
                    40 <= j < 40 + L for j in keep
                )
                misses += 0 if hits else 1
            simulated.append(misses / k)
        ax1.plot(ks_small, analytic, color=color, linewidth=2)
        ax1.plot(ks_small, simulated, color=color, marker=marker,
                 linestyle="none", markersize=6)
        ax1.text(16.4, analytic[-1], f"L = {L}", fontsize=9, color=color,
                 va="center")
    ax1.set_xlabel("Decimation step k (frames)")
    ax1.set_ylabel("Fraction of phases missing the whole response")
    ax1.set_title("(a) Pulse train: analytic lines, markers = phase enumeration")
    ax1.set_ylim(0, 1.05)
    ax1.set_xlim(1.5, 19.0)

    # (b) irregular dt: time weighting vs count weighting
    frames_i, times_i = synthetic_irregular_session()
    grid = CellGrid(dr=0.5, dtheta=2.0, r_max=25.0)
    w_t = frame_time_weights(times_i)
    dens_t, _ = grid.histogram(frames_i, w_t)
    dens_c, _ = grid.histogram(frames_i, np.ones(len(frames_i)))
    cells = {"cell A (sparse half)": (6.0, 10.0), "cell B (dense half)": (15.0, -20.0)}
    xs = np.arange(2)
    width = 0.38
    time_vals, count_vals = [], []
    for (r, th) in cells.values():
        ir = int((r - grid.r_edges[0]) / grid.dr)
        it = int((th - grid.theta_edges[0]) / grid.dtheta)
        time_vals.append(dens_t[ir, it])
        count_vals.append(dens_c[ir, it])
    ax2.bar(xs - width / 2, time_vals, width, color=C_BLUE,
            edgecolor=SURFACE, linewidth=1.5, label="Time-weighted")
    ax2.bar(xs + width / 2, count_vals, width, color=C_ORANGE,
            edgecolor=SURFACE, linewidth=1.5, label="Count-weighted")
    ax2.axhline(0.5, color=MUTED, linestyle=":", linewidth=1.2)
    ax2.text(0.5, 0.56, "0.5 (equal duration)", fontsize=8, color=MUTED,
             va="bottom", ha="center")
    for x, v in zip(xs - width / 2, time_vals):
        ax2.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    for x, v in zip(xs + width / 2, count_vals):
        ax2.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8, color=INK2)
    ax2.set_xticks(xs)
    ax2.set_xticklabels(list(cells.keys()), fontsize=9)
    ax2.set_ylabel("Normalized occupancy (sum = 1)")
    ax2.set_title("(b) Irregular frame gaps: time vs count weighting")
    ax2.set_ylim(0, 1.12)
    ax2.set_xlim(-0.55, 1.55)
    ax2.legend(loc="upper left")

    fig.savefig(figures_dir / "fig5_synthetic_validation.png", dpi=150)
    plt.close(fig)


# ------------------------------------------------------------------- summary
def _write_summary(results_dir, log1, log2, per_k_uniform, per_k_strat, wp_rows,
                   sr_rows, ks, n, T, args, bm1, parity_tv):
    def u(k):
        return [ev for _, ev in per_k_uniform[k]]

    def s(k):
        return [ev for _, ev in per_k_strat[k]]

    pick = [k for k in (2, 4, 8, 16) if k in ks]
    lines = []
    lines.append("# 结果摘要（由 cli.py 自动生成，种子 %d）" % args.seed)
    lines.append("")
    lines.append("## 数据")
    lines.append("")
    lines.append("| 输入 | 文件 | 帧数 | 时长 (s) | 目标行 |")
    lines.append("| --- | --- | --- | --- | --- |")
    lines.append(
        f"| 主输入 | `{log1.source.name}` | {log1.n_frames} | "
        f"{log1.duration_s:.2f} | {int(log1.counts.sum())} |"
    )
    lines.append(
        f"| 第二输入 | `{log2.source.name}` | {log2.n_frames} | "
        f"{log2.duration_s:.2f} | {int(log2.counts.sum())} |"
    )
    lines.append("")
    lines.append("完整 sha256 见 `data_sha256.txt`。")
    lines.append("")
    lines.append(
        f"**帧奇偶结构诊断：** 主输入中偶数帧与奇数帧的时间加权占据直方图"
        f" TV 距离 = {parity_tv:.4f}（会话内容本身按帧序奇偶交替，"
        f"偶数 k 的等间隔抽稀会锁定单一奇偶类，见下表锯齿）。"
    )
    lines.append("")
    lines.append("## 误差-压缩比（等间隔 vs 时间分层）")
    lines.append("")
    lines.append("| k | 压缩比 | L1 均值(相位) | L1 最差(相位) | 分层 L1 均值±std | 分层 L1 最差 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for k in pick:
        uevs, sevs = u(k), s(k)
        l1u = np.array([e["l1"] for e in uevs])
        l1s = np.array([e["l1"] for e in sevs])
        ratio = n / np.mean([e["m"] for e in uevs])
        lines.append(
            f"| {k} | {ratio:.2f} | {l1u.mean():.4f} | {l1u.max():.4f} | "
            f"{l1s.mean():.4f}±{l1s.std():.4f} | {l1s.max():.4f} |"
        )
    lines.append("")
    lines.append("## 最坏相位（等间隔抽样）")
    lines.append("")
    lines.append("| k | 最差相位 phi | L1 最差 | L1 中位 | L1 最好 | 最差相位短命保留 |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for r in wp_rows:
        if r["k"] in pick:
            lines.append(
                f"| {r['k']} | {r['worst_phase']} | {r['l1_worst']:.4f} | "
                f"{r['l1_median']:.4f} | {r['l1_best']:.4f} | "
                f"{r['retention_worst']:.3f} |"
            )
    lines.append("")
    lines.append("## 短命响应保留（按出现时长分桶，T=%.1f s）" % T)
    lines.append("")
    uniq_methods = sorted(set(r["method"] for r in sr_rows))
    for mth in uniq_methods:
        lines.append(f"- **{mth}**：" + "；".join(
            f"{r['bucket_lo_s']:.1f}-{r['bucket_hi_s']:.1f}s: "
            f"{r['retention_rate']:.2f} (n={r['n_cells']})"
            for r in sr_rows if r["method"] == mth
        ))
    lines.append("")
    lines.append("## 字节预算（按平均每帧字节数估算）")
    lines.append("")
    lines.append(
        f"- 主输入原始体积 {bm1['total_mb']:.2f} MB，平均每帧 "
        f"{bm1['per_frame_bytes']:.0f} B；k=8 约 "
        f"{(bm1['header_bytes'] + bm1['per_frame_bytes'] * (n // 8)) / 1e6:.2f} MB，"
        f"k=16 约 "
        f"{(bm1['header_bytes'] + bm1['per_frame_bytes'] * (n // 16)) / 1e6:.2f} MB。"
    )
    lines.append("")
    lines.append("## 复现")
    lines.append("")
    lines.append("```bash")
    lines.append("cd community/projects/a100-log-decimation-fidelity")
    lines.append("PYTHONPATH=src python3 src/decimation/cli.py")
    lines.append("```")
    lines.append("")
    with open(results_dir / "summary.md", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Log decimation fidelity experiments (deterministic, "
        f"master seed {MASTER_SEED})"
    )
    p.add_argument("--data", type=Path, default=DATA1_DEFAULT,
                   help="primary RawTarget CSV")
    p.add_argument("--data2", type=Path, default=DATA2_DEFAULT,
                   help="secondary RawTarget CSV")
    p.add_argument("--project-dir", type=Path, default=PROJECT_DIR_DEFAULT)
    p.add_argument("--seed", type=int, default=MASTER_SEED)
    p.add_argument("--short-t", type=float, default=1.0,
                   help="short-lived cell threshold in seconds [s]")
    p.add_argument("--dr", type=float, default=0.5, help="range bin [m]")
    p.add_argument("--dtheta", type=float, default=2.0, help="angle bin [deg]")
    p.add_argument("--kmin", type=int, default=2)
    p.add_argument("--kmax", type=int, default=16)
    p.add_argument("--replicates", type=int, default=10,
                   help="stratified draws per k")
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    return run_experiments(args)


if __name__ == "__main__":
    sys.exit(main())
