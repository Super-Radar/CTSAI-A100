"""One-command reproduction of the CA-CFAR false-alarm calibration study.

Runs every experiment with fixed seeds, writes ``results/*.csv`` and
``figures/*.png`` next to this project directory, prints the key numbers,
and stores a machine-readable ``results/summary.json``.

Examples
--------
    PYTHONPATH=src python -m cfar_calib.cli                 # full run
    PYTHONPATH=src python -m cfar_calib.cli --trials 500     # more trials
    PYTHONPATH=src python -m cfar_calib.cli --quick          # smoke run
    PYTHONPATH=src python -m cfar_calib.cli --skip-real-data
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from .calibration import (  # noqa: E402
    N_TRAIN,
    PFA_TARGETS,
    cfg_tag,
    clopper_pearson,
    collect_db_margins,
    collect_sample,
    count_above,
    make_rng,
    summarize,
)
from .cfar import (  # noqa: E402
    alpha_from_pfa_single_look,
    alpha_k_look,
    cfar_detect,
    kernel_count,
    pfa_k_look,
)
from .pipeline import (  # noqa: E402
    RadarConfig,
    global_noise_db,
    load_real_adc,
    range_doppler_power,
    sha256_file,
    synth_real_noise,
    to_db,
)

# validated colorblind-safe categorical palette, fixed slot order
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
          "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
RHOS = (0.0, 0.3, 0.7, 1.0)
KS = (1, 2, 3, 4)
K_DB_COARSE = (2.0, 3.0, 4.0, 5.0, 6.0, 8.0)


def _setup_matplotlib() -> None:
    plt.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": False,
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.8,
    })


def write_csv(path: Path, rows: list, fields: list) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore",
                                lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"  wrote {path.name} ({len(rows)} rows)")


# --------------------------------------------------------------------------

def nominal_row_flatten(row: dict, **extra) -> dict:
    out = dict(extra)
    out.update({k: v for k, v in row.items()})
    return out


def exp1_selfcheck(grid, cfg, rect_summary, fig_dir, res_dir):
    """1) implementation self-check: single channel, nominal vs empirical."""
    rows = []
    for window, s in (("hann", grid[(1, 0.0)]), ("rect", rect_summary)):
        for row in s.nominal_rows:
            r = nominal_row_flatten(row, window=window, k_channels=1, rho=0.0)
            r["factor_ci_lo"] = row["ci_lo_thresh"] / row["pfa_nominal"]
            r["factor_ci_hi"] = row["ci_hi_thresh"] / row["pfa_nominal"]
            rows.append(r)
    write_csv(res_dir / "exp1_selfcheck.csv", rows,
              ["window", "k_channels", "rho", "pfa_nominal", "alpha_nominal",
               "count_thresh", "n_cells_thresh", "pfa_emp_thresh", "ci_lo_thresh",
               "ci_hi_thresh", "within_ci_thresh", "factor_thresh",
               "count_full", "pfa_emp_full", "ci_lo_full", "ci_hi_full",
               "within_ci_full", "factor_full", "factor_ci_lo", "factor_ci_hi",
               "n_trials"])

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    styles = {("hann", "thresh"): dict(color=COLORS[0], ls="-", marker="o"),
              ("hann", "full"): dict(color=COLORS[0], ls="--", marker="o", mfc="none"),
              ("rect", "thresh"): dict(color=COLORS[2], ls="-", marker="s"),
              ("rect", "full"): dict(color=COLORS[2], ls="--", marker="s", mfc="none")}
    labels = {("hann", "thresh"): "official hann window, threshold rule",
              ("hann", "full"): "official hann window, full detector",
              ("rect", "thresh"): "rect window (control), threshold rule",
              ("rect", "full"): "rect window (control), full detector"}
    for window, s in (("hann", grid[(1, 0.0)]), ("rect", rect_summary)):
        for stat in ("thresh", "full"):
            x, y, lo, hi = [], [], [], []
            for row in s.nominal_rows:
                x.append(row["pfa_nominal"])
                y.append(row[f"factor_{stat}"])
                lo.append(row[f"ci_lo_{stat}"] / row["pfa_nominal"])
                hi.append(row[f"ci_hi_{stat}"] / row["pfa_nominal"])
            y, lo, hi = np.array(y), np.array(lo), np.array(hi)
            err = np.array([np.maximum(y - lo, 1e-3), np.maximum(hi - y, 1e-3)])
            ax.errorbar(x, y, yerr=err, capsize=2.5,
                        label=labels[(window, stat)], **styles[(window, stat)])
    ax.axhline(1.0, color="#555555", lw=1.0, ls=":")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("nominal Pfa (single-look iid formula)")
    ax.set_ylabel("empirical / nominal Pfa")
    s0 = grid[(1, 0.0)]
    ax.set_title(f"K=1 self-check through the chain "
                 f"({s0.n_trials} trials, {s0.n_cells:.3g} cells, N={N_TRAIN})")
    ax.legend(loc="best")
    fig.savefig(fig_dir / "fig1_selfcheck.png")
    plt.close(fig)
    key = {}
    for window, s in (("hann", grid[(1, 0.0)]), ("rect", rect_summary)):
        for row in s.nominal_rows:
            key[f"{window}_pfa{row['pfa_nominal']:g}_factor"] = row["factor_thresh"]
            key[f"{window}_pfa{row['pfa_nominal']:g}_within_ci"] = row["within_ci_thresh"]
            key[f"{window}_pfa{row['pfa_nominal']:g}_factor_full"] = row["factor_full"]
    return key


def exp2_k_effect(grid, cfg, fig_dir, res_dir):
    """2) independent-channel count effect at the single-look nominal alpha."""
    rows = []
    for k in KS:
        for row in grid[(k, 0.0)].nominal_rows:
            rows.append(nominal_row_flatten(row, k_channels=k, rho=0.0))
    write_csv(res_dir / "exp2_k_effect.csv", rows,
              ["k_channels", "rho", "pfa_nominal", "alpha_nominal", "count_thresh",
               "n_cells_thresh", "pfa_emp_thresh", "ci_lo_thresh", "ci_hi_thresh",
               "within_ci_thresh", "factor_thresh", "count_full", "pfa_emp_full",
               "ci_lo_full", "ci_hi_full", "within_ci_full", "factor_full", "n_trials"])

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3), sharey=True)
    label_dy = {1: 0, 2: -14, 3: -28, 4: -42}   # stagger: curve ends bunch up
    theory_level = {1: 1e-2, 2: 1e-5, 3: 1e-8, 4: 1e-11}
    for k in KS:
        s = grid[(k, 0.0)]
        axes[0].plot(s.curve_alphas, s.curve_thresh, color=COLORS[k - 1])
        theory = np.array([pfa_k_look(a, N_TRAIN, k) for a in s.curve_alphas])
        axes[1].plot(s.curve_alphas, theory, color=COLORS[k - 1])
        # label the MC curve where it ends (tail runs out of events)
        m = np.isfinite(s.curve_thresh)
        if m.any():
            axes[0].annotate(f"K={k}",
                             xy=(s.curve_alphas[m][-1], s.curve_thresh[m][-1]),
                             xytext=(4, label_dy[k]), textcoords="offset points",
                             color=COLORS[k - 1], fontsize=9, va="center")
        # label the theory curve where it crosses a per-K level
        j = np.nonzero(theory >= theory_level[k])[0]
        if j.size:
            axes[1].annotate(f"K={k}", xy=(s.curve_alphas[j[-1]], theory_level[k]),
                             xytext=(4, 0), textcoords="offset points",
                             color=COLORS[k - 1], fontsize=9, va="center")
    for p in PFA_TARGETS:
        for ax in axes:
            ax.axhline(p, color="#888888", lw=0.7, ls=":")
    alpha_nom = {p: alpha_from_pfa_single_look(p, N_TRAIN) for p in PFA_TARGETS}
    for k in KS:
        for p in PFA_TARGETS:
            axes[0].plot(alpha_nom[p], p, marker="+", color=COLORS[k - 1],
                         ms=8, mew=1.4, ls="none")
    axes[0].set_yscale("log")
    axes[0].set_xscale("log")
    axes[0].set_ylim(1e-16, 3)
    axes[0].set_xlim(0.35, 30)
    axes[0].set_xlabel("threshold factor alpha")
    axes[0].set_ylabel("empirical Pfa")
    axes[0].set_title("official chain (Monte Carlo, hann windows)")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_ylim(1e-16, 3)
    axes[1].set_xlim(0.35, 30)
    axes[1].set_xlabel("threshold factor alpha")
    axes[1].set_ylabel("iid-theory Pfa")
    axes[1].set_title("iid theory: 1 - I_{a/(a+N)}(K, NK)")
    s1 = grid[(1, 0.0)]
    fig.suptitle(f"Pfa(alpha) vs channel count K (rho=0), threshold rule - "
                 f"{s1.n_trials} trials/curve; '+' mark nominal single-look alphas",
                 fontsize=10)
    fig.savefig(fig_dir / "fig2_k_effect.png")
    plt.close(fig)

    key = {}
    for k in KS:
        row = [r for r in grid[(k, 0.0)].nominal_rows if r["pfa_nominal"] == 1e-4][0]
        key[f"K{k}_pfa1e-4_factor"] = row["factor_thresh"]
        key[f"K{k}_pfa1e-4_factor_full"] = row["factor_full"]
    return key


def exp3_rho_effect(grid, cfg, fig_dir, res_dir):
    """3) inter-channel correlation effect (K=4)."""
    rows = []
    for rho in RHOS:
        for row in grid[(4, rho)].nominal_rows:
            rows.append(nominal_row_flatten(row, k_channels=4, rho=rho))
    write_csv(res_dir / "exp3_rho_effect.csv", rows,
              ["k_channels", "rho", "pfa_nominal", "alpha_nominal", "count_thresh",
               "n_cells_thresh", "pfa_emp_thresh", "ci_lo_thresh", "ci_hi_thresh",
               "within_ci_thresh", "factor_thresh", "count_full", "pfa_emp_full",
               "ci_lo_full", "ci_hi_full", "within_ci_full", "factor_full", "n_trials"])

    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for i, rho in enumerate(RHOS):
        s = grid[(4, rho)]
        ax.plot(s.curve_alphas, s.curve_thresh, color=COLORS[i])
        m = np.isfinite(s.curve_thresh)
        if m.any():
            ax.annotate(f"rho={rho:g}",
                        xy=(s.curve_alphas[m][-1], s.curve_thresh[m][-1]),
                        xytext=(4, -14 * i), textcoords="offset points",
                        color=COLORS[i], fontsize=9, va="center")
    s1 = grid[(1, 0.0)]
    ax.plot(s1.curve_alphas, s1.curve_thresh, color="#222222", ls="--", lw=1.4)
    m = np.isfinite(s1.curve_thresh)
    # the K=1 curve ends where rho=1 collapses onto it -> park its label above
    ax.annotate("K=1 reference", xy=(s1.curve_alphas[m][-1], s1.curve_thresh[m][-1]),
                xytext=(4, 18), textcoords="offset points", color="#222222",
                fontsize=9, va="center")
    for p in PFA_TARGETS:
        ax.axhline(p, color="#888888", lw=0.7, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(1e-16, 3)
    ax.set_xlim(0.35, 26)
    ax.set_xlabel("threshold factor alpha")
    ax.set_ylabel("empirical Pfa")
    ax.set_title(f"K=4 averaged power vs inter-channel correlation rho "
                 f"(threshold rule)\n{grid[(4, 0.0)].n_trials} trials/curve; "
                 f"rho=1 must collapse onto the K=1 reference")
    fig.savefig(fig_dir / "fig3_rho_effect.png")
    plt.close(fig)

    # cross-check: rho=1 must collapse onto the K=1 curve (compare only the
    # region where both curves hold enough counts to be stable)
    a = grid[(4, 1.0)].curve_thresh
    b = grid[(1, 0.0)].curve_thresh
    m = np.isfinite(a) & np.isfinite(b) & (a >= 1e-3) & (b >= 1e-3)
    assert m.any()
    rel = float(np.nanmax(np.abs(np.log10(a[m]) - np.log10(b[m]))))
    return {"rho1_vs_k1_max_log10_deviation": rel,
            "rho1_vs_k1_alphas_compared": int(m.sum())}


def exp4_db_domain(cfg, trials, seed, fig_dir, res_dir):
    """4) dB-domain 'mean + k*sigma' misuse vs linear-domain CA threshold."""
    k_grid = np.arange(1.5, 10.001, 0.25)
    curves, req = {}, {}
    rows_pfa, rows_req = [], []
    margins = {}
    for k in KS:
        m = collect_db_margins(cfg, k, trials, seed)
        margins[k] = m
        n = m.size
        curves[k] = np.array([count_above(m, kk) / n for kk in k_grid])
        for kk in K_DB_COARSE:
            rows_pfa.append({"k_channels": k, "k_db": kk,
                             "pfa_emp": count_above(m, kk) / n, "n_cells": n,
                             "n_trials": trials})
        req[k] = {}
        for p in PFA_TARGETS:
            kd = float(np.quantile(m, 1.0 - p))
            req[k][p] = kd
            rows_req.append({"k_channels": k, "pfa_target": p, "k_required_db": kd,
                             "n_cells": n, "n_trials": trials})
    write_csv(res_dir / "exp4_db_domain.csv", rows_pfa,
              ["k_channels", "k_db", "pfa_emp", "n_cells", "n_trials"])
    write_csv(res_dir / "exp4_k_required.csv", rows_req,
              ["k_channels", "pfa_target", "k_required_db", "n_cells", "n_trials"])

    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.3))
    for k in KS:
        axes[0].plot(k_grid, curves[k], color=COLORS[k - 1])
        j = np.nonzero(curves[k] >= 1e-5)[0]
        if j.size:
            axes[0].annotate(f"K={k}", xy=(k_grid[j[-1]], 1e-5), xytext=(4, 0),
                             textcoords="offset points", color=COLORS[k - 1],
                             fontsize=9, va="center")
    for p in PFA_TARGETS:
        axes[0].axhline(p, color="#888888", lw=0.7, ls=":")
    axes[0].set_yscale("log")
    axes[0].set_ylim(3e-8, 1)
    axes[0].set_xlim(1.4, 10.6)
    axes[0].set_xlabel("k  (threshold = mean_dB + k * std_dB)")
    axes[0].set_ylabel("empirical Pfa")
    axes[0].set_title("dB-domain mean+k*sigma rule in pure noise")

    xs = np.arange(len(KS))
    vals = [req[k][1e-4] for k in KS]
    axes[1].bar(xs, vals, width=0.6, color=[COLORS[k - 1] for k in KS])
    for x, v in zip(xs, vals):
        axes[1].annotate(f"{v:.1f}", xy=(x, v), xytext=(0, 3),
                         textcoords="offset points", ha="center", fontsize=9)
    axes[1].set_xticks(xs, [f"K={k}" for k in KS])
    axes[1].set_ylim(0, max(vals) * 1.25)
    axes[1].set_ylabel("k required for Pfa = 1e-4")
    axes[1].set_title("k is not transferable across K")
    fig.suptitle("dB(power)-domain mean+k*sigma thresholds "
                 f"(reduced config, {trials} trials)", fontsize=10)
    fig.savefig(fig_dir / "fig5_db_domain.png")
    plt.close(fig)
    return {f"db_k_required_1e-4_K{k}": req[k][1e-4] for k in KS}


def exp5_calibration(grid, fig_dir, res_dir):
    """5) deliverable: calibrated alpha for the K x rho grid."""
    rows = []
    for (k, rho), s in sorted(grid.items()):
        for p in PFA_TARGETS:
            a_single = alpha_from_pfa_single_look(p, N_TRAIN)
            a_klook = alpha_k_look(p, N_TRAIN, k)
            a_t = s.alpha_cal[("thresh", p)]
            a_f = s.alpha_cal[("full", p)]
            rows.append({
                "k_channels": k, "rho": rho, "pfa_target": p,
                "alpha_single_look_theory": a_single,
                "alpha_klook_theory": a_klook,
                "alpha_cal_thresh": a_t,
                "ratio_cal_over_single_thresh": a_t / a_single,
                "snr_overdesign_db_thresh": 10 * np.log10(a_t / a_single),
                "alpha_cal_full": a_f,
                "ratio_cal_over_single_full": a_f / a_single,
                "snr_overdesign_db_full": 10 * np.log10(a_f / a_single),
                "n_trials": s.n_trials, "n_cells": s.n_cells,
            })
    write_csv(res_dir / "exp5_alpha_calibration.csv", rows,
              ["k_channels", "rho", "pfa_target", "alpha_single_look_theory",
               "alpha_klook_theory", "alpha_cal_thresh",
               "ratio_cal_over_single_thresh", "snr_overdesign_db_thresh",
               "alpha_cal_full", "ratio_cal_over_single_full",
               "snr_overdesign_db_full", "n_trials", "n_cells"])

    p = 1e-4
    a_single = alpha_from_pfa_single_look(p, N_TRAIN)
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.0))
    for ax, stat, name in ((axes[0], "thresh", "threshold rule"),
                           (axes[1], "full", "full detector")):
        data = np.zeros((len(KS), len(RHOS)))
        for i, k in enumerate(KS):
            for j, rho in enumerate(RHOS):
                kk = 1 if k == 1 else k
                data[i, j] = grid[(kk, rho if k > 1 else 0.0)].alpha_cal[(stat, p)] / a_single
        im = ax.imshow(data, cmap="Blues", vmin=0, vmax=max(1.05, data.max()),
                       aspect="auto")
        for i in range(len(KS)):
            for j in range(len(RHOS)):
                ax.annotate(f"{data[i, j]:.2f}", xy=(j, i), ha="center",
                            va="center", fontsize=9,
                            color="#222222" if data[i, j] < 0.62 * data.max() else "white")
        ax.set_xticks(range(len(RHOS)), [f"{r:g}" for r in RHOS])
        ax.set_yticks(range(len(KS)), [f"K={k}" for k in KS])
        ax.set_xlabel("inter-channel correlation rho")
        if ax is axes[0]:
            ax.set_ylabel("averaged channels K")
        ax.set_title(f"{name}: alpha_cal / alpha_single-look-theory")
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.85,
                     label="alpha_cal / alpha_single-look")
    fig.suptitle(f"threshold over-design when reusing the single-look factor "
                 f"(target Pfa = {p:g}, reduced config)", fontsize=10)
    fig.savefig(fig_dir / "fig4_alpha_calibration.png")
    plt.close(fig)
    key = {}
    for k, rho in ((4, 0.0), (4, 0.3), (4, 0.7), (2, 0.0), (3, 0.0)):
        s = grid[(k, rho)]
        key[f"alpha_cal_K{k}_rho{rho:g}_1e-4"] = s.alpha_cal[("thresh", 1e-4)]
        key[f"snr_overdesign_db_K{k}_rho{rho:g}_1e-4"] = 10 * np.log10(
            s.alpha_cal[("thresh", 1e-4)]
            / alpha_from_pfa_single_look(1e-4, N_TRAIN))
    return key


def exp6_fullsize(reduced_grid, trials, seed, res_dir):
    """6) confirm the reduced-config conclusions at the full-size config."""
    full = RadarConfig()
    rows = []
    key = {}
    for k in (1, 4):
        red = reduced_grid[(k, 0.0)]
        sample = collect_sample(full, k, 0.0, trials, seed)
        s_full = summarize(sample)
        for p in PFA_TARGETS:
            for stat, arr in (("thresh", sample.ratios),
                              ("full", sample.ratios_lm)):
                a_red = red.alpha_cal[(stat, p)]
                n = sample.n_cells
                c = count_above(arr, a_red)
                p_emp = c / n
                lo, hi = clopper_pearson(c, n)
                rows.append({
                    "k_channels": k, "rho": 0.0, "pfa_target": p, "stat": stat,
                    "alpha_cal_reduced": a_red,
                    "count": c, "n_cells": n, "pfa_emp": p_emp,
                    "ci_lo": lo, "ci_hi": hi,
                    "within_ci_of_target": bool(lo <= p <= hi),
                    "alpha_cal_fullsize": s_full.alpha_cal[(stat, p)],
                    "ratio_full_over_reduced": s_full.alpha_cal[(stat, p)] / a_red,
                    "n_trials": trials, "config": cfg_tag(full),
                })
                if p == 1e-4 and stat == "thresh":
                    key[f"fullsize_K{k}_pfa_at_reduced_alpha"] = p_emp
                    key[f"fullsize_K{k}_within_ci"] = bool(lo <= p <= hi)
                    key[f"fullsize_K{k}_alpha_ratio"] = (
                        s_full.alpha_cal[(stat, p)] / a_red)
    write_csv(res_dir / "exp6_fullsize_check.csv", rows,
              ["k_channels", "rho", "pfa_target", "stat", "alpha_cal_reduced",
               "count", "n_cells", "pfa_emp", "ci_lo", "ci_hi",
               "within_ci_of_target", "alpha_cal_fullsize",
               "ratio_full_over_reduced", "n_trials", "config"])
    return key


def exp7_real_data(proj, seed, grid, fig_dir, res_dir):
    """Optional: spectral-shape comparison with the repo's real far-beam data.

    Honest scope: the captures are unlabelled, so only detection COUNTS are
    reported (never real detection/false-alarm rates); the synthetic frame is
    shown for noise-floor shape comparison only.
    """
    repo = proj.parents[2]
    data_dir = repo / "ADC数据采集" / "示例adc数据和结果"
    files = sorted(p for p in data_dir.glob("*.txt") if "_f0_" in p.name)
    if len(files) != 4:
        print(f"  [exp7] expected 4 far-beam files in {data_dir}, "
              f"found {len(files)} - skipping")
        return {"skipped": True}
    cfg_real = RadarConfig(n_chirps=256, adc_sample_num=1024,
                           rng_nfft=1024, vel_nfft=256)
    adc, n_values = load_real_adc(files, cfg_real.rng_nfft, cfg_real.n_chirps)
    power = range_doppler_power(adc, cfg_real)
    db = to_db(power)
    gn_real = global_noise_db(db)

    rng = make_rng(seed, 7, 4)
    power_s = range_doppler_power(
        synth_real_noise(rng, cfg_real, 4, 0.0), cfg_real)
    db_s = to_db(power_s)
    gn_syn = global_noise_db(db_s)

    a_nom = alpha_from_pfa_single_look(1e-4, N_TRAIN)
    a_cal = grid[(4, 0.0)].alpha_cal[("thresh", 1e-4)]
    mask_nom, _ = cfar_detect(power, a_nom)
    mask_cal, _ = cfar_detect(power, a_cal)
    rows = []
    for path, nv in zip(files, n_values):
        rows.append({"file": path.name, "sha256": sha256_file(path),
                     "values": nv,
                     "note": "uint32 words, 2 int16 samples each"})
    rows.append({"file": "(map)", "sha256": "",
                 "values": f"{power.shape[0]}x{power.shape[1]}",
                 "note": f"detections nominal alpha={a_nom:.2f}: {int(mask_nom.sum())}; "
                         f"calibrated alpha={a_cal:.2f}: {int(mask_cal.sum())}; "
                         f"global noise {gn_real:.2f} dB (real) / {gn_syn:.2f} dB (synthetic), "
                         "counts on unlabelled data, not rates"})
    write_csv(res_dir / "exp7_real_data_inventory.csv", rows,
              ["file", "sha256", "values", "note"])

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1))
    extent = [0, power.shape[1], 0, power.shape[0]]
    for ax, mask, a, tag in ((axes[0], mask_nom, a_nom, "single-look nominal"),
                             (axes[1], mask_cal, a_cal, "K=4 calibrated")):
        im = ax.imshow(db - gn_real, origin="lower", aspect="auto", cmap="Blues_r",
                       vmin=-8, vmax=35, extent=extent)
        yy, xx = np.nonzero(mask)
        # black halo under the orange ring keeps markers visible on
        # saturated bright streaks
        ax.plot(xx, yy, "o", ms=5.4, mec="black", mfc="none", mew=0.5, ls="none")
        ax.plot(xx, yy, "o", ms=4.0, mec="#eda100", mfc="none", mew=0.9, ls="none")
        ax.set_title(tag, fontsize=9)
        ax.annotate(f"alpha={a:.2f}\n{int(mask.sum())} detections",
                    xy=(0.02, 0.98), xycoords="axes fraction", va="top",
                    ha="left", fontsize=9)
        ax.set_xlabel("Doppler bin (fftshifted)")
    axes[0].set_ylabel("range bin")
    fig.colorbar(im, ax=axes[:2], shrink=0.85, label="dB rel. global noise")
    bins = np.linspace(-15, 45, 121)
    axes[2].hist((db - gn_real).ravel(), bins=bins, histtype="step",
                 color=COLORS[0], label="real far-beam f0 (K=4 avg)")
    axes[2].hist((db_s - gn_syn).ravel(), bins=bins, histtype="stepfilled",
                 color=COLORS[2], alpha=0.35, label="synthetic K=4 white noise")
    axes[2].set_yscale("log")
    axes[2].set_xlabel("map dB minus its own global-noise level")
    axes[2].set_ylabel("cell count")
    axes[2].set_title("spectral shape (noise floor region)")
    axes[2].legend(loc="upper left")
    fig.suptitle("Real far-beam f0 captures - spectrum comparison only "
                 "(unlabelled data: counts, not rates)", fontsize=10)
    fig.savefig(fig_dir / "fig6_real_spectrum.png")
    plt.close(fig)
    return {"real_detections_nominal": int(mask_nom.sum()),
            "real_detections_calibrated": int(mask_cal.sum()),
            "global_noise_real_db": gn_real, "global_noise_synthetic_db": gn_syn}


# --------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Reproduce the CA-CFAR false-alarm calibration study.")
    parser.add_argument("--trials", type=int, default=200,
                        help="Monte-Carlo trials per configuration (>=200)")
    parser.add_argument("--full-trials", type=int, default=80,
                        help="trials for the full-size confirmation config")
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--quick", action="store_true",
                        help="small smoke run (fewer trials, skip real data)")
    parser.add_argument("--skip-real-data", action="store_true")
    args = parser.parse_args(argv)
    if args.quick:
        args.trials = min(args.trials, 24)
        args.full_trials = min(args.full_trials, 8)
        args.skip_real_data = True

    t_start = time.perf_counter()
    _setup_matplotlib()
    proj = Path(__file__).resolve().parents[2]
    fig_dir = proj / "figures"
    res_dir = proj / "results"
    fig_dir.mkdir(exist_ok=True)
    res_dir.mkdir(exist_ok=True)
    print(f"project dir : {proj}")
    print(f"python      : {platform.python_version()}, numpy {np.__version__}")
    print(f"seed={args.seed} trials={args.trials} full-trials={args.full_trials} "
          f"N_train={N_TRAIN} (training=(8,6), guard=(2,2))")

    cfg = RadarConfig().reduced()
    timings = {}
    results = {}

    t = time.perf_counter()
    print("[grid] collecting K x rho Monte-Carlo samples "
          f"({cfg_tag(cfg)}, {args.trials} trials each) ...")
    grid = {}
    combos = [(k, r) for k in KS for r in RHOS if not (k == 1 and r > 0)]
    for k, r in combos:
        sample = collect_sample(cfg, k, r, args.trials, args.seed)
        grid[(k, r)] = summarize(sample)
        print(f"  K={k} rho={r:g}: {sample.n_cells:.3g} cells "
              f"({time.perf_counter() - t:.1f}s elapsed)")
        del sample
    rect_summary = summarize(collect_sample(cfg, 1, 0.0, args.trials, args.seed,
                                            window="rect"))
    timings["grid"] = time.perf_counter() - t

    t = time.perf_counter()
    print("[exp1] implementation self-check ...")
    results["exp1"] = exp1_selfcheck(grid, cfg, rect_summary, fig_dir, res_dir)
    print("[exp2] channel-count effect ...")
    results["exp2"] = exp2_k_effect(grid, cfg, fig_dir, res_dir)
    print("[exp3] inter-channel correlation effect ...")
    results["exp3"] = exp3_rho_effect(grid, cfg, fig_dir, res_dir)
    timings["exp123"] = time.perf_counter() - t

    t = time.perf_counter()
    print("[exp4] dB-domain misuse ...")
    results["exp4"] = exp4_db_domain(cfg, args.trials, args.seed, fig_dir, res_dir)
    timings["exp4"] = time.perf_counter() - t

    t = time.perf_counter()
    print("[exp5] alpha calibration table ...")
    results["exp5"] = exp5_calibration(grid, fig_dir, res_dir)
    timings["exp5"] = time.perf_counter() - t

    t = time.perf_counter()
    print("[exp6] full-size confirmation "
          f"({cfg_tag(RadarConfig())}, {args.full_trials} trials) ...")
    results["exp6"] = exp6_fullsize(grid, args.full_trials, args.seed, res_dir)
    timings["exp6"] = time.perf_counter() - t

    if not args.skip_real_data:
        t = time.perf_counter()
        print("[exp7] real-data spectral comparison (optional) ...")
        try:
            results["exp7"] = exp7_real_data(proj, args.seed, grid, fig_dir, res_dir)
        except Exception as exc:  # optional experiment must never fail the run
            print(f"  [exp7] skipped after error: {exc}")
            results["exp7"] = {"skipped": True, "error": str(exc)}
        timings["exp7"] = time.perf_counter() - t

    total = time.perf_counter() - t_start
    summary = {
        "seed": args.seed, "trials": args.trials,
        "full_trials": args.full_trials,
        "n_train_cells": N_TRAIN,
        "reduced_config": cfg_tag(cfg),
        "full_config": cfg_tag(RadarConfig()),
        "timings_s": {k: round(v, 2) for k, v in timings.items()},
        "total_s": round(total, 2),
        "results": results,
    }
    with open(res_dir / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False, default=str)
    print(f"\n=== key numbers ===")
    for name, vals in results.items():
        print(f"[{name}] {vals}")
    print(f"\nwrote {res_dir / 'summary.json'}")
    print(f"total wall time: {total:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
