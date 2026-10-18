"""Experiment driver for the chirp phase-sensitivity study.

Runs, with a fixed seed (20261003):

  1. synthetic_gain          -- equal-amplitude complex tone, N=256/128,
                                IID Gaussian phase sweep; G(sigma) vs theory
                                (unwindowed + hann weighted);
  2. correlation_effect      -- AR(1) phases (fixed marginal sigma and fixed
                                innovation sigma) and random walk; G vs
                                correlation length, theory vs simulation;
  3. real_phase_stats        -- official init0 captures, dominant non-DC
                                range bin per RX, adjacent-chirp phase
                                difference statistics (measured reference);
  4. doppler_degradation     -- injected perturbations on the real dominant
                                bins: Doppler peak amplitude ratio, spectral
                                second central moment, neighborhood energy.

Outputs (all under results/): the four CSV tables, data_sha256.txt,
figures/fig1..fig5.png and summary.md (Chinese).

Usage:
    PYTHONPATH=src python -m phase_sens.cli [--data-dir DIR] [--outdir DIR]
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .measures import (
    adjacent_phase_diff,
    circular_stats,
    coherent_gain,
    coherent_gain_matrix,
    coherent_gain_weighted,
    demodulate_peak,
    doppler_metrics_trials,
    theoretical_gain_ar1,
    theoretical_gain_iid,
    theoretical_gain_rw,
    theoretical_gain_weighted_iid,
)
from .perturb import (
    ar1_phase_innovation_batch,
    ar1_phase_marginal_batch,
    correlation_length,
    random_walk_phase_batch,
)
from .pipeline import (
    SENSOR_CONFIGS,
    dominant_bin,
    hann_periodic,
    load_range_matrix,
)

SEED = 20261003

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "ADC数据采集" / "示例adc数据和结果"

SIGMA_GRID = np.linspace(0.0, 1.5, 16)
A_GRID = [0.0, 0.5, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999]
MARGINAL_SIGMAS = [0.2, 0.5, 1.0]
INNOVATION_SIGMAS = [0.05, 0.1, 0.2]

MC_SYNTH = 400
MC_CORR = 300
MC_DOPPLER = 50


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def _write_csv(path: Path, header, rows) -> None:
    lines = [",".join(header)]
    lines += [",".join(str(v) for v in row) for row in rows]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {path.name} ({len(rows)} rows)")


def _wrap(x: np.ndarray) -> np.ndarray:
    return (np.asarray(x) + np.pi) % (2.0 * np.pi) - np.pi


# --------------------------------------------------------------------------
# experiments
# --------------------------------------------------------------------------

def experiment_synthetic_gain(rng) -> list:
    print("[1/4] synthetic G(sigma) benchmark")
    rows = []
    for n in (256, 128):
        # DC-referenced equal-amplitude tone: sum_k s_k is a coherent
        # integrator only for aligned phases (see README theory section),
        # so both gain variants use the DC tone.
        win = hann_periodic(n)
        for sigma in SIGMA_GRID:
            seqs = np.exp(1j * rng.normal(0.0, sigma, size=(MC_SYNTH, n)))
            g_u = coherent_gain_matrix(seqs)
            g_w = coherent_gain_matrix(seqs, win)
            rows.append((
                n, f"{sigma:.6g}", MC_SYNTH,
                f"{g_u.mean():.6f}", f"{g_u.std(ddof=1):.6f}",
                f"{theoretical_gain_iid(sigma, n):.6f}",
                f"{g_w.mean():.6f}", f"{g_w.std(ddof=1):.6f}",
                f"{theoretical_gain_weighted_iid(sigma, win):.6f}",
            ))
    return rows


def experiment_correlation(rng) -> list:
    print("[2/4] correlated phase processes (AR(1) / random walk)")
    rows = []
    n = 256
    # DC-referenced equal-amplitude tone (gain measures need aligned phases)

    def gains(phis):
        seqs = np.exp(1j * phis)
        return coherent_gain_matrix(seqs)

    for sigma in MARGINAL_SIGMAS:
        for a in A_GRID:
            phis = ar1_phase_marginal_batch(rng, MC_CORR, n, a, sigma)
            g = gains(phis)
            rows.append((
                "ar1_marginal", n, f"{sigma:.6g}", f"{a:.6g}",
                f"{correlation_length(a):.6g}", "", MC_CORR,
                f"{g.mean():.6f}", f"{g.std(ddof=1):.6f}",
                f"{theoretical_gain_ar1(sigma, a, n, 'marginal'):.6f}",
            ))
    for sigma_eps in INNOVATION_SIGMAS:
        for a in A_GRID:
            phis = ar1_phase_innovation_batch(rng, MC_CORR, n, a, sigma_eps)
            g = gains(phis)
            rows.append((
                "ar1_innovation", n, "", f"{a:.6g}",
                f"{correlation_length(a):.6g}", f"{sigma_eps:.6g}", MC_CORR,
                f"{g.mean():.6f}", f"{g.std(ddof=1):.6f}",
                f"{theoretical_gain_ar1(sigma_eps, a, n, 'innovation'):.6f}",
            ))
        # random walk: the a -> 1 limit, same innovation std
        phis = random_walk_phase_batch(rng, MC_CORR, n, sigma_eps)
        g = gains(phis)
        rows.append((
            "random_walk", n, "", "", f"{n}", f"{sigma_eps:.6g}", MC_CORR,
            f"{g.mean():.6f}", f"{g.std(ddof=1):.6f}",
            f"{theoretical_gain_rw(sigma_eps, n):.6f}",
        ))
    for sigma_step in np.geomspace(0.002, 0.3, 12):
        sigma_step = float(sigma_step)
        phis = random_walk_phase_batch(rng, MC_CORR, n, sigma_step)
        g = gains(phis)
        rows.append((
            "random_walk_sweep", n, "", "", f"{n}", f"{sigma_step:.6g}", MC_CORR,
            f"{g.mean():.6f}", f"{g.std(ddof=1):.6f}",
            f"{theoretical_gain_rw(sigma_step, n):.6f}",
        ))
    return rows


def experiment_real_phase(rng, data_dir: Path, cfg):
    print("[3/4] official captures: dominant-bin slow-time phase statistics")
    files = sorted(data_dir.glob("adc_*_f0_*.txt"))
    if len(files) != 4:
        raise FileNotFoundError(f"expected 4 f0 captures in {data_dir}, got {len(files)}")
    rows = []
    context = []
    for rx, path in enumerate(files):
        range_matrix, report = load_range_matrix(path, cfg)
        b = dominant_bin(range_matrix)
        seq = range_matrix[b, :]
        # DC-reference the sequence (remove the dominant Doppler ramp) so the
        # plain sum is a valid coherent integrator for the gain measures
        seq_d = demodulate_peak(seq)
        dphi = adjacent_phase_diff(seq)
        circ = circular_stats(dphi)
        residual = _wrap(dphi - circ["mean_angle_rad"])
        lag1 = float(np.corrcoef(residual[:-1], residual[1:])[0, 1])
        win = hann_periodic(cfg.nchirp)
        rows.append((
            f"RX{rx}", path.name, report["actual_len"], report["expected_len"],
            report["extra_count"], report["first_extra_value"],
            b, f"{cfg.bin_range_meters(b):.3f}",
            f"{coherent_gain(seq_d):.6f}",
            f"{coherent_gain_weighted(seq_d, win):.6f}",
            f"{circ['mean_angle_rad']:.6f}", f"{circ['circular_std_rad']:.6f}",
            f"{float(np.std(residual, ddof=1)):.6f}", f"{lag1:.6f}",
        ))
        context.append({
            "rx": rx, "file": path.name, "bin": b, "seq": seq_d,
            "residual": residual, "circ": circ,
        })
        print(f"  RX{rx}: bin={b} (~{cfg.bin_range_meters(b):.2f} m), "
              f"extra={report['extra_count']} (first extra value "
              f"{report['first_extra_value']}), "
              f"circ-std(dphi)={circ['circular_std_rad']:.4f} rad")
    return rows, context


def experiment_doppler_degradation(rng, cfg, context) -> list:
    print("[4/4] Doppler peak / spread degradation under injected phases")
    rows = []
    n = cfg.nchirp
    # context sequences are already DC-referenced (demodulate_peak); Doppler
    # metrics are shift-invariant under that deterministic ramp removal, and
    # the coherent-gain column is only meaningful on the aligned sequence.
    for item in context:
        rx, seq = item["rx"], item["seq"]
        _degradation_curve(rows, rng, cfg, f"RX{rx}", seq,
                           sigma_meas=item["circ"]["circular_std_rad"])
    # synthetic clean-tone reference: opposite M2 direction (spread grows);
    # DC-referenced so the coherent-gain column is meaningful (peak-centred
    # Doppler metrics are shift-invariant under the demodulation)
    tone = demodulate_peak(
        np.exp(1j * 2 * np.pi * 20 * np.arange(n) / n))
    _degradation_curve(rows, rng, cfg, "SYN256", tone,
                       models=("iid",))
    return rows


def _degradation_curve(rows, rng, cfg, rx, seq,
                       models=("iid", "ar1_marginal_a0.95", "rw_terminal_sigma"),
                       sigma_meas=None) -> None:
    n = cfg.nchirp
    ref = doppler_metrics_trials(seq[None, :], cfg)
    peak0 = float(ref["peak_amplitude"][0])
    m2_0 = float(ref["second_central_moment"][0])
    nb0 = float(ref["neighborhood_energy_ratio"][0])
    g0 = coherent_gain(seq)

    def run(model, sigma, phis):
        pert = seq[None, :] * np.exp(1j * phis)
        met = doppler_metrics_trials(pert, cfg)
        g = coherent_gain_matrix(pert)
        rows.append((
            f"{rx}", model, f"{sigma:.6g}", MC_BATCH[model],
            f"{(met['peak_amplitude'] / peak0).mean():.6f}",
            f"{(met['peak_amplitude'] / peak0).std(ddof=1):.6f}",
            f"{met['second_central_moment'].mean():.6f}",
            f"{met['second_central_moment'].std(ddof=1):.6f}",
            f"{met['neighborhood_energy_ratio'].mean():.6f}",
            f"{met['neighborhood_energy_ratio'].std(ddof=1):.6f}",
            f"{g.mean():.6f}", f"{g.std(ddof=1):.6f}",
        ))

    for sigma in SIGMA_GRID:
        if sigma == 0.0:
            phis = np.zeros((MC_DOPPLER, n))
        else:
            phis = rng.normal(0.0, sigma, size=(MC_DOPPLER, n))
        if "iid" in models:
            run("iid", sigma, phis)
        if "ar1_marginal_a0.95" in models:
            phis_ar = (ar1_phase_marginal_batch(rng, MC_DOPPLER, n, 0.95, sigma)
                       if sigma > 0 else np.zeros((MC_DOPPLER, n)))
            run("ar1_marginal_a0.95", sigma, phis_ar)
        if "rw_terminal_sigma" in models:
            phis_rw = (random_walk_phase_batch(rng, MC_DOPPLER, n,
                                               sigma / np.sqrt(n))
                       if sigma > 0 else np.zeros((MC_DOPPLER, n)))
            run("rw_terminal_sigma", sigma, phis_rw)
    if sigma_meas is not None:
        # injection at the measured reference level (see caveats in README)
        phis = rng.normal(0.0, sigma_meas, size=(200, n))
        run("iid_measured_sigma", sigma_meas, phis)
    print(f"  {rx}: peak0={peak0:.1f}, M2(sigma=0)={m2_0:.2f} bins^2, "
          f"NBHD(sigma=0)={nb0:.3f}, G={g0:.3f}")


MC_BATCH = {"iid": MC_DOPPLER, "ar1_marginal_a0.95": MC_DOPPLER,
            "rw_terminal_sigma": MC_DOPPLER, "iid_measured_sigma": 200}


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------

def _style(ax):
    ax.grid(True, alpha=0.3)


def figure_synthetic(rows, outdir: Path) -> None:
    data = {}
    for r in rows:
        n, sigma = int(r[0]), float(r[1])
        data[(n, sigma)] = (float(r[3]), float(r[5]), float(r[6]), float(r[8]))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, weighted, title in ((axes[0], False, "(a) unweighted"),
                                (axes[1], True, "(b) hann-weighted (slow-time window)")):
        for n, color, marker in ((256, "C0", "o"), (128, "C1", "s")):
            sigmas = sorted(s for (nn, s) in data if nn == n)
            sim = np.array([data[(n, s)][2 if weighted else 0] for s in sigmas])
            theo = np.array([data[(n, s)][3 if weighted else 1] for s in sigmas])
            ax.plot(sigmas, theo, "-", color=color, lw=1.5,
                    label=f"theory N={n}")
            ax.plot(sigmas, sim, marker, color=color, ms=4, ls="none",
                    mfc="none", label=f"MC N={n} ({MC_SYNTH} trials)")
        ax.set_yscale("log")
        ax.set_xlabel("Phase error standard deviation $\\sigma$ (rad)")
        ax.set_ylabel("Coherent integration gain $G$ (normalized)")
        ax.set_title(title)
        ax.legend(fontsize=7, loc="lower left")
        _style(ax)
    fig.suptitle("Synthetic tone: G($\\sigma$), simulation vs theory")
    fig.tight_layout()
    fig.savefig(outdir / "fig1_synthetic_gain.png", dpi=150)
    plt.close(fig)
    print("  wrote fig1_synthetic_gain.png")


def figure_correlation(rows, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for j, sigma in enumerate(MARGINAL_SIGMAS):
        ax = axes[0, j]
        sub = [r for r in rows if r[0] == "ar1_marginal" and float(r[2]) == sigma]
        lens = [float(r[4]) for r in sub]
        ax.plot(lens, [float(r[9]) for r in sub], "-", color="C0",
                label="theory (AR1)")
        ax.plot(lens, [float(r[7]) for r in sub], "o", ms=4, color="C1",
                mfc="none", label=f"MC ({MC_CORR} trials)")
        iid_theory = theoretical_gain_iid(sigma, 256)
        ax.axhline(iid_theory, color="gray", ls=":", lw=1,
                   label="IID theory (a=0)")
        ax.set_title(f"fixed marginal $\\sigma$ = {sigma} rad")
        if j == 0:
            ax.set_ylabel("Coherent gain $G$ (normalized)")
            ax.legend(fontsize=7)
    for j, sigma_eps in enumerate(INNOVATION_SIGMAS):
        ax = axes[1, j]
        sub = [r for r in rows if r[0] == "ar1_innovation"
               and float(r[5]) == sigma_eps]
        lens = [float(r[4]) for r in sub]
        ax.plot(lens, [float(r[9]) for r in sub], "-", color="C0",
                label="theory (AR1)")
        ax.plot(lens, [float(r[7]) for r in sub], "o", ms=4, color="C1",
                mfc="none", label=f"MC ({MC_CORR} trials)")
        rw = [r for r in rows if r[0] == "random_walk"
              and float(r[5]) == sigma_eps]
        if rw:
            ax.plot(float(rw[0][4]), float(rw[0][7]), "*", ms=11,
                    color="C3", label="random walk (a$\\to$1)")
        ax.set_title(f"fixed innovation $\\sigma_\\varepsilon$ = {sigma_eps} rad/chirp")
        ax.set_xscale("log")
        ax.set_xlabel("Phase correlation length (chirps)")
        if j == 0:
            ax.set_ylabel("Coherent gain $G$ (normalized)")
            ax.legend(fontsize=7)
    for ax in axes[0]:
        ax.set_xscale("log")
        ax.set_xlabel("Phase correlation length (chirps)")
    fig.suptitle("N=256: correlation length effect — fixed marginal "
                 "$\\sigma$ (top) vs fixed innovation $\\sigma_\\varepsilon$ (bottom)")
    fig.tight_layout()
    fig.savefig(outdir / "fig2_correlation_effect.png", dpi=150)
    plt.close(fig)
    print("  wrote fig2_correlation_effect.png")


def figure_real_phase(context, outdir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for item, ax in zip(context, axes.ravel()):
        res = item["residual"]
        sigma = item["circ"]["circular_std_rad"]
        counts, edges, _ = ax.hist(
            res, bins=41, range=(-np.pi, np.pi), density=True,
            color="C0", alpha=0.65, edgecolor="none")
        ax.set_ylim(0.0, max(counts.max() * 1.35, 0.35))
        xs = np.linspace(-np.pi, np.pi, 400)
        ax.plot(xs, np.exp(-xs ** 2 / (2 * sigma ** 2))
                / (np.sqrt(2 * np.pi) * sigma), "C3-", lw=1.5,
                label="Gaussian, $\\sigma=\\sigma_{circ}$")
        ax.axhline(1.0 / (2 * np.pi), color="gray", ls=":", lw=1,
                   label="uniform density")
        ax.set_title(f"RX{item['rx']} (bin {item['bin']})", fontsize=10)
        ax.text(0.03, 0.95,
                f"$\\sigma_{{circ}}$ = {sigma:.3f} rad\n"
                f"bin-center mean removed",
                transform=ax.transAxes, va="top", fontsize=8,
                bbox=dict(fc="white", ec="none", alpha=0.8, pad=1.5))
        _style(ax)
        ax.legend(fontsize=7, loc="upper right")
    for ax in axes[1]:
        ax.set_xlabel("Adjacent-chirp phase difference (rad)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Probability density (1/rad)")
    fig.suptitle("Measured slow-time adjacent phase differences, "
                 "dominant non-DC range bin per RX (init0, f0 captures)")
    fig.tight_layout()
    fig.savefig(outdir / "fig3_real_phase_distribution.png", dpi=150)
    plt.close(fig)
    print("  wrote fig3_real_phase_distribution.png")


def figure_degradation(rows, outdir: Path, which: str, fname: str,
                       ylabel: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    col = 4 if which == "peak" else 6
    for rx in range(4):
        sub = [r for r in rows if r[0] == f"RX{rx}" and r[1] == "iid"]
        sig = [float(r[2]) for r in sub]
        val = [float(r[col]) for r in sub]
        ax.plot(sig, val, "o-", ms=3.5, lw=1.4, label=f"RX{rx} (iid)")
    sub0 = [r for r in rows if r[0] == "RX0" and r[1] == "ar1_marginal_a0.95"]
    ax.plot([float(r[2]) for r in sub0], [float(r[col]) for r in sub0],
            "--", lw=1.4, color="C4", label="RX0 AR(1), a=0.95 (marginal)")
    subrw = [r for r in rows if r[0] == "RX0" and r[1] == "rw_terminal_sigma"]
    ax.plot([float(r[2]) for r in subrw], [float(r[col]) for r in subrw],
            "-.", lw=1.4, color="C5", label="RX0 random walk (terminal $\\sigma$)")
    subsyn = [r for r in rows if r[0] == "SYN256" and r[1] == "iid"]
    ax.plot([float(r[2]) for r in subsyn], [float(r[col]) for r in subsyn],
            ":", lw=1.6, color="k", label="synthetic tone (iid)")
    ax.set_xlabel("Injected phase $\\sigma$ (rad)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    _style(ax)
    fig.tight_layout()
    fig.savefig(outdir / fname, dpi=150)
    plt.close(fig)
    print(f"  wrote {fname}")


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------

def write_summary(outdir: Path, synth_rows, corr_rows, real_rows,
                  degr_rows, data_dir: Path) -> None:
    cfg = SENSOR_CONFIGS["init0"]
    # theory agreement
    dev_u = max(abs(float(r[3]) - float(r[5])) for r in synth_rows)
    dev_w = max(abs(float(r[6]) - float(r[8])) for r in synth_rows)
    # correlation direction
    g_iid_05 = theoretical_gain_iid(0.5, 256)
    g_long = [r for r in corr_rows if r[0] == "ar1_marginal"
              and float(r[2]) == 0.5 and float(r[3]) == 0.999][0]
    innov = [r for r in corr_rows if r[0] == "ar1_innovation"
             and float(r[5]) == 0.1]
    g_innov_short = innov[1]    # a = 0.5
    g_innov_long = innov[-1]    # a = 0.999 (last of the a-grid)
    rw_row = [r for r in corr_rows if r[0] == "random_walk"
              and float(r[5]) == 0.1][0]
    # real stats
    sigma_meas = [float(r[11]) for r in real_rows]
    # degradation at sigma = 1.0
    def degr_at(sigma, model="iid", rx="RX0"):
        for r in degr_rows:
            if (r[0] == rx and r[1] == model
                    and abs(float(r[2]) - sigma) < 1e-9):
                return r
        return None
    d10 = degr_at(1.0)
    syn0 = degr_at(0.0, rx="SYN256")
    syn10 = degr_at(1.0, rx="SYN256")
    meas_row = [r for r in degr_rows if r[1] == "iid_measured_sigma"][0]

    lines = [
        "# a100-chirp-phase-sensitivity 结果摘要",
        "",
        f"- 随机种子：{SEED}；合成基准 MC={MC_SYNTH}（相关实验 {MC_CORR}，"
        f"Doppler 实验 {MC_DOPPLER}）。",
        f"- 数据：`{data_dir}` 下 8 个采集 txt（哈希见 data_sha256.txt）；"
        f"实测统计基于 init0（rng_nfft=1024, nchirp=256, adc_sample_num="
        f"{cfg.adc_sample_num}）与 f0 4 个 RX 通道。",
        "",
        "## 1. 合成基准：G(σ) 与理论一致性",
        "",
        f"- 无窗最大偏差 max|G_sim−G_theory| = {dev_u:.4f}；"
        f"hann 加窗最大偏差 = {dev_w:.4f}（N=256/128 合并统计，MC 容差内）。",
        f"- 理论锚点：无窗 E[G]=1/N+(1−1/N)e^(−σ²)；加窗 E[G_w]=r+(1−r)e^(−σ²)，"
        f"r=Σw²/(Σw)²；均对 DC 参考序列（或先按主导 Doppler bin 解调）成立。",
        "",
        "## 2. 相关长度效应（N=256）",
        "",
        f"- 固定边缘 σ=0.5 rad：IID（相关长度 1 chirp）G_theory={g_iid_05:.3f}，"
        f"AR(1) a=0.999（相关长度 1000 chirps）G_sim={float(g_long[7]):.3f}——"
        f"**同边缘 σ 下相关越强，损失越小（G 越大）**，与逐对理论 "
        f"exp(−σ²(1−a^h)) 一致。",
        f"- 固定每步新息 σ_ε=0.1 rad：a=0.5 时 G={float(g_innov_short[7]):.3f}，"
        f"a=0.999 时 G={float(g_innov_long[7]):.3f}，随机游走极限 "
        f"G={float(rw_row[7]):.3f}——**同新息下相关越强，G 越小**（相位漂移"
        f"幅度随时间增大），两种参数化给出相反单调方向，报告同时给出。",
        "",
        "## 3. 实测慢时间相位稳定度（描述统计，init0 主导 bin）",
        "",
        "| RX | 主导 bin | 等效距离(m) | 相邻相位差圆标准差(rad) | 残差线性 std(rad) | 实测 G |",
        "|----|---------|------------|------------------------|-------------------|--------|",
    ]
    for r in real_rows:
        lines.append(f"| {r[0]} | {r[6]} | {r[7]} | {r[11]} | {r[12]} | {r[8]} |")
    lines += [
        "",
        f"- 4 个 RX 的相邻相位差圆标准差均值 = {np.mean(sigma_meas):.4f} rad"
        f"（min {min(sigma_meas):.4f} / max {max(sigma_meas):.4f}）：该批采集"
        f"主导非 DC bin 的慢时间序列接近杂波/噪声主导（解调后 G 仅 "
        f"{min(float(r[8]) for r in real_rows):.2f}–"
        f"{max(float(r[8]) for r in real_rows):.2f}），相邻相位近似去相关。",
        "- 声明：以上仅为该批采集自身相位稳定度的描述统计；注入的扰动模型"
        "（IID 高斯 / AR / 随机游走）不等价于真实晶振或 PLL 相位噪声，"
        "也不能由实测谱扩散反推硬件相位噪声。",
        "",
        "## 4. 注入扰动对 Doppler 谱的影响（RX0，IID 模型）",
        "",
    ]
    if d10 is not None:
        lines += [
            f"- 实测 RX0：σ=1.0 rad 时 Doppler 峰幅度比 = {float(d10[4]):.3f}"
            f"（相对 σ=0），峰邻域(±3 bin)能量占比 {float(d10[8]):.3f}"
            f"（σ=0 时 {float(degr_at(0.0)[8]):.3f}），相干增益 "
            f"{float(d10[10]):.3f}（σ=0 时 {float(degr_at(0.0)[10]):.3f}）；"
            f"谱二阶中心矩由 {float(degr_at(0.0)[6]):.0f} bins² 变为 "
            f"{float(d10[6]):.0f} bins²——杂波主导的实测谱本身已很宽，"
            f"相位扰动对其主要表现为平滑（略降），峰浓度下降才是主要效应。",
        ]
    if syn0 is not None and syn10 is not None:
        lines += [
            f"- 合成纯净单音对照（SYN256）：M2 由 {float(syn0[6]):.2f} bins² "
            f"（hann 核理论值 1/3）增至 {float(syn10[6]):.1f} bins²，"
            f"峰幅度比降至 {float(syn10[4]):.3f}——干净目标上扩散显著增大。",
        ]
    lines += [
        f"- 在实测参考水平注入（RX0，σ={meas_row[2]} rad）：峰幅度比 = "
        f"{float(meas_row[4]):.3f}，二阶中心矩 = {float(meas_row[6]):.2f} "
        f"bins²，相干增益 = {float(meas_row[10]):.3f}——显著低于实测序列自身 "
        f"G（实测 Δφ 散布主要来自杂波起伏而非目标相位噪声，再次说明不能"
        f"反推硬件）。",
        "",
        "## 5. 产物清单",
        "",
        "- synthetic_gain.csv / correlation_effect.csv / real_phase_stats.csv /"
        " doppler_degradation.csv",
        "- figures/fig1..fig5（PNG，150 dpi）",
        "- data_sha256.txt",
        "",
        "## 6. 局限",
        "",
        "- 注入模型 ≠ 真实晶振/PLL 相位噪声；不能由实测谱扩散反推硬件相位"
        "噪声；结论仅限受控扰动实验；离线分析，未做硬件实测。",
        "",
    ]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("  wrote summary.md")


def write_hashes(data_dir: Path, outdir: Path) -> list:
    lines = []
    files = sorted(data_dir.glob("*.txt"))
    for f in files:
        digest = hashlib.sha256(f.read_bytes()).hexdigest()
        lines.append(f"{digest}  {f.name}")
    (outdir / "data_sha256.txt").write_text("\n".join(lines) + "\n",
                                            encoding="utf-8")
    print(f"  wrote data_sha256.txt ({len(files)} files)")
    return lines


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--outdir", type=Path,
                        default=PROJECT_ROOT / "results")
    args = parser.parse_args(argv)
    args.outdir.mkdir(parents=True, exist_ok=True)
    figdir = args.outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    print(f"seed = {SEED}")
    print(f"data dir = {args.data_dir}")
    write_hashes(args.data_dir, args.outdir)

    rng = np.random.default_rng(SEED)
    cfg0 = SENSOR_CONFIGS["init0"]

    synth_rows = experiment_synthetic_gain(rng)
    _write_csv(args.outdir / "synthetic_gain.csv",
               ["n", "sigma_rad", "mc_trials",
                "gain_unweighted_mean", "gain_unweighted_std",
                "gain_unweighted_theory",
                "gain_weighted_mean", "gain_weighted_std",
                "gain_weighted_theory"], synth_rows)
    figure_synthetic(synth_rows, figdir)

    corr_rows = experiment_correlation(rng)
    _write_csv(args.outdir / "correlation_effect.csv",
               ["model", "n", "sigma_marginal_rad", "a", "corr_len_chirps",
                "sigma_step_rad", "mc_trials", "gain_mean", "gain_std",
                "gain_theory"], corr_rows)
    figure_correlation(corr_rows, figdir)

    real_rows, context = experiment_real_phase(rng, args.data_dir, cfg0)
    _write_csv(args.outdir / "real_phase_stats.csv",
               ["rx", "file", "actual_len", "expected_len", "extra_count",
                "first_extra_value", "dominant_bin", "bin_range_m",
                "coherent_gain", "coherent_gain_weighted",
                "dphi_circ_mean_rad", "dphi_circ_std_rad",
                "dphi_residual_std_rad", "dphi_residual_lag1_corr"],
               real_rows)
    figure_real_phase(context, figdir)

    degr_rows = experiment_doppler_degradation(rng, cfg0, context)
    _write_csv(args.outdir / "doppler_degradation.csv",
               ["rx", "model", "sigma_rad", "mc_trials",
                "peak_amp_ratio_mean", "peak_amp_ratio_std",
                "second_moment_bins2_mean", "second_moment_bins2_std",
                "nbhd_energy_ratio_mean", "nbhd_energy_ratio_std",
                "coherent_gain_mean", "coherent_gain_std"], degr_rows)
    figure_degradation(degr_rows, figdir, "peak",
                       "fig4_doppler_peak_degradation.png",
                       "Doppler peak amplitude (relative to $\\sigma$=0)",
                       "Doppler peak degradation vs injected phase error")
    figure_degradation(degr_rows, figdir, "moment",
                       "fig5_spectral_spread.png",
                       "Second central moment (bins$^2$)",
                       "Doppler spectral spread vs injected phase error")

    write_summary(args.outdir, synth_rows, corr_rows, real_rows,
                  degr_rows, args.data_dir)
    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
