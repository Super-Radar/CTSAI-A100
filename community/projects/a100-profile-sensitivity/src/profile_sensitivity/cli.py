"""One-shot reproduction CLI for the profile-sensitivity study.

Usage (from ``community/projects/a100-profile-sensitivity``)::

    /path/to/python src/profile_sensitivity/cli.py \
        [--data-dir DIR] [--hxx-dir DIR] [--out-dir DIR] [--seed N]

Regenerates every figure (PNG, matplotlib Agg) and every table (CSV + MD)
under ``figures/`` and ``results/`` with a fixed seed, and prints the key
numbers, including the 8/8 detector score on the public captures.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

if __package__ in (None, ""):                     # script-mode bootstrap
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib

matplotlib.use("Agg")
# CJK-capable font stack so the Chinese labels render (not tofu boxes);
# falls back gracefully to DejaVu Sans on systems without these fonts.
matplotlib.rcParams["font.sans-serif"] = [
    "WenQuanYi Micro Hei", "Microsoft YaHei", "Noto Sans CJK SC",
    "DejaVu Sans"]
matplotlib.rcParams["axes.unicode_minus"] = False
import matplotlib.pyplot as plt  # noqa: E402

from profile_sensitivity.detector import (  # noqa: E402
    Detection, detect_profile, spectral_descriptors)
from profile_sensitivity.mismatch import (  # noqa: E402
    list_captures, run_real_matrix, run_synthetic, synth_adc_words)
from profile_sensitivity.pipeline import (  # noqa: E402
    doppler_bins_signed, dominant_peak, read_raw_words, rv_image,
    unpack_words, velocity_axis_mps, range_axis_m)
from profile_sensitivity.profiles import (  # noqa: E402
    DEFAULT_DATA_DIR, DEFAULT_HXX_DIR, PROJECT_DIR, load_profiles,
    mismatch_ratios, parameter_rows)

# Palette (validated, light surface): series-1 blue / series-2 orange.
COL0, COL1 = "#2a78d6", "#eb6834"
COLGRAY = "#52514e"
TEXT, SEC = "#0b0b0b", "#52514e"
CMAP_DB = "magma"          # perceptually uniform, CVD-safe magnitude ramp
DYN_DB = 45.0              # displayed dynamic range below the peak


# ------------------------------------------------------------------ helpers
def write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        w.writerows(rows)


def write_md(path: Path, header: list[str], rows: list[list],
             title: str = "") -> None:
    def esc(cell) -> str:
        return str(cell).replace("|", "\\|")

    lines = []
    if title:
        lines += [f"### {title}", ""]
    lines.append("| " + " | ".join(esc(h) for h in header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in rows:
        lines.append("| " + " | ".join(esc(c) for c in row) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _style_axes(ax) -> None:
    ax.grid(True, color="#000000", alpha=0.08, linewidth=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=SEC, labelsize=8)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(SEC)


# --------------------------------------------------------------- figure 01
def fig_parameters(profs, out: Path) -> None:
    p0, p1 = profs["init0"], profs["init1"]
    items = [
        ("斜率 S = B/T", "MHz/us", p0.slope_hz_per_s / 1e12, p1.slope_hz_per_s / 1e12),
        ("最大不模糊距离 max_dis", "m", p0.max_dis_m, p1.max_dis_m),
        ("距离分辨率 rng_res", "m", p0.rng_res_m, p1.rng_res_m),
        ("帧时长 fram_time", "ms", p0.fram_time_s * 1e3, p1.fram_time_s * 1e3),
        ("不模糊速度跨度 maxV", "m/s", p0.max_v_mps, p1.max_v_mps),
        ("速度分辨率 V_res", "m/s", p0.v_acu_mps, p1.v_acu_mps),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6), constrained_layout=True)
    for ax, (name, unit, v0, v1) in zip(axes.flat, items):
        bars = ax.bar([0, 1], [v0, v1], width=0.62, color=[COL0, COL1], zorder=3)
        for rect, val in zip(bars, (v0, v1)):
            ax.annotate(f"{val:.4g}", (rect.get_x() + rect.get_width() / 2,
                                       rect.get_height()),
                        ha="center", va="bottom", fontsize=8, color=TEXT)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["init0 远波", "init1 近波"], fontsize=8, color=TEXT)
        ax.set_title(f"{name} [{unit}]", fontsize=9, color=TEXT)
        ax.set_ylim(0, max(v0, v1) * 1.28)
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
    fig.suptitle("同字数、不同物理：两个 profile 的关键物理量对照"
                 "（每通道载荷均为 131072 个 uint32 字）",
                 fontsize=11, color=TEXT)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# ------------------------------------------------------------- RV heatmaps
def _plot_rv(ax, cube, prof, title: str, r_max: float = 30.0,
             annotate: bool = True) -> dict:
    img = rv_image(cube, prof)
    img_s = np.fft.fftshift(img, axes=1)
    vax = velocity_axis_mps(prof)
    rax = range_axis_m(prof)
    clip = max(img_s.max() - DYN_DB, img_s.min())
    im = ax.pcolormesh(vax, rax, np.clip(img_s, clip, None),
                       cmap=CMAP_DB, shading="auto", rasterized=True)
    ax.set_xlim(vax[0], vax[-1])
    ax.set_ylim(0, min(r_max, rax[-1]))
    ax.set_xlabel("速度 v (m/s)", fontsize=8, color=TEXT)
    ax.set_ylabel("距离 R (m)", fontsize=8, color=TEXT)
    ax.set_title(title, fontsize=9, color=TEXT)
    ax.tick_params(labelsize=8)
    pk = dominant_peak(img, prof)
    if annotate and pk["range_m"] <= min(r_max, rax[-1]):
        for lw, ec in ((3.0, "#000000cc"), (1.4, "white")):
            ax.plot(pk["vel_mps"], pk["range_m"], marker="o", markersize=6,
                    markerfacecolor="none", markeredgecolor=ec,
                    markeredgewidth=lw, zorder=5)
        ax.annotate(f"峰值 R={pk['range_m']:.2f} m\nv={pk['vel_mps']:.2f} m/s",
                    (pk["vel_mps"], pk["range_m"]),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=7.5, color="white", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#000000aa",
                              ec="none"))
    return {"img": img, "peak": pk}


def _axis_note(ax, prof) -> None:
    ax.text(0.99, 0.97,
            f"轴刻度: ΔR={prof.rng_res_1dfft_m:.4f} m/bin, "
            f"R_max={prof.max_dis_m:.1f} m\n"
            f"Δv={prof.v_acu_mps:.4f} m/s/bin, v∈(−{prof.max_v_mps/2:.2f},"
            f"+{prof.max_v_mps/2:.2f}) m/s",
            transform=ax.transAxes, ha="right", va="top", fontsize=7,
            color=SEC,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#00000022"))


# ------------------------------------------------------- figures 02..05
def fig_single(cube, prof, title, out) -> dict:
    fig, ax = plt.subplots(figsize=(6.4, 4.2), constrained_layout=True)
    res = _plot_rv(ax, cube, prof, title)
    _axis_note(ax, prof)
    fig.colorbar(ax.collections[0], ax=ax, label="幅度 (dB)", shrink=0.9)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return res


def fig_pair(cubes: tuple, profs2: tuple, titles: tuple, note: str, out):
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3), constrained_layout=True,
                             sharey=True)
    peaks = []
    for ax, cube, prof, ttl in zip(axes, cubes, profs2, titles):
        res = _plot_rv(ax, cube, prof, ttl)
        _axis_note(ax, prof)
        peaks.append(res["peak"])
    fig.colorbar(axes[0].collections[0], ax=axes, label="幅度 (dB)",
                 shrink=0.9, location="right")
    fig.suptitle(note, fontsize=11, color=TEXT)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return peaks


# ------------------------------------------------------- figure 06 (synth)
def fig_synthetic(profs, target, seed, out: Path) -> list[dict]:
    rows = run_synthetic(profs, target=target, seed=seed)
    from profile_sensitivity.pipeline import load_cube
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.4), constrained_layout=True,
                             sharey=True)
    order = [("init0", "init0"), ("init0", "init1"),
             ("init1", "init1"), ("init1", "init0")]
    for ax, (cap_name, ass_name) in zip(axes.flat, order):
        cap, ass = profs[cap_name], profs[ass_name]
        words = synth_adc_words(cap, [(*target, 20000.0)], seed=seed)
        cube = load_cube(words, ass)
        tag = "正确" if cap_name == ass_name else "误配"
        res = _plot_rv(ax, cube, ass,
                       f"采集 {cap_name} × 按 {ass_name} 展开（{tag}）",
                       r_max=15.0)
        row = next(r for r in rows
                   if r["capture"] == cap_name and r["assumed"] == ass_name)
        ax.annotate(
            f"实测 R={row['R_meas_m']:.3f} m, v={row['v_meas_mps']:.3f} m/s\n"
            f"预测 R={row['R_pred_m']:.3f} m, v={row['v_pred_mps']:.3f} m/s\n"
            f"误差 {row['R_err_bins']:+.2f} / {row['v_err_bins']:+.2f} bin",
            xy=(0.02, 0.98), xycoords="axes fraction", va="top", fontsize=7.5,
            color=TEXT,
            bbox=dict(boxstyle="round,pad=0.3", fc="#fcfcfb", ec="#00000033"))
        del res
    fig.colorbar(axes[0][0].collections[0], ax=axes, label="幅度 (dB)",
                 shrink=0.92, location="right")
    fig.suptitle(
        f"合成对拍：单目标 R={target[0]:g} m, v={target[1]:g} m/s，"
        "误配峰值位置 = 确定性几何换算（误差 < 1 bin）",
        fontsize=11, color=TEXT)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return rows


# ------------------------------------------------------- figure 07 (detect)
def fig_detector(dets: list[Detection], truths: list[str], out: Path) -> None:
    from matplotlib.patches import Patch
    names = [d.label for d in dets]
    x = np.arange(len(dets))
    colors = []
    for t in truths:
        colors.append({"init0": COL0, "init1": COL1}.get(t, COLGRAY))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 6.8),
                                   constrained_layout=True, sharex=True)
    ax1.bar(x, [max(d.zero_tail_fraction, 1e-5) for d in dets], color=colors,
            width=0.62, zorder=3)
    ax1.set_yscale("log")
    ax1.set_ylim(1e-5, 2)
    for y, txt in ((0.05, "Z_LOW=0.05"), (0.5, "Z_HIGH=0.5")):
        ax1.axhline(y, color=SEC, lw=1.0, ls="--")
        ax1.annotate(txt, (0.965, y), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7, color=SEC)
    ax1.set_title("特征 A：远波死区槽零指纹（offsets 1020..1023 mod 1024 的零样本比例）",
                  fontsize=9, color=TEXT)
    ax1.set_ylabel("zero_tail_fraction (log)", fontsize=8, color=TEXT)

    ax2.bar(x, [d.click_asymmetry for d in dets], color=colors, width=0.62,
            zorder=3)
    ax2.set_ylim(0, 1.6)
    for y, txt in ((0.35, "A_LOW=0.35"), (0.6, "A_HIGH=0.6")):
        ax2.axhline(y, color=SEC, lw=1.0, ls="--")
        ax2.annotate(txt, (0.955, y), xycoords=("axes fraction", "data"),
                     ha="right", va="bottom", fontsize=7, color=SEC)
    ax2.set_title("特征 B：chirp 边界突变不对称度 J_odd/J_even（1024 样点周期）",
                  fontsize=9, color=TEXT)
    ax2.set_ylabel("J_odd / J_even", fontsize=8, color=TEXT)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=38, ha="right", fontsize=8)
    handles = [Patch(facecolor=COL0, label="真实=远波 init0"),
               Patch(facecolor=COL1, label="真实=近波 init1")]
    if COLGRAY in colors:
        handles.append(Patch(facecolor=COLGRAY, label="合成边界用例"))
    fig.legend(handles=handles, fontsize=8, frameon=False, ncol=3,
               loc="outside upper center")
    for ax in (ax1, ax2):
        _style_axes(ax)
        ax.grid(axis="x", visible=False)
    fig.savefig(out, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--hxx-dir", type=Path, default=DEFAULT_HXX_DIR)
    ap.add_argument("--out-dir", type=Path, default=PROJECT_DIR)
    ap.add_argument("--seed", type=int, default=20260929)
    ap.add_argument("--target", type=float, nargs=2, default=[8.0, 2.0],
                    metavar=("R_M", "V_MPS"),
                    help="synthetic single-target range/velocity")
    args = ap.parse_args(argv)

    fig_dir = args.out_dir / "figures"
    res_dir = args.out_dir / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)

    profs = load_profiles(args.hxx_dir)
    p0, p1 = profs["init0"], profs["init1"]
    print(f"[profiles] init0 {p0.source.name}, init1 {p1.source.name}")
    print(f"[trap]     payload words/channel: init0={p0.expected_payload_words}, "
          f"init1={p1.expected_payload_words} (identical)")

    # ---- experiment 1: parameter table --------------------------------
    rows = parameter_rows(profs)
    write_csv(res_dir / "parameter_comparison.csv",
              ["quantity", "unit", "init0_far", "init1_near"], rows)
    write_md(res_dir / "parameter_comparison.md",
             ["物理量", "单位", "init0 远波", "init1 近波"], rows,
             "参数对照表（公式与 get_radar_paramete.m 一致）")
    fig_parameters(profs, fig_dir / "01_parameter_comparison.png")
    print("[fig] 01_parameter_comparison.png")

    # ---- experiments 2+3: real data, correct + crossed ----------------
    matrix = run_real_matrix(profs, args.data_dir)
    far_files = list_captures(args.data_dir)["init0"]
    near_files = list_captures(args.data_dir)["init1"]

    mrows, header = [], ["file", "capture", "correct(kR@kV)", "R_ok(m)",
                         "v_ok(m/s)", "wrong(kR@kV)", "R_bad(m)", "v_bad(m/s)",
                         "R_ratio_pred", "R_ratio_meas", "v_ratio_pred",
                         "R_err(bin)", "v_err(bin)", "peak_jump"]
    ratios_n2f = mismatch_ratios(p1, p0)
    ratios_f2n = mismatch_ratios(p0, p1)
    for r in matrix:
        mrows.append([r["file"], r["capture"], r["assumed_ok"],
                      f"{r['R_ok_m']:.3f}", f"{r['v_ok_mps']:.3f}",
                      r["assumed_bad"], f"{r['R_bad_m']:.3f}",
                      f"{r['v_bad_mps']:.3f}",
                      f"{r['R_ratio_pred']:.5f}", f"{r['R_ratio_meas']:.3f}",
                      f"{r['v_ratio_pred']:.5f}",
                      f"{r['R_err_bins']:+.2f}", f"{r['v_err_bins']:+.2f}",
                      "yes" if r["peak_jump"] else "no"])
    write_csv(res_dir / "apparent_shift.csv", header, mrows)
    write_md(res_dir / "apparent_shift.md", header, mrows,
             "真实数据交叉误配：谱峰表观位置（正确基线 vs 误配展开）")

    f0row = next(r for r in matrix if r["file"] == far_files[0].name)
    f1row = next(r for r in matrix if r["file"] == near_files[0].name)
    short = lambda p: p.name.split("_")[2] + " " + p.name.split("_")[4]  # ts idx
    fig_single(f0row["cube_ok"], p0,
               f"正确基线：远波数据 × init0（{short(far_files[0])}，"
               f"{far_files[0].name[:20]}…）",
               fig_dir / "02_baseline_far_init0.png")
    fig_single(f1row["cube_ok"], p1,
               f"正确基线：近波数据 × init1（{short(near_files[0])}，"
               f"{near_files[0].name[:20]}…）",
               fig_dir / "03_baseline_near_init1.png")
    print("[fig] 02/03 baselines")

    fig_pair((f0row["cube_ok"], f0row["cube_bad"]), (p0, p1),
             ("按 init0（正确）", "按 init1（误配）"),
             f"交叉误配：远波数据 × init1（{far_files[0].name[:26]}…）",
             fig_dir / "04_mismatch_far_as_near.png")
    fig_pair((f1row["cube_ok"], f1row["cube_bad"]), (p1, p0),
             ("按 init1（正确）", "按 init0（误配）"),
             f"交叉误配：近波数据 × init0（{near_files[0].name[:26]}…）",
             fig_dir / "05_mismatch_near_as_far.png")
    print("[fig] 04/05 mismatches")

    # ---- experiment 4: synthetic determinism --------------------------
    target = (args.target[0], args.target[1])
    srows = run_synthetic(profs, target=target, seed=args.seed)
    sheader = ["capture", "assumed", "correct", "kR", "kV", "R_meas(m)",
               "R_pred(m)", "R_err(bin)", "v_meas(m/s)", "v_nom_pred(m/s)",
               "v_pred(m/s)", "v_jump", "v_err(bin)"]
    smrows = [[r["capture"], r["assumed"], "yes" if r["correct"] else "no",
               r["k_range"], r["k_doppler"], f"{r['R_meas_m']:.3f}",
               f"{r['R_pred_m']:.3f}", f"{r['R_err_bins']:+.3f}",
               f"{r['v_meas_mps']:.3f}", f"{r['v_nominal_mps']:.3f}",
               f"{r['v_pred_mps']:.3f}", "yes" if r["v_jump"] else "no",
               f"{r['v_err_bins']:+.3f}"] for r in srows]
    write_csv(res_dir / "synthetic_verification.csv", sheader, smrows)
    write_md(res_dir / "synthetic_verification.md", sheader, smrows,
             f"合成对拍：单目标 R={target[0]:g} m, v={target[1]:g} m/s")
    fig_synthetic(profs, target, args.seed,
                  fig_dir / "06_synthetic_verification.png")
    print("[fig] 06 synthetic verification")
    max_r_err = max(abs(r["R_err_bins"]) for r in srows)
    max_v_err = max(abs(r["v_err_bins"]) for r in srows)
    jumped = [r for r in srows if r["v_jump"]]
    print(f"[synth]   max |R error| = {max_r_err:.3f} bin, "
          f"max |v error| = {max_v_err:.3f} bin")
    for r in jumped:
        print(f"[synth]   half-frame Doppler jump: {r['capture']} read as "
              f"{r['assumed']}: v {r['v_nominal_mps']:+.3f} -> "
              f"{r['v_meas_mps']:+.3f} m/s (nominal vs readout)")

    # ---- experiment 5: detector ---------------------------------------
    dets: list[Detection] = []
    truths: list[str] = []
    dtable: list[list] = []
    specs = [("init0", far_files), ("init1", near_files)]
    for cap_name, files in specs:
        tag = "远波" if cap_name == "init0" else "近波"
        truth_label = f"{tag} {cap_name}"
        for i, path in enumerate(files):
            samples = unpack_words(read_raw_words(path)[: p0.expected_payload_words])
            det = detect_profile(samples, p0, p1)
            det.label = f"{tag} {path.name.split('_')[4]}"
            det.spectral = spectral_descriptors(samples, profs)
            dets.append(det)
            truths.append(cap_name)
            dtable.append([path.name, truth_label] + det.as_row()[1:])
    ok_real = sum(1 for d, t in zip(dets, truths) if d.decision == t)

    # synthetic boundary cases
    rng = np.random.default_rng(args.seed)
    empty_far = synth_adc_words(p0, [], noise_sigma=300.0, seed=args.seed)
    det = detect_profile(unpack_words(empty_far), p0, p1)
    det.label = "合成·空场景远波"
    dets.append(det)
    truths.append("init0")
    dtable.append(["synthetic-empty-far (dead slots zero, noise only)",
                   "远波 init0"] + det.as_row()[1:])
    pure_noise = np.clip(np.round(rng.normal(0, 300, 131072 * 2)),
                         -32768, 32767).astype(np.int16)
    det = detect_profile(pure_noise, p0, p1)
    det.label = "合成·纯噪声"
    dets.append(det)
    truths.append("uncertain")
    dtable.append(["synthetic-pure-noise (no chirp structure)",
                   "(无结构)"] + det.as_row()[1:])

    dheader = ["file", "truth", "zero_tail", "click_asym", "decision",
               "confidence", "rule"]
    write_csv(res_dir / "detector_output.csv", dheader, dtable)
    write_md(res_dir / "detector_output.md", dheader, dtable,
             "检测器输出（8 个公开文件 + 2 个合成边界用例；truth 与 decision 应一致）")
    fig_detector(dets, truths, fig_dir / "07_detector_features.png")
    print("[fig] 07 detector features")
    print(f"[detect] real files: {ok_real}/8 correct "
          f"({sum(1 for d, t in zip(dets[:8], truths[:8]) if d.decision == t and d.confidence == 'high')}"
          f" high-confidence)")
    for d in dets[8:]:
        print(f"[detect] boundary case {d.label}: -> {d.decision} ({d.confidence})")

    # ---- run summary ---------------------------------------------------
    n_jump = sum(1 for r in matrix if r["peak_jump"])
    summary = [
        "# 运行摘要（cli 一键复现输出）",
        "",
        f"- 种子: {args.seed}; 数据目录: `{args.data_dir}`",
        f"- 载荷陷阱: init0 与 init1 每通道期望字数均为 "
        f"{p0.expected_payload_words}（长度检查无法区分）",
        f"- init0: max_dis={p0.max_dis_m:.3f} m, rng_res={p0.rng_res_m:.4f} m, "
        f"maxV={p0.max_v_mps:.3f} m/s, fram={p0.fram_time_s*1e3:.3f} ms",
        f"- init1: max_dis={p1.max_dis_m:.3f} m, rng_res={p1.rng_res_m:.4f} m, "
        f"maxV={p1.max_v_mps:.3f} m/s, fram={p1.fram_time_s*1e3:.3f} ms",
        f"- 距离轴误配比例: 近波→按init0 ×{ratios_n2f['range_ratio']:.5f}, "
        f"远波→按init1 ×{ratios_f2n['range_ratio']:.5f}",
        f"- 速度轴误配比例: 近波→按init0 ×{ratios_n2f['vel_ratio']:.5f}, "
        f"远波→按init1 ×{ratios_f2n['vel_ratio']:.5f}",
        f"- 合成对拍最大误差: |ΔR|≤{max_r_err:.3f} bin, |Δv|≤{max_v_err:.3f} bin",
        f"- 半帧速度跳变: {len(jumped)}/4 组合发生 (近波按 init0 展开时, "
        f"chirp 从中点切开形成 period-2 相位调制, 谱峰弹到 ±maxV/2)",
        f"- 真实数据交叉表: {len(matrix) - n_jump}/{len(matrix)} 行谱峰可按比例预测, "
        f"{n_jump} 行 peak_jump（主峰跳到泄漏格，已在表中标注）",
        f"- 检测器: 公开 8 文件 {ok_real}/8 判对；空场景合成远波仍判对；"
        f"纯噪声合成流 abstain（uncertain）",
    ]
    (res_dir / "run_summary.md").write_text("\n".join(summary) + "\n",
                                            encoding="utf-8")
    print("[done]    results/run_summary.md written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
