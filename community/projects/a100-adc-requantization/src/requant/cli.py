"""One-click reproduction of the requantisation budget study.

Run from the project directory (or anywhere; the repo root is auto-detected
three levels above this file)::

    /home/cnj/openSource/.venv-a100/bin/python \
        community/projects/a100-adc-requantization/src/requant/cli.py

Fixed seed 20261003 for every stochastic stage.  Produces under
``community/projects/a100-adc-requantization/``:

* results/synthetic_sweep.csv
* results/real_per_file.csv
* results/data_sha256.txt
* results/summary.md            (Chinese, key numbers inline)
* figures/fig1_time_rmse.png .. fig5_peak_drift.png
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

from . import __version__, SEED
from .config import DATA_DIR, INIT0, INIT1, REAL_FILES, SYNTH
from . import fidelity as fid
from . import figures as figs
from .pipeline import load_adc_txt, spectral_chain, unpack_words
from .requantize import requantize
from .synthetic import (NOISE_BAND, TONE_RNG_BIN, TONE_VEL_BIN, WEAK_RNG_BIN,
                        WEAK_VEL_BIN, synthetic_cases)

K_LIST = list(range(4, 15)) + [16]
MODES = ("round", "trunc")
RETENTION_CRITERION = 0.99
VISIBLE_DB = 10.0

CSV_FIELDS = ["case", "cfg", "k", "mode",
              "time_rmse_lsb", "time_rmse_over_rms",
              "rng_rmse", "rng_rmse_over_peak",
              "rd_rmse", "rd_rmse_over_peak",
              "band_retention", "peak_amp_ratio", "peak_amp_db",
              "rng_bin_shift", "vel_bin_shift", "weak_ratio_db"]

PROJECT_DIR = Path(__file__).resolve().parents[2]


def repo_root_default() -> Path:
    """<root>/community/projects/<project>/src/requant/cli.py -> repo root."""
    return PROJECT_DIR.parents[2]


def _fmt(v, spec="%.6g"):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return ""
    return spec % v


def synth_band_for(case: str, shape: tuple[int, int]) -> np.ndarray:
    """Ground-truth signal band for a synthetic case (predefined, not k-adapted).

    Tone cases: +/-2 bins around the tone; two-tone: union of both boxes;
    band-limited noise: the true fast-time support x all Doppler bins.
    """
    if case == "two_tone":
        return fid.signal_band(shape, [(TONE_RNG_BIN, TONE_VEL_BIN, 2, 2),
                                       (WEAK_RNG_BIN, WEAK_VEL_BIN, 2, 2)])
    if case == "bandlimited_noise":
        mid = (NOISE_BAND[0] + NOISE_BAND[1]) // 2
        half = (NOISE_BAND[1] - NOISE_BAND[0]) // 2
        return fid.signal_band(shape, [(mid, SYNTH.vel_nfft // 2, half,
                                        SYNTH.vel_nfft)])
    if case == "tone_weak_alone":
        return fid.signal_band(shape, [(WEAK_RNG_BIN, WEAK_VEL_BIN, 2, 2)])
    return fid.signal_band(shape, [(TONE_RNG_BIN, TONE_VEL_BIN, 2, 2)])


def sweep_one(samples: np.ndarray, cfg, case: str, cfg_name: str,
              band: np.ndarray | None = None
              ) -> tuple[list[dict], dict]:
    """Requantise ``samples`` over K_LIST x MODES; return (rows, extras).

    ``band`` is the predefined retention band: ground-truth signal band for
    synthetic cases, baseline top-1% bins for measured files.  extras carries
    the baseline products needed by figures/summary (band, baseline spectra,
    per-k RD maps for the two-tone figure).
    """
    base_rng, base_rd = spectral_chain(samples, cfg)
    if band is None:
        band = fid.predefined_band(base_rd, frac=0.01)
    base_peak = fid.normalize_peak(base_rd)
    base_rng_peak = fid.normalize_peak(base_rng)
    base_rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    rows: list[dict] = []
    maps_by_k: dict[int, np.ndarray] = {}
    for k in K_LIST:
        for mode in MODES:
            q = requantize(samples, k, mode)
            rng_s, rd_m = spectral_chain(q, cfg)
            if k in (6, 8, 10, 12, 16) and mode == "round":
                maps_by_k[k] = rd_m
            drift = fid.peak_drift(rd_m, base_rd)
            ratio = drift["peak_amp_ratio"]
            row = {
                "case": case, "cfg": cfg_name, "k": k, "mode": mode,
                "time_rmse_lsb": fid.time_rmse(q, samples),
                "time_rmse_over_rms": None if base_rms == 0 else
                fid.time_rmse(q, samples) / base_rms,
                "rng_rmse": fid.spectrum_rmse(rng_s, base_rng),
                "rng_rmse_over_peak": fid.spectrum_rmse(rng_s, base_rng)
                / base_rng_peak,
                "rd_rmse": fid.spectrum_rmse(rd_m, base_rd),
                "rd_rmse_over_peak": fid.spectrum_rmse(rd_m, base_rd)
                / base_peak,
                "band_retention": fid.band_energy_retention(rd_m, base_rd,
                                                            band),
                "peak_amp_ratio": ratio,
                "peak_amp_db": (20.0 * np.log10(ratio) if ratio > 0.0
                                else float("-inf")),
                "rng_bin_shift": drift["rng_bin_shift"],
                "vel_bin_shift": drift["vel_bin_shift"],
                "weak_ratio_db": None,
            }
            rows.append(row)
    extras = {"band": band, "base_rng": base_rng, "base_rd": base_rd,
              "maps_by_k": maps_by_k, "base_rms": base_rms,
              "base_peak": base_peak}
    return rows, extras


def _weak_visibility_rows(rows: list[dict], samples, cfg) -> None:
    """Fill weak_ratio_db for the two-tone case (expected weak bin known)."""
    for r in rows:
        q = requantize(samples, r["k"], r["mode"])
        _, rd_m = spectral_chain(q, cfg)
        vis = fid.weak_tone_visibility(rd_m, WEAK_RNG_BIN, WEAK_VEL_BIN)
        r["weak_ratio_db"] = vis["ratio_db"]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_FIELDS)
        for r in rows:
            w.writerow([
                r["case"], r["cfg"], r["k"], r["mode"],
                _fmt(r["time_rmse_lsb"]), _fmt(r["time_rmse_over_rms"]),
                _fmt(r["rng_rmse"]), _fmt(r["rng_rmse_over_peak"]),
                _fmt(r["rd_rmse"]), _fmt(r["rd_rmse_over_peak"]),
                _fmt(r["band_retention"]), _fmt(r["peak_amp_ratio"]),
                _fmt(r["peak_amp_db"]),
                r["rng_bin_shift"], r["vel_bin_shift"],
                _fmt(r["weak_ratio_db"]),
            ])


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_k(rows: list[dict], mode: str, case: str,
           criterion: float = RETENTION_CRITERION,
           two_sided: bool = False) -> int | None:
    """Smallest k in 4..16 meeting the retention criterion.

    Measured data: one-sided R >= criterion (energy only ever leaks out).
    Synthetic tones: two-sided |R - 1| <= 1-criterion as well, because a
    tone barely above half a step converts to a square wave whose band
    energy *overshoots* (e.g. R ~ 1.6) -- distortion, not preservation.
    """
    ret = {r["k"]: r["band_retention"] for r in rows
           if r["mode"] == mode and r["case"] == case}
    tol = 1.0 - criterion
    for k in K_LIST:
        r = ret.get(k, 0.0)
        ok = r >= criterion and (abs(r - 1.0) <= tol if two_sided else True)
        if ok:
            return k
    return None


def survival_k(rows: list[dict], mode: str, case: str) -> int | None:
    """Smallest k whose band retention is non-zero (hard annihilation floor)."""
    ret = {r["k"]: r["band_retention"] for r in rows
           if r["mode"] == mode and r["case"] == case}
    for k in K_LIST:
        if ret.get(k, 0.0) > 0.0:
            return k
    return None


def mean_over_files(per_file: dict[str, list[dict]], field: str,
                    mode: str, k: int) -> float:
    vals = [next(r[field] for r in rows if r["mode"] == mode and r["k"] == k)
            for rows in per_file.values()]
    return float(np.mean(vals))


def max_abs_over_files(per_file: dict[str, list[dict]], field: str,
                       mode: str, k: int) -> float:
    vals = [next(r[field] for r in rows if r["mode"] == mode and r["k"] == k)
            for rows in per_file.values()]
    return float(np.max(np.abs(vals)))


# ------------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="a100-adc-requantization spectral fidelity budget (seed "
                    f"{SEED})")
    ap.add_argument("--repo-root", type=Path, default=repo_root_default())
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--project-dir", type=Path, default=PROJECT_DIR)
    args = ap.parse_args(argv)

    res_dir = args.project_dir / "results"
    fig_dir = args.project_dir / "figures"
    res_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    print(f"[requant] version {__version__}, seed {args.seed}, "
          f"repo root {args.repo_root}")

    # ---------------------------------------------------------- synthetic
    print("[requant] synthetic benchmarks ...")
    synth_rows: list[dict] = []
    synth_extras: dict[str, dict] = {}
    for name, frame in synthetic_cases(args.seed).items():
        _, base_rd = spectral_chain(frame, SYNTH)
        rows, extras = sweep_one(frame, SYNTH, name, "synthetic",
                                 band=synth_band_for(name, base_rd.shape))
        if name == "two_tone":
            _weak_visibility_rows(rows, frame, SYNTH)
        synth_rows.extend(rows)
        synth_extras[name] = extras
    write_csv(res_dir / "synthetic_sweep.csv", synth_rows)

    # -------------------------------------------------------------- real
    print("[requant] real ADC files ...")
    real_rows: list[dict] = []
    per_file: dict[str, list[dict]] = {}
    file_stats: dict[str, dict] = {}
    sha_lines: list[str] = []
    for fname, cfg, ch in REAL_FILES:
        path = args.repo_root / DATA_DIR / fname
        lr = load_adc_txt(path, cfg)
        if lr.had_excess:
            print(f"[requant] {fname}: {lr.n_values} values vs expected "
                  f"{lr.expected}; dropping {lr.excess_count} excess value(s), "
                  f"first excess value = {lr.excess_head[0]}")
        samples = unpack_words(lr.words, cfg)
        smin, smax = int(samples.min()), int(samples.max())
        srms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        grid4 = float(np.mean(samples.astype(np.int64) % 4 == 0))
        file_stats[fname] = {"min": smin, "max": smax, "rms": srms,
                             "excess": lr.excess_count, "grid4": grid4,
                             "first_excess": (lr.excess_head or [None])[0]}
        sha_lines.append(f"{sha256_of(path)}  {DATA_DIR}/{fname}")
        rows, _ = sweep_one(samples, cfg, fname, cfg.name)
        real_rows.extend(rows)
        per_file[fname] = rows
    write_csv(res_dir / "real_per_file.csv", real_rows)
    (res_dir / "data_sha256.txt").write_text("\n".join(sha_lines) + "\n",
                                             encoding="utf-8")

    # ---------------------------------------------------------- figures
    print("[requant] figures ...")
    ks = np.array(K_LIST, dtype=float)

    def synth_curve(case, field, mode):
        return [next(r[field] for r in synth_rows
                     if r["case"] == case and r["mode"] == mode
                     and r["k"] == k) for k in K_LIST]

    def real_mean(field, mode):
        return [mean_over_files(per_file, field, mode, k) for k in K_LIST]

    def theory(k, kind):
        d = 2.0 ** (16 - np.asarray(k))
        return d / np.sqrt(12.0) if kind == "round" else d / np.sqrt(3.0)

    figs.fig1_time_rmse(K_LIST, {
        "synth tone A=12000 (round)": (ks, synth_curve("tone_A12000", "time_rmse_lsb", "round"), figs.C_ROUND, "-"),
        "synth tone A=12000 (trunc)": (ks, synth_curve("tone_A12000", "time_rmse_lsb", "trunc"), figs.C_TRUNC, "-"),
        "synth band-limited noise (round)": (ks, synth_curve("bandlimited_noise", "time_rmse_lsb", "round"), figs.C_SYNTH, "-"),
        "real files mean (round)": (ks, real_mean("time_rmse_lsb", "round"), figs.C_REAL, "-"),
        "real files mean (trunc)": (ks, real_mean("time_rmse_lsb", "trunc"), figs.C_REAL, "--"),
        "round theory $\\Delta/\\sqrt{12}$": (ks, theory(ks, "round"), figs.C_REF, ":"),
        "trunc theory $\\Delta/\\sqrt{3}$": (ks, theory(ks, "trunc"), figs.C_REF, "-."),
    }, str(fig_dir / "fig1_time_rmse.png"))

    figs.fig2_spectrum_rmse(K_LIST, {
        "synth tone (round)": (ks, synth_curve("tone_A12000", "rng_rmse_over_peak", "round"), figs.C_ROUND, "-"),
        "synth tone (trunc)": (ks, synth_curve("tone_A12000", "rng_rmse_over_peak", "trunc"), figs.C_TRUNC, "-"),
        "real mean (round)": (ks, real_mean("rng_rmse_over_peak", "round"), figs.C_REAL, "-"),
        "real mean (trunc)": (ks, real_mean("rng_rmse_over_peak", "trunc"), figs.C_REAL, "--"),
    }, {
        "synth tone (round)": (ks, synth_curve("tone_A12000", "rd_rmse_over_peak", "round"), figs.C_ROUND, "-"),
        "synth tone (trunc)": (ks, synth_curve("tone_A12000", "rd_rmse_over_peak", "trunc"), figs.C_TRUNC, "-"),
        "real mean (round)": (ks, real_mean("rd_rmse_over_peak", "round"), figs.C_REAL, "-"),
        "real mean (trunc)": (ks, real_mean("rd_rmse_over_peak", "trunc"), figs.C_REAL, "--"),
    }, str(fig_dir / "fig2_spectrum_rmse.png"))

    fig3_curves = {
        "synth band-limited noise": (ks, synth_curve("bandlimited_noise", "band_retention", "round"), figs.C_SYNTH, "-"),
        "synth tone A=12000": (ks, synth_curve("tone_A12000", "band_retention", "round"), figs.C_ROUND, "-"),
        "synth tone A=47": (ks, synth_curve("tone_A47", "band_retention", "round"), figs.C_TRUNC, "-"),
    }
    for i, (fname, rows) in enumerate(per_file.items()):
        style = "-" if fname.startswith("adc_远波") else "--"
        v = [next(r["band_retention"] for r in rows
                  if r["mode"] == "round" and r["k"] == k) for k in K_LIST]
        lbl = "real far-beam (init0, 4 ch)" if i == 0 else (
            "real near-beam (init1, 4 ch)" if i == 4 else None)
        if lbl:
            fig3_curves[lbl] = (ks, v, figs.C_REAL, style)
    figs.fig3_band_retention(K_LIST, fig3_curves,
                             str(fig_dir / "fig3_band_retention.png"))

    figs.fig4_two_tone(synth_extras["two_tone"]["maps_by_k"],
                       [16, 12, 10, 8, 6], SYNTH.rng_nfft // 2,
                       str(fig_dir / "fig4_two_tone_visibility.png"),
                       weak_bin=(WEAK_RNG_BIN, WEAK_VEL_BIN))

    figs.fig5_peak_drift(K_LIST, {
        "synth tone (round)": (ks, synth_curve("tone_A12000", "peak_amp_db", "round"), figs.C_ROUND, "-"),
        "synth tone (trunc)": (ks, synth_curve("tone_A12000", "peak_amp_db", "trunc"), figs.C_TRUNC, "-"),
        "real mean (round)": (ks, real_mean("peak_amp_db", "round"), figs.C_REAL, "-"),
        "real mean (trunc)": (ks, real_mean("peak_amp_db", "trunc"), figs.C_REAL, "--"),
    }, {
        "synth tone (round)": (ks, np.abs(np.array(synth_curve("tone_A12000", "rng_bin_shift", "round"), dtype=float)), figs.C_ROUND, "-"),
        "real max |shift| (round)": (ks, [max_abs_over_files(per_file, "rng_bin_shift", "round", k) for k in K_LIST], figs.C_REAL, "-"),
        "real max |shift| (trunc)": (ks, [max_abs_over_files(per_file, "rng_bin_shift", "trunc", k) for k in K_LIST], figs.C_REAL, "--"),
    }, str(fig_dir / "fig5_peak_drift.png"))

    # ---------------------------------------------------------- summary
    print("[requant] summary.md ...")
    write_summary(res_dir / "summary.md", synth_rows, per_file, file_stats,
                  args.seed)
    print(f"[requant] done. results in {res_dir}, figures in {fig_dir}")
    return 0


def write_summary(path: Path, synth_rows, per_file, file_stats, seed) -> None:
    lines: list[str] = []
    add = lines.append
    add("# a100-adc-requantization 结果摘要\n")
    add(f"（固定种子 {seed}；口径：所有指标相对同数据 k=16 基线，全程固定满量程"
        " int16 [-32768, 32767]，无逐文件自适应归一化。"
        "实测频带 = 基线 RD 谱能量前 1% bin；合成频带 = 真值信号 bin ±2。\n"
        "判据：A=单向 R ≥ 99%（能量不被丢掉）；B=双侧 |R−1| ≤ 1%（保真，"
        "同时惩罚丢失与注入——再量化噪声、trunc 直流偏置、勉强过半步长的"
        "单音方波化（R 上冲如 ≈1.6）都会把能量加进频带，那是失真不是保持。）\n")

    add("## 1. 数据事实\n")
    for fname, st in file_stats.items():
        add(f"- `{fname}`: 解包样本范围 [{st['min']}, {st['max']}]，"
            f"RMS {st['rms']:.1f} LSB；文件值数 131073 = 目标 131072 + 1，"
            f"首个多余值 = {st['first_excess']}（显式报告后截断，"
            "与官方 load_adc_data.m 的 warning 截断一致）")
    fs_abs = max(max(abs(st["min"]), abs(st["max"]))
                 for st in file_stats.values())
    add(f"\n8 个文件解包后的样本绝对值最大仅 {fs_abs} LSB "
        f"(≈满量程 32768 的 {fs_abs / 32768 * 100:.1f}%)，"
        "即实测数据本身只占用约 10 位动态范围；"
        "再量化损失因此在这批数据上比满量程信号更早出现，"
        "安全下限结论只对这批数据成立。")
    g4 = min(st["grid4"] for st in file_stats.values())
    add(f"\n进一步：全部 8 文件 {g4 * 100:.1f}% 样本都是 4 的倍数"
        "（低 2 位恒为 0），即这批 dump 的存储粒度本身就是 ≤14 位；"
        "因此 k=14 的再量化对实测数据是**逐位恒等**"
        "（real_per_file.csv 中 k=14 的 band_retention 恰为 1）。"
        "这从数据侧直接印证了“16 位存储 ≠ 16 位有效精度”。\n")

    add("## 2. round 与 trunc 的差异\n")
    r16 = next(r for r in synth_rows if r["case"] == "tone_A12000"
               and r["k"] == 10 and r["mode"] == "round")
    t16 = next(r for r in synth_rows if r["case"] == "tone_A12000"
               and r["k"] == 10 and r["mode"] == "trunc")
    add(f"- 合成单音 A=12000 @ k=10（Δ=64）：时域 RMSE "
        f"round {r16['time_rmse_lsb']:.1f} LSB vs trunc "
        f"{t16['time_rmse_lsb']:.1f} LSB，比值 "
        f"{t16['time_rmse_lsb'] / r16['time_rmse_lsb']:.2f}"
        "（理论 2.0：无偏 Δ²/12 vs 含 Δ²/4 直流偏置的 Δ²/3）")
    add("- trunc（丢低位=floor）误差落在 [0, Δ) 内、均值 +Δ/2，"
        "偏置进入谱后集中在零多普勒/低距离 bin，"
        "使同 k 下频带能量保持率更低、安全下限普遍比 round 更高"
        "（逐文件数值见下表）。")

    add("\n## 3. k 的“安全下限”\n")
    add("### 3.1 实测 8 文件，两种显式判据\n")
    add("判据 A（单向，任务书口径）：基线 top-1% bin 频带能量保持率 R ≥ 99%，"
        "度量“信号能量是否被丢掉”。")
    add("判据 B（双侧，保真口径）：|R − 1| ≤ 1%，同时惩罚能量丢失与能量注入"
        "（再量化噪声与 trunc 直流偏置都会把能量**加进**频带）。\n")
    add("| 数据 | A: round | A: trunc | B: round | B: trunc |")
    add("|---|---|---|---|---|")
    a_round, a_trunc, b_round, b_trunc = [], [], [], []
    for fname, rows in per_file.items():
        ka_r = safe_k(rows, "round", fname)
        ka_t = safe_k(rows, "trunc", fname)
        kb_r = safe_k(rows, "round", fname, two_sided=True)
        kb_t = safe_k(rows, "trunc", fname, two_sided=True)
        a_round.append(ka_r); a_trunc.append(ka_t)
        b_round.append(kb_r); b_trunc.append(kb_t)
        add(f"| {fname} | {ka_r if ka_r is not None else '>14'} |"
            f" {ka_t if ka_t is not None else '>14'} |"
            f" {kb_r if kb_r is not None else '>14'} |"
            f" {kb_t if kb_t is not None else '>14'} |")
    worst = lambda seq: max((k for k in seq if k is not None), default=None)
    add("")
    add(f"- 判据 A：全部 8 文件 round 最低 k = {worst(a_round)} 位。"
        f"trunc 表面为 {worst(a_trunc)} 位，**但该判据对 trunc 失效**："
        "trunc 的 +Δ/2 直流偏置把大量假能量注进频带内的零多普勒/低距离 bin"
        "（如 f0_0 通道 0 在 k=4 时 R=42.3、RD 峰值跳到 (0,0) 且达基线峰值 "
        "42 倍），单向 ≥99% 被这种注入“满足”。")
    add(f"- 判据 B（推荐）：全部 8 文件 round 最低 k = {worst(b_round)} 位、"
        f"trunc = {worst(b_trunc)} 位。round 与 trunc 的差距（约 2-3 位）"
        "是 round vs trunc 最实质的差异：偏置让 trunc 既抬高谱底又污染频带。"
        "两最弱文件（RMS≈178/220 LSB）需到 k=14——由于低 2 位恒 0，"
        "k=14 恰为逐位恒等，故 k=14 是这批数据在 4-网格存储下的“无损”边界。")

    add("\n### 3.2 合成基准（判据 B：真值信号 bin±2 能量双侧 |R−1|≤1%；"
        "并列硬存活下限 Δ/2 < A）\n")
    add("| 合成案例 | 幅度 A (LSB) | round 安全 k | trunc 安全 k | 硬存活 k (R>0) |")
    add("|---|---|---|---|---|")
    for case, amp in (("tone_A12000", 12000), ("tone_A188", 188),
                      ("tone_A47", 47), ("tone_A12", 12), ("tone_A3", 3),
                      ("tone_A1", 1), ("tone_weak_alone", 38),
                      ("two_tone", 12000), ("bandlimited_noise", 2000)):
        kr = safe_k(synth_rows, "round", case, two_sided=True)
        kt = safe_k(synth_rows, "trunc", case, two_sided=True)
        ks = survival_k(synth_rows, "round", case)
        add(f"| {case} | {amp} | {kr if kr is not None else '>14'} |"
            f" {kt if kt is not None else '>14'} |"
            f" {ks if ks is not None else '>14'} |")
    add("")
    add("- 满幅附近（A=12000，约 -8.7 dBFS）下限低（个位数位数即可），"
        "小信号下限随幅度抬升，直至 A < Δ/2 时被整段消灭（R=0）："
        "硬存活边界与 16 − log2(2A) 一致。")
    add("- 双侧判据比单向严：如 A=47 在 k=10 存活但 R≈1.66（方波化），"
        "属于失真，不计为安全。")

    add("\n## 4. 弱音可见性（合成双音：强音 12000，弱音 38 ≈ −50 dB，"
        "局部底噪判据 ≥ +10 dB）\n")
    add("| k | round (dB) | trunc (dB) |")
    add("|---|---|---|")
    for k in K_LIST:
        rr = next(r for r in synth_rows if r["case"] == "two_tone"
                  and r["k"] == k and r["mode"] == "round")
        tt = next(r for r in synth_rows if r["case"] == "two_tone"
                  and r["k"] == k and r["mode"] == "trunc")
        add(f"| {k} | {rr['weak_ratio_db']:.1f} | {tt['weak_ratio_db']:.1f} |")
    v16 = next(r for r in synth_rows if r["case"] == "two_tone"
               and r["k"] == 16 and r["mode"] == "round")["weak_ratio_db"]
    v4 = next(r for r in synth_rows if r["case"] == "two_tone"
              and r["k"] == 4 and r["mode"] == "round")["weak_ratio_db"]
    add(f"\n- 可见性随 k 单调丧失：k=16 约 {v16:.0f} dB → k=4 约 {v4:.0f} dB"
        f"（损失 {v16 - v4:.0f} dB，来自再量化噪声底抬升）。")
    add("- 注意（重要物理）：骑在强载波上的弱音并不会完全消失——"
        "round/trunc 的电平穿越时刻被弱音调制（类自然采样 PWM / Δ-Σ 的亚 LSB "
        "传递效应），弱音峰值几乎保持；丧失的是对底噪的可见余量。"
        "而无载波支撑的孤立弱音（A=38）在 Δ/2 ≥ 38（k ≤ 8）时被整段归零"
        "（见 3.2 表 tone_weak_alone 行与 tests/）。")

    add("\n## 5. 峰值漂移\n")
    max_shift = 0
    max_amp_dev = 0.0
    for rows in per_file.values():
        for r in rows:
            if r["k"] >= 12:
                max_shift = max(max_shift, abs(r["rng_bin_shift"]),
                                abs(r["vel_bin_shift"]))
                max_amp_dev = max(max_amp_dev, abs(r["peak_amp_db"]))
    add(f"- 实测数据在 k ≥ 12 时峰值位置漂移最大 {max_shift} bin，"
        f"峰值幅度偏差最大 {max_amp_dev:.2f} dB；"
        "大漂移集中在低 k（量化噪声重排 argmax），见 fig5。")

    add("\n## 6. 主要 caveat\n")
    add("- 16 位**存储**≠16 位**有效 ADC 精度**：本研究不推断硬件 ENOB；")
    add("- 再量化结果不代表更换真实低位 ADC（AGC/滤波/噪声色彩都会不同）；")
    add("- 离线分析，未做硬件实测；官方 JPG 仅作定性参考；")
    add(f"- 实测数据仅占满量程约 {fs_abs / 32768 * 100:.1f}%（≈10 位），"
        "安全下限结论依赖该幅度分布；")
    add("- 可见性/底噪使用局部环形中位数底噪口径（再量化失真稀疏，"
        "全局中位数在低 k 会塌缩导致虚高）。")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
