"""Experiment orchestration for the ADC dropout study.

Five experiments (all seeded, all offline):

1. **Zero-perturbation regression** - every corruption model at severity 0 is
   an exact identity, and every metric is exactly 0 / 1.
2. **Synthetic ground truth** - a single-target frame (init0 parameters,
   known range and integer-bin Doppler) corrupted at increasing severities;
   medians and IQRs over >= 20 random repetitions per point.
3. **Real-data degradation** - the shipped far-wave (init0) and near-wave
   (init1) 4-channel captures corrupted the same way; the far wave runs the
   full grid, the near wave a reduced grid (smaller frame, coarser sweep) to
   keep total runtime within ~5 minutes.
4. **Actionable thresholds** - per model and strategy, the largest severity
   whose medians still satisfy "peak drift < 1 bin AND |peak power error|
   < 1 dB", plus a linearly interpolated crossing where the grid brackets it.
5. **Relation to format-level auditing** - a static mapping (written to
   ``results/summary.md``) of which models a length audit in the style of the
   in-review ``a100-adc-audit`` project would catch, and which pass silently.

Outputs: ``results/*.csv``, ``results/summary.md``, ``figures/*.png``.
"""

from __future__ import annotations

import csv
import hashlib
import time
from pathlib import Path

import numpy as np

from . import metrics as M
from . import plots
from .corruption import MODELS, STRATEGIES, chirp_keep_mask, corrupt_words
from .pipeline import (
    INIT0,
    INIT1,
    RadarConfig,
    load_adc_words,
    rv_map_average,
    synth_single_target_words,
    unpack_words,
)

BASE_SEED = 20260929
SYNTH_RANGE_M = 5.0
SYNTH_DOPPLER_BIN = 12  # exact integer Doppler bin -> velocity is an exact multiple of vel_per_bin

FULL_GRIDS = {
    "random_drop": [0.0005, 0.001, 0.0025, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2],
    "zero_segment": [1, 2, 4, 8, 16, 32, 64],
    "chirp_zero": [1, 2, 4, 8, 16, 32],
    "tail_trunc": [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
}
NEAR_GRIDS = {
    "random_drop": [0.001, 0.005, 0.01, 0.05, 0.2],
    "zero_segment": [4, 16, 64],
    "chirp_zero": [1, 4, 16],
    "tail_trunc": [0.01, 0.05, 0.1],
}
QUICK_GRIDS = {
    "random_drop": [0.001, 0.01, 0.05, 0.2],
    "zero_segment": [1, 8, 64],
    "chirp_zero": [1, 8, 32],
    "tail_trunc": [0.005, 0.03, 0.1],
}
POSITION_LENS = (1, 4, 16, 64)

DATASET_LABEL = {
    "synthetic-init0": "合成单目标帧 (init0)",
    "real-far-init0": "真实远波 4 通道 (init0)",
    "real-near-init1": "真实近波 4 通道 (init1)",
}


# --------------------------------------------------------------------------- helpers

def rep_seed(label: str, model: str, sev, rep: int) -> int:
    key = f"{BASE_SEED}|{label}|{model}|{sev!r}|{rep}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


def reference_map(base_words, cfg: RadarConfig):
    return rv_map_average([unpack_words(w) for w in base_words], cfg)


def evaluate_instance(base_words, cfg, ref, model, sev, rep, label,
                      strategies=STRATEGIES, position="middle"):
    """Corrupt once, then evaluate every strategy on the same damaged stream."""
    rng = np.random.default_rng(rep_seed(label, model, sev, rep))
    corrupted = [corrupt_words(w, model, sev, rng, position) for w in base_words]
    out = []
    for strat in strategies:
        if strat == "reject":
            keep = chirp_keep_mask([m for _, m in corrupted])
            if not keep.any():
                out.append((strat, True, 0, None))
                continue
            frames = [unpack_words(w[keep]) for w, _ in corrupted]
        else:
            frames = [unpack_words(w) for w, _ in corrupted]
        mp = rv_map_average(frames, cfg)
        out.append((strat, False, int(frames[0].shape[0]), M.compare_maps(ref, mp)))
    return out


def run_sweep(base_words, cfg, label, grids, strategies, n_rep, ref, position="middle"):
    rows = []
    for model in MODELS:
        for sev in grids[model]:
            for rep in range(n_rep):
                for strat, unusable, nch, mm in evaluate_instance(
                    base_words, cfg, ref, model, sev, rep, label, strategies, position
                ):
                    rows.append({
                        "dataset": label,
                        "model": model,
                        "position": position if model == "zero_segment" else "",
                        "severity": sev,
                        "strategy": strat,
                        "rep": rep,
                        "unusable": unusable,
                        "nchirp_used": nch,
                        **(mm if mm is not None else {k: np.nan for k in M.METRIC_KEYS}),
                    })
    return rows


def _med(vals):
    vals = [v for v in vals if v is not None]
    return float(np.median(vals)) if vals else None


def _med_iqr(vals):
    if not vals:
        return None, None, None
    q25, med, q75 = np.percentile(vals, [25, 50, 75])
    return float(med), float(q25), float(q75)


def aggregate(rows):
    """Median / IQR per (dataset, model, position, strategy, severity)."""
    groups: dict = {}
    for r in rows:
        key = (r["dataset"], r["model"], r["position"], r["strategy"], r["severity"])
        groups.setdefault(key, []).append(r)
    agg = []
    def sort_key(kv):
        ds, model, pos, strat, sev = kv[0]
        return (ds, MODELS.index(model), pos, strat, float(sev))
    for (ds, model, pos, strat, sev), rs in sorted(groups.items(), key=sort_key):
        out = {
            "dataset": ds, "model": model, "position": pos, "strategy": strat,
            "severity": sev, "n_rep": len(rs),
            "n_unusable": sum(1 for r in rs if r["unusable"]),
            "nchirp_used_med": _med([r["nchirp_used"] for r in rs if not r["unusable"]]),
        }
        for k in M.METRIC_KEYS:
            vals = [r[k] for r in rs if not r["unusable"] and np.isfinite(r[k])]
            med, q25, q75 = _med_iqr(vals)
            out[f"{k}_med"] = med
            out[f"{k}_q25"] = q25
            out[f"{k}_q75"] = q75
        agg.append(out)
    return agg


AGG_FIELDS = (
    ["dataset", "model", "position", "strategy", "severity", "n_rep",
     "n_unusable", "nchirp_used_med"]
    + [f"{k}_{s}" for k in M.METRIC_KEYS for s in ("med", "q25", "q75")]
)


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore",
                           lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


# --------------------------------------------------------------------------- exp 1

def exp1_zero_perturbation(base_words, cfg, ref, label):
    rows = []
    for model in MODELS:
        corrupted = [corrupt_words(w, model, 0, np.random.default_rng(1)) for w in base_words]
        for w_base, (w2, mask) in zip(base_words, corrupted):
            if not (np.array_equal(w2, w_base) and not mask.any()):
                raise AssertionError(f"{model}: severity 0 is not an identity")
        for strat in STRATEGIES:
            frames = [unpack_words(w2) for w2, _ in corrupted]
            mp = rv_map_average(frames, cfg)
            mm = M.compare_maps(ref, mp)
            ok = (
                mm["drift_max_bins"] == 0.0
                and abs(mm["peak_power_error_db"]) <= 1e-9
                and abs(mm["total_energy_error_db"]) <= 1e-9
                and abs(mm["spectral_correlation"] - 1.0) <= 1e-12
                and mm["doppler_spurious_delta"] == 0
            )
            rows.append({"dataset": label, "model": model, "strategy": strat,
                         "severity": 0, "passed": bool(ok), **mm})
    return rows


# --------------------------------------------------------------------------- exp 4

def _crossing(sevs, vals, level):
    """(largest severity still below level, interpolated boundary or None)."""
    ok = [v is not None and np.isfinite(v) and v < level for v in vals]
    if all(ok):
        return sevs[-1], None  # never crossed within the tested range
    i0 = next(i for i, o in enumerate(ok) if not o)
    last_ok = sevs[i0 - 1] if i0 > 0 else None
    bound = None
    if i0 > 0 and vals[i0 - 1] is not None and vals[i0] is not None:
        y0, y1 = float(vals[i0 - 1]), float(vals[i0])
        if np.isfinite(y0) and np.isfinite(y1) and y1 != y0:
            bound = sevs[i0 - 1] + (level - y0) * (sevs[i0] - sevs[i0 - 1]) / (y1 - y0)
    return last_ok, bound


def threshold_rows(agg, dataset):
    """Largest severity meeting 'drift < 1 bin AND |peak power err| < 1 dB' (medians)."""
    rows = []
    for model in MODELS:
        for strat in STRATEGIES:
            rs = [r for r in agg
                  if r["dataset"] == dataset and r["model"] == model
                  and r["strategy"] == strat]
            rs.sort(key=lambda r: float(r["severity"]))
            if not rs:
                continue
            sevs = [float(r["severity"]) for r in rs]
            pow_abs = [abs(r["peak_power_error_db_med"])
                       if r["peak_power_error_db_med"] is not None else None for r in rs]
            drt = [r["drift_max_bins_med"] for r in rs]
            both = [p is not None and d is not None and p < 1.0 and d < 1.0
                    for p, d in zip(pow_abs, drt)]
            max_pass = None
            for s, b in zip(sevs, both):
                if b:
                    max_pass = s
            _, bound_pow = _crossing(sevs, pow_abs, 1.0)
            _, bound_drt = _crossing(sevs, drt, 1.0)
            if all(both):
                note = "all tested severities pass"
            elif not any(both):
                note = "no tested severity passes"
            else:
                note = ""
            rows.append({
                "dataset": dataset, "model": model, "strategy": strat,
                "max_pass_severity": max_pass,
                "power_1db_interp": bound_pow,
                "drift_1bin_interp": bound_drt,
                "n_rep": rs[0]["n_rep"],
                "note": note,
            })
    return rows


THRESHOLD_FIELDS = ["dataset", "model", "strategy", "max_pass_severity",
                    "power_1db_interp", "drift_1bin_interp", "n_rep", "note"]

# --------------------------------------------------------------------------- exp 5 (static)

AUDIT_TABLE_MD = """\
| 缺损模型 | 长度审计（a100-adc-audit 式） | 内容级启发式 | 本项目结论 |
|---|---|---|---|
| 尾部截断（丢 ≥1 字） | **可捕获**：文件值数 ≠ 131073（131072 载荷 + 1 尾零惯例） | 无需 | 任何尾部丢字都会改变文件长度 |
| 随机丢字（按位补零） | 静默通过：长度不变 | 散布单字零值难以从数据侧发现 | 0.5% 丢字已带来可测谱退化 |
| 连续零段（1–64 字） | 静默通过：长度不变 | 较长全零游程可被"全零游程"规则捕获 | 位置影响大于长度（头/尾 vs 中） |
| 整 chirp 置零 | 静默通过：长度不变 | "整 chirp 全零"规则可稳定捕获 | 8 chirp 起峰功率损失明显 |

注：长度/哈希/饱和等格式级规则引自在审社区项目 `community/projects/a100-adc-audit` 的思路，本项目不重复实现；
哈希比对只能发现"同一文件两份拷贝不一致"，对首次采集的新数据没有基线可比。谱级影响请参考本项目的退化曲线与阈值表。\
"""


# --------------------------------------------------------------------------- summary

def _fmt(v, fmt="{:g}"):
    return "n/a" if v is None else fmt.format(v)


def _threshold_md(thr, dataset):
    lines = [
        f"### {DATASET_LABEL.get(dataset, dataset)}（{dataset}）",
        "",
        "| 缺损模型 | 应对策略 | 最大可通过强度 | 峰功率 1 dB 插值边界 | 峰漂移 1 bin 插值边界 | 备注 |",
        "|---|---|---|---|---|---|",
    ]
    for r in thr:
        if r["dataset"] != dataset:
            continue
        lines.append(
            "| {} | {} | {} | {} | {} | {} |".format(
                r["model"], r["strategy"],
                _fmt(r["max_pass_severity"]),
                _fmt(r["power_1db_interp"], "{:.5g}"),
                _fmt(r["drift_1bin_interp"], "{:.4g}"),
                r["note"],
            )
        )
    return "\n".join(lines)


def _key_numbers(agg_far, thr):
    """A few headline numbers for summary.md, the CLI printout and the README."""
    def get(model, strat, sev, key):
        for r in agg_far:
            if (r["model"] == model and r["strategy"] == strat
                    and abs(float(r["severity"]) - sev) < 1e-12):
                return r[f"{key}_med"]
        return None
    return {
        "far_random_05pct_power_db": get("random_drop", "zerofill", 0.005, "peak_power_error_db"),
        "far_random_1pct_power_db": get("random_drop", "zerofill", 0.01, "peak_power_error_db"),
        "far_chirp8_power_db": get("chirp_zero", "zerofill", 8, "peak_power_error_db"),
        "far_tail5_zerofill_power_db": get("tail_trunc", "zerofill", 0.05, "peak_power_error_db"),
        "far_tail5_reject_power_db": get("tail_trunc", "reject", 0.05, "peak_power_error_db"),
        "far_tail5_zerofill_drift": get("tail_trunc", "zerofill", 0.05, "drift_max_bins"),
        "far_tail5_reject_drift": get("tail_trunc", "reject", 0.05, "drift_max_bins"),
        "thresholds": thr,
    }


def write_summary(path, elapsed_s, exp1_rows, thr, agg_far, seeds_info):
    kn = _key_numbers(agg_far, thr)
    md = []
    md.append("# a100-dropout-study 结果摘要（自动生成）\n")
    md.append(f"- 生成命令：`PYTHONPATH=src python3 -m dropout_study.cli`（固定种子，可复现）")
    md.append(f"- 总耗时：{elapsed_s:.1f} s；随机基础种子：{BASE_SEED}")
    md.append(f"- 合成帧：init0，单目标 R={SYNTH_RANGE_M:g} m，多普勒 bin={SYNTH_DOPPLER_BIN}"
              f"（v={SYNTH_DOPPLER_BIN * INIT0.vel_per_bin_mps:.4f} m/s）")
    md.append(f"- 数据：{seeds_info}\n")
    n_pass = sum(1 for r in exp1_rows if r["passed"])
    md.append(f"## 实验 1：零扰动回归\n")
    md.append(f"- {n_pass}/{len(exp1_rows)} 项全部通过：severity=0 时字流恒等，"
              "峰位漂移=0、峰功率误差=0 dB、总能量误差=0 dB、谱相关=1、杂散峰增量=0。\n")
    md.append("## 实验 4：可操作阈值（判定：中位数峰位漂移 <1 bin 且 |峰功率误差| <1 dB）\n")
    for ds in ("synthetic-init0", "real-far-init0", "real-near-init1"):
        md.append(_threshold_md(thr, ds))
        md.append("")
    md.append("## 关键数值（真实远波，4 通道平均）\n")
    md.append(f"- 随机丢字 0.5% / 1%（保位补零）：中位数峰功率误差 "
              f"{_fmt(kn['far_random_05pct_power_db'], '{:.2f}')} / "
              f"{_fmt(kn['far_random_1pct_power_db'], '{:.2f}')} dB")
    md.append(f"- 整 chirp 置零 8 个（保位补零）：中位数峰功率误差 "
              f"{_fmt(kn['far_chirp8_power_db'], '{:.2f}')} dB")
    md.append(f"- 尾部截断 5%：保位补零 {(_fmt(kn['far_tail5_zerofill_power_db'], '{:.2f}'))} dB / "
              f"峰漂移 {_fmt(kn['far_tail5_zerofill_drift'], '{:.2f}')} bin；"
              f"截断到完整 chirp {(_fmt(kn['far_tail5_reject_power_db'], '{:.2f}'))} dB / "
              f"峰漂移 {_fmt(kn['far_tail5_reject_drift'], '{:.2f}')} bin\n")
    md.append("## 实验 5：与格式级审计的关系\n")
    md.append(AUDIT_TABLE_MD)
    md.append("\n## 诚实性声明\n")
    md.append("人工注入的缺损模型不代表真实采集链路的故障分布；以上结论仅限于"
              "“受控缺损模型 + 官方 MATLAB 链路复现”条件下的谱退化规律。详见项目 README 的“局限”一节。")
    path.write_text("\n".join(md) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- main

def default_data_dir() -> Path:
    return Path(__file__).resolve().parents[5] / "ADC数据采集" / "示例adc数据和结果"


def _example_maps(far_words, ref_far):
    def build(model, sev, strat, position="middle"):
        rng = np.random.default_rng(rep_seed("fig4", model, sev, 0))
        corrupted = [corrupt_words(w, model, sev, rng, position) for w in far_words]
        if strat == "reject":
            keep = chirp_keep_mask([m for _, m in corrupted])
            frames = [unpack_words(w[keep]) for w, _ in corrupted]
            extra = f"{int(keep.sum())} chirps kept"
        else:
            frames = [unpack_words(w) for w, _ in corrupted]
            extra = "zero-fill"
        mp = rv_map_average(frames, INIT0)
        mm = M.compare_maps(ref_far, mp)
        return mp, f"ΔP={mm['peak_power_error_db']:+.2f} dB, drift={mm['drift_max_bins']:.2f} bin ({extra})"

    mp_seg, note_seg = build("zero_segment", 64, "zerofill")
    mp_chirp, note_chirp = build("chirp_zero", 16, "zerofill")
    mp_tf, note_tf = build("tail_trunc", 0.05, "zerofill")
    mp_tr, note_tr = build("tail_trunc", 0.05, "reject")
    return [
        ("clean baseline", "reference (4-channel average)", ref_far),
        ("zero segment, 64 words", note_seg, mp_seg),
        ("16 chirps zeroed", note_chirp, mp_chirp),
        ("tail 5% -> zero-fill", note_tf, mp_tf),
        ("tail 5% -> reject", note_tr, mp_tr),
    ]


def run_all(project_dir=None, quick=False, data_dir=None):
    t0 = time.time()
    project_dir = Path(project_dir) if project_dir else Path(__file__).resolve().parents[2]
    results = project_dir / "results"
    figures = project_dir / "figures"
    results.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    data_dir = Path(data_dir) if data_dir else default_data_dir()
    far_files = sorted(data_dir.glob("adc_远波_*_f0_*.txt"))
    near_files = sorted(data_dir.glob("adc_近波_*_f1_*.txt"))
    if len(far_files) < 4 or len(near_files) < 4:
        raise FileNotFoundError(
            f"expected 4 far-wave and 4 near-wave ADC txt files under {data_dir}, "
            f"found {len(far_files)} / {len(near_files)}"
        )

    n_rep_synth, n_rep_far, n_rep_near, n_rep_pos = (20, 20, 10, 20)
    grid_synth = grid_far = FULL_GRIDS
    grid_near = NEAR_GRIDS
    if quick:
        n_rep_synth, n_rep_far, n_rep_near, n_rep_pos = (5, 5, 3, 5)
        grid_synth = grid_far = grid_near = QUICK_GRIDS

    print(f"[study] data dir: {data_dir}")
    far_words, far_reports = [], []
    for p in far_files:
        w, rep = load_adc_words(p, INIT0)
        far_words.append(w)
        far_reports.append(rep)
    near_words, near_reports = [], []
    for p in near_files:
        w, rep = load_adc_words(p, INIT1)
        near_words.append(w)
        near_reports.append(rep)
    write_csv(results / "load_report.csv",
              [r.as_dict() for r in far_reports + near_reports],
              ["path", "sha256", "n_values", "expected_words",
               "n_extra_truncated", "n_zero_padded", "notes"])
    print(f"[study] loaded {len(far_words)} far + {len(near_words)} near channels "
          f"({time.time() - t0:.1f} s)")

    synth = [synth_single_target_words(
        INIT0, SYNTH_RANGE_M, SYNTH_DOPPLER_BIN * INIT0.vel_per_bin_mps)]
    ref_synth = reference_map(synth, INIT0)
    ref_far = reference_map(far_words, INIT0)
    ref_near = reference_map(near_words, INIT1)

    # ---- experiment 1: zero-perturbation regression -------------------------
    exp1 = (exp1_zero_perturbation(synth, INIT0, ref_synth, "synthetic-init0")
            + exp1_zero_perturbation(far_words, INIT0, ref_far, "real-far-init0"))
    if not all(r["passed"] for r in exp1):
        raise AssertionError("zero-perturbation regression failed")
    write_csv(results / "exp1_zero_perturbation.csv", exp1,
              ["dataset", "model", "strategy", "severity", "passed"] + M.METRIC_KEYS)
    print(f"[study] exp1 zero-perturbation: {len(exp1)}/{len(exp1)} checks passed "
          f"({time.time() - t0:.1f} s)")

    # ---- experiment 2: synthetic sweep --------------------------------------
    rows_synth = run_sweep(synth, INIT0, "synthetic-init0", grid_synth,
                           STRATEGIES, n_rep_synth, ref_synth)
    agg_synth = aggregate(rows_synth)
    write_csv(results / "synthetic_init0_degradation.csv", agg_synth, AGG_FIELDS)
    print(f"[study] exp2 synthetic sweep: {len(rows_synth)} evaluations "
          f"({time.time() - t0:.1f} s)")

    # ---- experiment 3: real-data sweeps -------------------------------------
    rows_far = run_sweep(far_words, INIT0, "real-far-init0", grid_far,
                         STRATEGIES, n_rep_far, ref_far)
    agg_far = aggregate(rows_far)
    write_csv(results / "real_far_init0_degradation.csv", agg_far, AGG_FIELDS)
    print(f"[study] exp3 far sweep: {len(rows_far)} evaluations ({time.time() - t0:.1f} s)")

    rows_near = run_sweep(near_words, INIT1, "real-near-init1", grid_near,
                          STRATEGIES, n_rep_near, ref_near)
    agg_near = aggregate(rows_near)
    write_csv(results / "real_near_init1_degradation.csv", agg_near, AGG_FIELDS)
    print(f"[study] exp3 near sweep: {len(rows_near)} evaluations ({time.time() - t0:.1f} s)")

    # ---- zero-segment position mini-study ------------------------------------
    rows_pos = []
    for pos in ("head", "middle", "tail"):
        for length in POSITION_LENS:
            for rep in range(n_rep_pos):
                for strat, unusable, nch, mm in evaluate_instance(
                    synth, INIT0, ref_synth, "zero_segment", length, rep,
                    f"synthetic-init0-pos-{pos}", ("zerofill",), position=pos
                ):
                    rows_pos.append({
                        "dataset": "synthetic-init0", "model": "zero_segment",
                        "position": pos, "severity": length, "strategy": strat,
                        "rep": rep, "unusable": unusable, "nchirp_used": nch,
                        **(mm if mm is not None else {k: np.nan for k in M.METRIC_KEYS}),
                    })
    agg_pos = aggregate(rows_pos)
    write_csv(results / "synthetic_position_study.csv", agg_pos, AGG_FIELDS)
    print(f"[study] position study: {len(rows_pos)} evaluations ({time.time() - t0:.1f} s)")

    # ---- experiment 4: thresholds --------------------------------------------
    thr = (threshold_rows(agg_synth, "synthetic-init0")
           + threshold_rows(agg_far, "real-far-init0")
           + threshold_rows(agg_near, "real-near-init1"))
    write_csv(results / "thresholds.csv", thr, THRESHOLD_FIELDS)

    # ---- figures --------------------------------------------------------------
    plots.fig_power_curves(
        agg_synth, "synthetic-init0", MODELS, figures / "fig1_synthetic_power.png",
        "合成单目标帧（init0）：峰功率误差 vs 缺损强度（中位数，IQR 带）")
    plots.fig_power_curves(
        agg_far, "real-far-init0", MODELS, figures / "fig2_real_far_power.png",
        "真实远波 4 通道（init0）：峰功率误差 vs 缺损强度（中位数，IQR 带）")
    plots.fig_drift_spurious(
        agg_synth, "synthetic-init0", agg_far, "real-far-init0", MODELS,
        figures / "fig3_drift_spurious.png",
        "峰位漂移与速度维杂散峰（横轴：受影响字占全流比例）")
    plots.fig_example_maps(
        _example_maps(far_words, ref_far),
        figures / "fig4_example_maps.png",
        "真实远波数据：典型缺损的距离-多普勒功率图（dB，与洁净基线同色标）")
    plots.fig_position_study(
        agg_pos, figures / "fig5_position_effect.png",
        "合成单目标帧：连续零段位置（头/中/尾）对峰功率误差的影响")
    print(f"[study] figures written ({time.time() - t0:.1f} s)")

    # ---- summary ---------------------------------------------------------------
    elapsed = time.time() - t0
    write_summary(results / "summary.md", elapsed, exp1, thr, agg_far,
                  f"远波 {len(far_files)} 通道 + 近波 {len(near_files)} 通道（sha256 见 load_report.csv）")
    print(f"[study] done in {elapsed:.1f} s; results in {results}")

    return {
        "elapsed_s": elapsed,
        "exp1_passed": f"{len(exp1)}/{len(exp1)}",
        "key_numbers": _key_numbers(agg_far, thr),
        "thresholds": thr,
    }


__all__ = ["run_all", "default_data_dir", "BASE_SEED"]
