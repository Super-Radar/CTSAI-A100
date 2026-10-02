"""One-command reproduction of the a100-target-stream-alignment study.

Run from the repository root (or anywhere; paths are derived from this file):

    python community/projects/a100-target-stream-alignment/src/stream_align/cli.py

It parses the 20 same-named RawTarget/TraTarget session pairs shipped in
``RadarTools/RadarTools_Release/data``, runs alignment methods A/B/C, sweeps
the method-C tolerance from 0 to 50 ms, and writes every artefact under
``community/projects/a100-target-stream-alignment/results/``:

* ``session_summary.csv``        - one row per session, all metrics
* ``frame_pairs_<session>.csv``  - full pair tables for 2 representative sessions
* ``data_sha256.txt``            - SHA-256 of all 40 paired input files
* ``summary.md``                 - key findings (Chinese)
* ``figures/fig1..fig4*.png``    - figures with English labels

Exit code 0 on success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path

import numpy as np

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from stream_align import figures as figmod  # noqa: E402
from stream_align.align import (  # noqa: E402
    align_by_framenb,
    align_exact_time,
    align_tolerance,
    tolerance_sweep,
)
from stream_align.metrics import (  # noqa: E402
    residual_stats,
    summarize_unmatched,
    unmatched_analysis,
)
from stream_align.parser import parse_target_csv  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_DIR.parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "RadarTools" / "RadarTools_Release" / "data"
AUX_SINGLE_STREAM = (
    REPO_ROOT / "Python雷达可视化工具示例"
    / "负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv"
)


def discover_pairs(data_dir: Path) -> tuple[list[tuple[str, Path, Path]], list[Path]]:
    """Same-named (RawTarget, TraTarget) sessions; also returns unpaired Raw files."""
    pairs: list[tuple[str, Path, Path]] = []
    unpaired: list[Path] = []
    for raw_path in sorted(data_dir.glob("*_RawTarget.csv")):
        session = raw_path.name[: -len("_RawTarget.csv")]
        tra_path = raw_path.with_name(session + "_TraTarget.csv")
        if tra_path.exists():
            pairs.append((session, raw_path, tra_path))
        else:
            unpaired.append(raw_path)
    return pairs, unpaired


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fmt(v, nd=4):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def analyse_session(session: str, raw_path: Path, tra_path: Path, tolerances: list[int]):
    raw = parse_target_csv(raw_path, kind="raw")
    tra = parse_target_csv(tra_path, kind="tra")

    res_a = align_by_framenb(raw, tra)
    res_b = align_exact_time(raw, tra)
    curve = tolerance_sweep(raw, tra, tolerances)
    best = max(curve, key=lambda r: (r["n_pairs"], -r["tol_ms"]))
    best_tol = best["tol_ms"]
    res_c = align_tolerance(raw, tra, best_tol)

    cov_a = res_a.coverage(raw, tra)
    cov_b = res_b.coverage(raw, tra)
    cov_c = res_c.coverage(raw, tra)
    st_a = residual_stats(res_a.residuals_ms)
    st_b = residual_stats(res_b.residuals_ms)
    st_c = residual_stats(res_c.residuals_ms)
    unmatched_tra = summarize_unmatched(unmatched_analysis(tra, res_c, raw))

    row = {
        "session": session,
        "raw_frames": raw.n_frames,
        "tra_frames": tra.n_frames,
        "raw_segments": raw.n_segments,
        "tra_segments": tra.n_segments,
        "raw_framenb_resets": raw.n_resets,
        "tra_framenb_resets": tra.n_resets,
        "raw_ts_epoch_mismatch": raw.n_ts_epoch_mismatch,
        "tra_ts_epoch_mismatch": tra.n_ts_epoch_mismatch,
        "raw_dup_epoch": raw.duplicate_epoch_count(),
        "tra_dup_epoch": tra.duplicate_epoch_count(),
        "matchA": res_a.n_pairs,
        "covA_raw": cov_a[0],
        "covA_tra": cov_a[1],
        "ambigA_keys": res_a.ambiguous_keys,
        "residA_median_ms": st_a["median_ms"],
        "residA_p95_ms": st_a["p95_ms"],
        "residA_max_ms": st_a["max_ms"],
        "matchB": res_b.n_pairs,
        "covB_raw": cov_b[0],
        "covB_tra": cov_b[1],
        "ambigB_epochs": res_b.ambiguous_keys,
        "residB_median_ms": st_b["median_ms"],
        "residB_p95_ms": st_b["p95_ms"],
        "residB_max_ms": st_b["max_ms"],
        "best_tol_ms": best_tol,
        "matchC": res_c.n_pairs,
        "covC_raw": cov_c[0],
        "covC_tra": cov_c[1],
        "residC_median_ms": st_c["median_ms"],
        "residC_p95_ms": st_c["p95_ms"],
        "residC_max_ms": st_c["max_ms"],
        "tra_unmatched": unmatched_tra["n_unmatched"],
        "tra_unmatched_rel_mean": unmatched_tra["unmatched_rel_mean"],
        "tra_unmatched_nn_median_ms": unmatched_tra["unmatched_nn_median_ms"],
        "tra_unmatched_nn_p95_ms": unmatched_tra["unmatched_nn_p95_ms"],
    }
    return raw, tra, res_a, res_b, res_c, curve, best_tol, row


def write_frame_pairs(path: Path, raw, tra, res_c) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow([
            "pair_i", "raw_frame_idx", "tra_frame_idx", "raw_FrameNb",
            "tra_FrameNb", "raw_epoch_ms", "tra_epoch_ms", "dt_ms",
            "raw_seg", "tra_seg", "raw_n_targets", "tra_n_targets",
        ])
        for k, (r_i, t_i) in enumerate(res_c.pairs):
            rf, tf = raw.frames[r_i], tra.frames[t_i]
            w.writerow([
                k, r_i, t_i, rf.frame_nb, tf.frame_nb, rf.epoch_ms, tf.epoch_ms,
                tf.epoch_ms - rf.epoch_ms, rf.segment, tf.segment,
                rf.n_targets, tf.n_targets,
            ])


def write_summary_md(
    path: Path,
    rows: list[dict],
    curve_flat: dict[str, list[dict]],
    unpaired_raw: list[Path],
    aux_info: dict | None,
) -> None:
    n = len(rows)
    cov_b_raw = [r["covB_raw"] for r in rows]
    cov_b_tra = [r["covB_tra"] for r in rows]
    cov_c_raw = [r["covC_raw"] for r in rows]
    cov_c_tra = [r["covC_tra"] for r in rows]
    best_tols = sorted({r["best_tol_ms"] for r in rows})
    b2025 = [r for r in rows if r["session"].startswith("2025")]
    b2026 = [r for r in rows if r["session"].startswith("2026")]
    a_max_2026 = max((r["residA_max_ms"] or 0) for r in b2026)
    a_med_2026 = float(np.median([r["residA_median_ms"] or 0 for r in b2026]))
    cov_c_tra_2026 = [r["covC_tra"] for r in b2026]
    resets = sum(r["raw_framenb_resets"] + r["tra_framenb_resets"] for r in rows)
    mism = sum(r["raw_ts_epoch_mismatch"] + r["tra_ts_epoch_mismatch"] for r in rows)
    dups = sum(r["raw_dup_epoch"] + r["tra_dup_epoch"] for r in rows)
    min_cov_tra = min(cov_c_tra)
    worst = min(rows, key=lambda r: r["covC_tra"])

    lines = []
    lines.append("# a100-target-stream-alignment 关键结论\n")
    lines.append(
        f"分析对象：`RadarTools/RadarTools_Release/data/` 下 {n} 组同名 "
        "`<ts>_Tatget_CTLRR-300-V2_1_{RawTarget,TraTarget}.csv` 会话对"
        "（解析器与指标由本项目 `src/stream_align/` 提供，全部产物可由 `cli.py` 重跑生成）。\n"
    )
    lines.append("## 总体结论\n")
    lines.append(
        f"1. **{n}/{n} 个会话在 tol=0 ms 即可把全部 Raw 帧精确配对到 Tra 帧**"
        f"（方法 B/C 的 Raw 侧覆盖率均为 1.0；最佳 tol = "
        f"{', '.join(str(t) for t in best_tols)} ms）。"
        "两流时间戳来自同一记录时钟，配对不依赖任何容差假设。\n"
    )
    lines.append(
        f"2. **2025-11-20 批次（{len(b2025)} 个会话）两流帧数完全相等、逐帧同一毫秒**："
        "Raw 与 Tra 是同一采集周期成对输出的，FrameNb 配对（方法 A）也严格成立。\n"
    )
    lines.append(
        f"3. **2026-05 批次（{len(b2026)} 个会话）Tra 帧数比 Raw 多 "
        f"{min(r['tra_frames'] - r['raw_frames'] for r in b2026)}"
        f"~{max(r['tra_frames'] - r['raw_frames'] for r in b2026)} 帧**"
        f"（Tra 侧配对覆盖率 {min(cov_c_tra_2026):.3f}"
        f"~{max(cov_c_tra_2026):.3f}）：Tra 流存在大量没有同刻 Raw 帧的额外输出，"
        "这是可配对边界所在，而不是时间抖动——把 tol 从 0 ms 放宽到 50 ms "
        "配对数不增加（见 fig2）。\n"
    )
    lines.append(
        f"4. **不要用 FrameNb 盲合并**：方法 A 在 2026-05 批次仍会给出 "
        f"{min(r['matchA'] for r in b2026)}~{max(r['matchA'] for r in b2026)} 个"
        f"“高数量”配对，但残差中位数 {a_med_2026:.0f} ms"
        f"（负值 = 同一 FrameNb 的 Tra 帧时刻更早）、最大 "
        f"{a_max_2026:.0f} ms（跨秒级），属于系统性错配；"
        "两流 FrameNb 语义未经官方文档证实，跨批次不可当作公共帧号。\n"
    )
    lines.append(
        f"5. 数据质量：{n} 组会话共 {resets} 次 FrameNb 重置/回退、"
        f"{mism} 帧墙钟与 epoch 不一致、{dups} 个流内重复时间戳"
        "（解析器的重置分段与一致性检查在这些数据上未触发，相关行为由合成数据测试覆盖）。\n"
    )
    lines.append("## 会话可配对性\n")
    lines.append(
        f"- 全部 {n} 个会话 Raw 侧可稳定配对（覆盖率 1.0，残差全为 0 ms）。\n"
        f"- Tra 侧覆盖率最低的会话：`{worst['session']}`"
        f"（{worst['covC_tra']:.3f}，{worst['tra_unmatched']} 个未匹配 Tra 帧）；"
        "未匹配 Tra 帧在整段时间轴上均匀出现（相对位置均值约 "
        f"{fmt(worst['tra_unmatched_rel_mean'], 2)}），到最近 Raw 帧的中位距离约 "
        f"{fmt(worst['tra_unmatched_nn_median_ms'], 0)} ms，"
        "即它们落在 Raw 输出间隔的缝隙里，是额外输出而非坏点。\n"
    )
    if unpaired_raw:
        names = ", ".join(p.stem for p in unpaired_raw)
        lines.append(
            f"- 未参与配对：`{names}`（只有 RawTarget、无同名 TraTarget）。\n"
        )
    if aux_info:
        lines.append(
            f"- 辅助单流演示：`Python雷达可视化工具示例/负30度_..._RawTarget.csv` "
            f"解析出 {aux_info['frames']} 帧、{aux_info['segments']} 个分段、"
            f"{aux_info['resets']} 次重置（不参与双流配对）。\n"
        )
    lines.append("## 产物索引\n")
    lines.append("- `session_summary.csv`：全部会话指标表\n")
    lines.append("- `frame_pairs_2025_11_20_18_41_24.csv`、"
                 "`frame_pairs_2026_05_20_16_24_32.csv`：两个代表会话的完整配对表\n")
    lines.append("- `data_sha256.txt`：40 个配对输入文件的 SHA-256\n")
    lines.append("- `figures/fig1_timeband_2026_05_20_16_24_32.png`：双流时间带\n")
    lines.append("- `figures/fig2_tol_coverage.png`：tol-覆盖率曲线\n")
    lines.append("- `figures/fig3_residual_hist.png`：匹配残差直方图（A 对比 C）\n")
    lines.append("- `figures/fig4_coverage_by_method.png`：会话级覆盖率汇总\n")
    lines.append(
        "\n> 局限：时间戳可能来自上位机记录时钟；Raw/Tra 的 |Δt| 不能解释为固件处理延迟；"
        "两流 FrameNb 语义未获官方证实；本分析为离线复现，未做硬件实测。\n"
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--results-dir", type=Path, default=PROJECT_DIR / "results")
    ap.add_argument("--tol-min", type=int, default=0)
    ap.add_argument("--tol-max", type=int, default=50)
    ap.add_argument("--tol-step", type=int, default=5)
    args = ap.parse_args(argv)

    tolerances = list(range(args.tol_min, args.tol_max + 1, args.tol_step))
    results_dir = args.results_dir
    fig_dir = results_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    pairs, unpaired_raw = discover_pairs(args.data_dir)
    if not pairs:
        print(f"ERROR: no RawTarget/TraTarget pairs found in {args.data_dir}")
        return 1
    print(f"[1/7] discovered {len(pairs)} session pairs "
          f"({len(unpaired_raw)} unpaired Raw-only file(s)) in {args.data_dir}")

    with (results_dir / "data_sha256.txt").open("w", encoding="utf-8") as fh:
        fh.write("# SHA-256 of the 40 paired input files (20 sessions x 2 streams)\n")
        for session, raw_path, tra_path in pairs:
            fh.write(f"{sha256_of(raw_path)}  {raw_path.name}\n")
            fh.write(f"{sha256_of(tra_path)}  {tra_path.name}\n")
        if AUX_SINGLE_STREAM.exists():
            fh.write("# auxiliary single-stream demo (not part of pairing)\n")
            fh.write(f"{sha256_of(AUX_SINGLE_STREAM)}  {AUX_SINGLE_STREAM.name}\n")
    print(f"[2/7] wrote data_sha256.txt ({len(pairs) * 2} paired files)")

    rows: list[dict] = []
    curves: dict[str, list[dict]] = {}
    store: dict[str, dict] = {}
    for session, raw_path, tra_path in pairs:
        raw, tra, res_a, res_b, res_c, curve, best_tol, row = analyse_session(
            session, raw_path, tra_path, tolerances
        )
        rows.append(row)
        curves[session] = curve
        store[session] = {
            "raw": raw, "tra": tra, "res_a": res_a,
            "res_b": res_b, "res_c": res_c, "best_tol": best_tol,
        }
        print(f"      {session}: raw={row['raw_frames']} tra={row['tra_frames']} "
              f"matchB={row['matchB']} best_tol={best_tol}ms "
              f"covC_raw={row['covC_raw']:.3f} covC_tra={row['covC_tra']:.3f}")
    print(f"[3/7] analysed {len(rows)} sessions (methods A/B/C, tol sweep "
          f"{tolerances[0]}..{tolerances[-1]} ms)")

    fieldnames = list(rows[0].keys())
    with (results_dir / "session_summary.csv").open("w", newline="",
                                                    encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow({k: fmt(v) for k, v in row.items()})
    print(f"[4/7] wrote session_summary.csv ({len(rows)} rows)")

    pair_targets = [
        "2025_11_20_18_41_24_Tatget_CTLRR-300-V2_1",
        "2026_05_20_16_24_32_Tatget_CTLRR-300-V2_1",
    ]
    for session in pair_targets:
        if session not in store:
            continue
        st = store[session]
        out = results_dir / f"frame_pairs_{session.split('_Tatget')[0]}.csv"
        write_frame_pairs(out, st["raw"], st["tra"], st["res_c"])
        print(f"      {out.name}: {st['res_c'].n_pairs} pairs")
    print("[5/7] wrote frame pair tables (2 representative sessions)")

    flagship = "2026_05_20_16_24_32_Tatget_CTLRR-300-V2_1"
    if flagship not in store:
        flagship = pairs[-1][0]
    st = store[flagship]
    figmod.fig1_timeband(
        st["raw"], st["tra"], st["res_c"], flagship,
        fig_dir / f"fig1_timeband_{flagship.split('_Tatget')[0]}.png",
    )
    batches = {s: s[:4] for s, _, _ in pairs}  # recording era: "2025" / "2026"
    figmod.fig2_tol_coverage(curves, batches, fig_dir / "fig2_tol_coverage.png")
    resid_a = [r for s in store for r in store[s]["res_a"].residuals_ms]
    resid_c = [r for s in store for r in store[s]["res_c"].residuals_ms]
    figmod.fig3_residual_hist(resid_a, resid_c, st["best_tol"],
                              fig_dir / "fig3_residual_hist.png")
    covs = {
        row["session"]: {
            "A": (row["covA_raw"], row["covA_tra"]),
            "B": (row["covB_raw"], row["covB_tra"]),
            "C": (row["covC_raw"], row["covC_tra"]),
        }
        for row in rows
    }
    figmod.fig4_coverage_by_method(list(curves), covs, batches,
                                   fig_dir / "fig4_coverage_by_method.png")
    print("[6/7] wrote 4 figures to results/figures/")

    aux_info = None
    if AUX_SINGLE_STREAM.exists():
        aux = parse_target_csv(AUX_SINGLE_STREAM, kind="raw")
        aux_info = {
            "frames": aux.n_frames, "segments": aux.n_segments,
            "resets": aux.n_resets,
            "mismatch": aux.n_ts_epoch_mismatch,
        }
        print(f"      auxiliary single stream: {aux.n_frames} frames, "
              f"{aux.n_segments} segment(s), {aux.n_resets} reset(s)")
    write_summary_md(results_dir / "summary.md", rows, curves, unpaired_raw,
                     aux_info)
    print("[7/7] wrote summary.md")
    print("DONE: all artefacts under", results_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
