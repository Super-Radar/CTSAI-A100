# -*- coding: utf-8 -*-
"""一键复现入口。

用法（仓库根目录或任意目录均可，数据路径参数化、默认仓内路径）::

    /path/to/python -m rx_ablation.cli            # 全流程：数据 -> CSV/MD -> PNG
    /path/to/python -m rx_ablation.cli --data-dir /other/adc --out-dir /tmp/out

输出：
- ``figures/*.png``  7 张图
- ``results/*.csv``  子集扫描 / 单通道峰表 / 留一消融 / 相关矩阵 / 峰通道占比
- ``results/summary.md``  关键数值汇总
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
from typing import Dict, List

import numpy as np

from . import ablation, metrics, plots
from .pipeline import (
    DATASETS,
    DEFAULT_CFG_DIR,
    DEFAULT_DATA_DIR,
    RadarConfig,
    load_dataset,
    repo_root_from_module,
)

# 峰检测参数（显式、全流程统一；无隐藏默认值）
# 在仓内两组示例数据上的检出规模：far 25 峰 / near 47 峰（top_n=64 未触顶）
PEAK_MARGIN_DB = 20.0
PEAK_MIN_RNG_SEP = 4
PEAK_MIN_VEL_SEP = 4
MATCH_RNG_TOL = 3
MATCH_VEL_TOL = 3


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def signed_vel_bin(bin_idx: int, vel_nfft: int) -> int:
    """官方 Doppler FFT 未做 fftshift，bin > N/2 对应负频率。"""
    return ((bin_idx + vel_nfft // 2) % vel_nfft) - vel_nfft // 2


def analyze_official_jpg(path: str) -> Dict[str, float]:
    """对官方 JPG 做保守的结构统计（无法逐像素对拍，见 README）：

    - 亮簇数：彩色 mesh 像素中亮度前 0.5% 的 8 连通分量数（面积 >= 30 px）；
    - 红色标记像素数（官方脚本用 'ro' 标出最强目标）。
    """
    from PIL import Image
    im = np.asarray(Image.open(path).convert("RGB")).astype(float)
    R, G, B = im[..., 0], im[..., 1], im[..., 2]
    colored = (R < 240) | (G < 240) | (B < 240)
    lum = 0.299 * R + 0.587 * G + 0.114 * B
    vals = lum[colored]
    thr = np.percentile(vals, 99.5)
    bright = colored & (lum >= thr)
    from scipy import ndimage
    lbl, n = ndimage.label(bright, structure=np.ones((3, 3)))
    sizes = ndimage.sum(bright, lbl, range(1, n + 1)) if n else []
    n_clusters = int(sum(1 for s in sizes if s >= 30))
    red = int(((R > 150) & (G < 110) & (B < 110)).sum())
    return {"bright_clusters": n_clusters, "red_marker_px": red}


# ---------------------------------------------------------------------------
# CSV 写出
# ---------------------------------------------------------------------------

def _write_csv(path: str, header: List[str], rows: List[dict]):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=header, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def run(data_dir: str, cfg_dir: str, out_dir: str, make_figures: bool = True,
        jpg_dir: str | None = None) -> Dict[str, object]:
    project_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    fig_dir = os.path.join(project_dir, "figures")
    res_dir = os.path.join(project_dir, "results")
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(res_dir, exist_ok=True)

    frames, cfgs, all_out = {}, {}, {}
    for ds in ("far", "near"):
        info = DATASETS[ds]
        cfg = RadarConfig.from_hxx(os.path.join(cfg_dir, info["hxx"]))
        frame = load_dataset(data_dir, cfg, info["marker"])
        frames[ds], cfgs[ds] = frame, cfg

        # ---- 峰检测（全集） --------------------------------------------
        peaks = metrics.detect_peaks(
            frame["rv_image"], margin_db=PEAK_MARGIN_DB,
            min_rng_sep=PEAK_MIN_RNG_SEP, min_vel_sep=PEAK_MIN_VEL_SEP,
        )
        # ---- 实验 2：单通道画像 -----------------------------------------
        singles = ablation.single_channel_profiles(
            frame["power"], cfg, PEAK_MARGIN_DB, PEAK_MIN_RNG_SEP,
            PEAK_MIN_VEL_SEP,
        )
        # ---- 实验 3：15 子集扫描 ----------------------------------------
        subset_results, _ = ablation.scan_subsets(
            frame["power"], PEAK_MARGIN_DB, PEAK_MIN_RNG_SEP, PEAK_MIN_VEL_SEP,
            MATCH_RNG_TOL, MATCH_VEL_TOL,
        )
        # ---- 实验 4：留一消融 --------------------------------------------
        loo_results, loo_diffs = ablation.leave_one_out(
            frame["power"], PEAK_MARGIN_DB, PEAK_MIN_RNG_SEP, PEAK_MIN_VEL_SEP,
            MATCH_RNG_TOL, MATCH_VEL_TOL,
        )
        # ---- 实验 5：平均 vs 求和 ----------------------------------------
        avg_sum = ablation.avg_vs_sum(frame["power"])
        identity = ablation.subset_energy_identity(frame["power"])
        corr = metrics.channel_correlation(frame["power"], domain="db")
        corr_lin = metrics.channel_correlation(frame["power"], domain="linear")
        shares = metrics.peak_channel_shares(
            frame["power"], sorted(peaks, key=lambda p: -p.db)[:5]
        )

        # ---- CSV ---------------------------------------------------------
        _write_csv(
            os.path.join(res_dir, f"{ds}_subset_scan.csv"),
            ["dataset", "subset", "k", "strongest_rng_bin", "strongest_rng_m",
             "strongest_vel_bin", "signed_vel_bin", "strongest_db", "floor_db",
             "snr_db", "jitter_rng_bin", "jitter_vel_bin", "cheb_jitter_bin",
             "jaccard", "n_peaks", "total_power"],
            [
                {
                    "dataset": ds, "subset": r.label, "k": r.k,
                    "strongest_rng_bin": r.strongest_rng_bin,
                    "strongest_rng_m": round(r.strongest_rng_bin * cfg.rng_res, 3),
                    "strongest_vel_bin": r.strongest_vel_bin,
                    "signed_vel_bin": signed_vel_bin(r.strongest_vel_bin, cfg.vel_nfft),
                    "strongest_db": round(r.strongest_db, 2),
                    "floor_db": round(r.floor_db, 2),
                    "snr_db": round(r.snr_db, 2),
                    "jitter_rng_bin": r.jitter_rng, "jitter_vel_bin": r.jitter_vel,
                    "cheb_jitter_bin": metrics.chebyshev_jitter((r.jitter_rng, r.jitter_vel)),
                    "jaccard": round(r.jaccard, 4), "n_peaks": r.n_peaks,
                    "total_power": f"{r.total_power:.6e}",
                }
                for r in subset_results
            ],
        )
        peak_rows = []
        for s in singles:
            for rank, p in enumerate(s["peaks"], start=1):
                peak_rows.append({
                    "dataset": ds, "channel": s["channel"], "rank": rank,
                    "rng_bin": p.rng_bin, "rng_m": round(p.rng_bin * cfg.rng_res, 3),
                    "vel_bin": p.vel_bin,
                    "signed_vel_bin": signed_vel_bin(p.vel_bin, cfg.vel_nfft),
                    "db": round(p.db, 2),
                })
        _write_csv(
            os.path.join(res_dir, f"{ds}_single_channel_peaks.csv"),
            ["dataset", "channel", "rank", "rng_bin", "rng_m", "vel_bin",
             "signed_vel_bin", "db"], peak_rows,
        )
        _write_csv(
            os.path.join(res_dir, f"{ds}_leave_one_out.csv"),
            ["dataset", "dropped_rx", "n_peaks", "jaccard", "jitter_rng_bin",
             "jitter_vel_bin", "diff_max_db", "diff_min_db", "top1_delta_db",
             "top2_delta_db", "top3_delta_db", "top4_delta_db", "top5_delta_db"],
            [
                {
                    "dataset": ds, "dropped_rx": r.dropped, "n_peaks": r.n_peaks,
                    "jaccard": round(r.jaccard, 4),
                    "jitter_rng_bin": r.jitter_rng, "jitter_vel_bin": r.jitter_vel,
                    "diff_max_db": round(r.diff_max_db, 2),
                    "diff_min_db": round(r.diff_min_db, 2),
                    **{f"top{i+1}_delta_db": round(r.top_peak_deltas_db[i], 2)
                       for i in range(min(5, len(r.top_peak_deltas_db)))},
                }
                for r in loo_results
            ],
        )
        _write_csv(
            os.path.join(res_dir, f"{ds}_channel_correlation.csv"),
            ["dataset", "domain", "row"] + [f"rx{k}" for k in range(4)],
            [
                {"dataset": ds, "domain": dom, "row": k,
                 **{f"rx{j}": round(float(m[k, j]), 4) for j in range(4)}}
                for m, dom in ((corr, "db"), (corr_lin, "linear")) for k in range(4)
            ],
        )
        _write_csv(
            os.path.join(res_dir, f"{ds}_peak_channel_shares.csv"),
            ["dataset", "rng_bin", "vel_bin", "db", "rx0_share", "rx1_share",
             "rx2_share", "rx3_share"],
            [{"dataset": ds, "rng_bin": s["rng_bin"], "vel_bin": s["vel_bin"],
              "db": round(s["db"], 2),
              **{f"rx{k}_share": round(s[f"rx{k}_share"], 4) for k in range(4)}}
             for s in shares],
        )

        all_out[ds] = {
            "cfg": cfg, "peaks": peaks, "singles": singles,
            "subsets": subset_results, "loo": loo_results, "loo_diffs": loo_diffs,
            "avg_sum": avg_sum, "identity": identity, "corr": corr,
            "corr_lin": corr_lin, "shares": shares, "files": frame["files"],
            "sha256": [sha256_of(f) for f in frame["files"]],
        }
        if peaks:
            print(
                f"[{ds}] adc_sample_num={cfg.adc_sample_num} "
                f"rng_res={cfg.rng_res:.4f} m V_res={cfg.v_res:.4f} m/s "
                f"npeaks(full)={len(peaks)} "
                f"strongest=({peaks[0].rng_bin},{peaks[0].vel_bin}) "
                f"{peaks[0].db:.1f} dB"
            )
        else:
            print(f"[{ds}] 未检测到峰（检查 margin 参数）")
        print(f"[{ds}] 功率恒等式 max_rel_err={identity['max_rel_err']:.2e}")
        print(f"[{ds}] 平均vs求和: 期望 {avg_sum['expected_db']:.4f} dB, "
              f"实测 [{avg_sum['delta_min_db']:.6f}, {avg_sum['delta_max_db']:.6f}], "
              f"偏差 {avg_sum['max_abs_dev_db']:.2e} dB")

    # ---- 图 -----------------------------------------------------------------
    peaks_by_ds = {ds: all_out[ds]["peaks"] for ds in ("far", "near")}
    if make_figures:
        plots.fig_accumulated_rv(
            frames, cfgs, peaks_by_ds, os.path.join(fig_dir, "fig1_rv_accumulated_4ch.png"))
        plots.fig_single_channels(
            frames, cfgs, os.path.join(fig_dir, "fig2_single_channel_rv.png"))
        plots.fig_leave_one_out(
            {ds: all_out[ds]["loo_diffs"] for ds in ("far", "near")}, cfgs,
            os.path.join(fig_dir, "fig3_leave_one_out_diff.png"))
        plots.fig_subset_scan(
            {ds: all_out[ds]["subsets"] for ds in ("far", "near")},
            os.path.join(fig_dir, "fig4_subset_scan.png"))
        plots.fig_avg_vs_sum(frames, os.path.join(fig_dir, "fig5_avg_vs_sum.png"))
        plots.fig_channel_corr(frames, os.path.join(fig_dir, "fig6_channel_corr.png"))
        jdir = jpg_dir or data_dir
        jpgs = {ds: os.path.join(jdir, DATASETS[ds]["jpg"]) for ds in ("far", "near")}
        if all(os.path.exists(p) for p in jpgs.values()):
            plots.fig_official_compare(
                jpgs, frames, cfgs, peaks_by_ds,
                os.path.join(fig_dir, "fig7_official_jpg_compare.png"))
            for ds in ("far", "near"):
                stat = analyze_official_jpg(jpgs[ds])
                print(f"[{ds}] 官方 JPG 结构统计: {stat}")
        else:
            print("[warn] 官方 JPG 不存在，跳过对照图: ", jpgs)

    _write_summary(all_out, frames, os.path.join(res_dir, "summary.md"), data_dir)
    return {"frames": frames, "cfgs": cfgs, **all_out}


def _write_summary(all_out, frames, path: str, data_dir: str):
    lines: List[str] = []
    lines.append("# a100-rx-ablation 结果汇总（由 cli 自动生成）\n")
    lines.append(f"- 数据目录：`{data_dir}`")
    lines.append("- 峰检测参数：margin = 噪声底(dB 中位数) + "
                 f"{PEAK_MARGIN_DB} dB，最小分离 ({PEAK_MIN_RNG_SEP}, "
                 f"{PEAK_MIN_VEL_SEP}) bin，匹配容差 ({MATCH_RNG_TOL}, {MATCH_VEL_TOL}) bin\n")
    for ds in ("far", "near"):
        o = all_out[ds]
        cfg = o["cfg"]
        lines.append(f"\n## {DATASETS[ds]['label']}（{ds}）\n")
        lines.append("| 参数 | 值 |")
        lines.append("|---|---|")
        s = cfg.summary()
        lines.append(f"| adc_sample_num（官方取整分支，dec_factor=1 无 +1） | {s['adc_sample_num']} |")
        lines.append(f"| rng_res | {s['rng_res_m']:.4f} m/bin |")
        lines.append(f"| V_res（名义，DDMA 未解码仅作参考） | {s['v_res_mps']:.4f} m/s/bin |")
        lines.append(f"| RV 尺寸 | {cfg.rng_bin_num} × {cfg.vel_nfft} |\n")
        top = sorted(o["peaks"], key=lambda p: -p.db)[:5]
        lines.append("4 通道平均图最强 5 峰（0 基 bin）：")
        lines.append("| # | rng_bin | rng(m) | vel_bin | 签名 vel bin | dB |")
        lines.append("|---|---|---|---|---|---|")
        for i, p in enumerate(top, 1):
            lines.append(
                f"| {i} | {p.rng_bin} | {p.rng_bin * cfg.rng_res:.2f} | {p.vel_bin} "
                f"| {signed_vel_bin(p.vel_bin, cfg.vel_nfft)} | {p.db:.1f} |")
        lines.append("\n单通道能量占比：")
        for sgl in o["singles"]:
            lines.append(f"- RX{sgl['channel']}: {sgl['energy_share'] * 100:.1f}%"
                         f"（max {sgl['max_db']:.1f} dB, {sgl['n_peaks']} 峰）")
        lines.append("\n子集扫描（相对全 4 通道；SNR = 最强峰 dB − 噪声底 dB）：")
        lines.append("| 子集 | k | 最强峰(rng,vel) | 抖动(r,v) bin | Jaccard | 最强峰 dB | SNR dB |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in o["subsets"]:
            lines.append(
                f"| {r.label} | {r.k} | ({r.strongest_rng_bin},{r.strongest_vel_bin}) "
                f"| ({r.jitter_rng},{r.jitter_vel}) | {r.jaccard:.3f} "
                f"| {r.strongest_db:.1f} | {r.snr_db:.1f} |")
        lines.append("\n留一消融（ΔdB>0 = 去掉后减弱）：")
        lines.append("| 去掉 | Jaccard | 抖动(r,v) | top5 峰 ΔdB |")
        lines.append("|---|---|---|---|")
        for r in o["loo"]:
            ds5 = ",".join(f"{d:+.1f}" for d in r.top_peak_deltas_db)
            lines.append(f"| RX{r.dropped} | {r.jaccard:.3f} "
                         f"| ({r.jitter_rng},{r.jitter_vel}) | {ds5} |")
        lines.append("\n通道相关矩阵（dB 域 / 线性功率域）：")
        for k in range(4):
            lines.append("- dB: " + " ".join(f"{o['corr'][k, j]:.3f}" for j in range(4))
                         + " | 线性: "
                         + " ".join(f"{o['corr_lin'][k, j]:.3f}" for j in range(4)))
        lines.append(f"\n平均 vs 求和：Δ = {o['avg_sum']['expected_db']:.4f} dB 常数，"
                     f"最大偏差 {o['avg_sum']['max_abs_dev_db']:.2e} dB。")
        lines.append(f"功率恒等式（Σ单通道 = 总和图）：max_rel_err = "
                     f"{o['identity']['max_rel_err']:.2e}。")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main(argv=None):
    repo = repo_root_from_module()
    ap = argparse.ArgumentParser(
        description="CTASI-A100 四接收通道非相干积累消融实验一键复现")
    ap.add_argument("--data-dir", default=os.path.join(repo, DEFAULT_DATA_DIR),
                    help="ADC 示例数据目录（默认仓内路径，不复制数据）")
    ap.add_argument("--cfg-dir", default=os.path.join(repo, DEFAULT_CFG_DIR),
                    help=".hxx 配置目录")
    ap.add_argument("--jpg-dir", default=None,
                    help="官方结果 JPG 所在目录（默认同 --data-dir）")
    ap.add_argument("--out-dir", default=None, help="兼容参数：图与结果固定写入项目内 figures/ results/")
    ap.add_argument("--no-figures", action="store_true", help="只算不出图")
    args = ap.parse_args(argv)
    run(args.data_dir, args.cfg_dir, args.out_dir or ".",
        make_figures=not args.no_figures, jpg_dir=args.jpg_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
