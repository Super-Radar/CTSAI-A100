# -*- coding: utf-8 -*-
"""出图（matplotlib Agg -> PNG）。

配色遵循无障碍数据可视化规范：
- 分类色（RX0..RX3 / 数据集）：固定顺序不循环，蓝/橙/水绿/黄；
- 幅值热图：单色相蓝渐变（浅=噪声底，深=强）；
- 留一差值图：蓝-红发散色，中点为中性灰（0 dB）；
- 细标记、2px 线、图例 + 直接标注、单轴（不用双 y 轴）。
"""

from __future__ import annotations

import os
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from . import metrics  # noqa: E402
from .pipeline import RadarConfig, accumulate_power, to_db  # noqa: E402

# ---- 调色板（已按 CVD 安全顺序选定，勿循环重排） --------------------------
RX_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # RX0..RX3
DATASET_COLORS = {"far": "#2a78d6", "near": "#eb6834"}
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]
)
DIV_CMAP = LinearSegmentedColormap.from_list(
    "div_blue_red", ["#2a78d6", "#9ec5f4", "#f0efec", "#ec835a", "#d03b3b"]
)

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"

DS_TITLE = {"far": "远波 f0（300 MHz / 256 chirp / rng_nfft 1024）",
            "near": "近波 f1（750 MHz / 128 chirp / rng_nfft 2048）"}


def set_style():
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.sans-serif": ["WenQuanYi Micro Hei", "Microsoft YaHei", "DejaVu Sans"],
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
        "text.color": INK,
        "axes.labelcolor": INK2,
        "axes.edgecolor": BASE,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9.5,
        "axes.titlesize": 10.5,
        "axes.labelsize": 9.5,
    })


def _rv_extent(cfg: RadarConfig):
    """(left, right, bottom, top)：x = Doppler bin（官方顺序，不做 fftshift），
    y = 距离 m（bin * rng_res，0 基）。"""
    return (0, cfg.vel_nfft, 0.0, cfg.rng_bin_num * cfg.rng_res)


def _imshow_db(ax, img_db, cfg, vmin=None, vmax=None, cmap=SEQ_CMAP):
    vmin = np.median(img_db) if vmin is None else vmin
    vmax = img_db.max() if vmax is None else vmax
    im = ax.imshow(
        img_db,
        origin="lower",
        aspect="auto",
        extent=_rv_extent(cfg),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    ax.set_xlabel("Doppler bin（官方顺序，未 fftshift）")
    ax.set_ylabel("距离 (m)")
    return im


def _mark_peaks(ax, peaks: Sequence[metrics.Peak], cfg: RadarConfig, n=3):
    for p in sorted(peaks, key=lambda q: -q.db)[:n]:
        x = p.vel_bin + 0.5
        y = (p.rng_bin + 0.5) * cfg.rng_res
        ax.annotate(
            f"{p.db:.1f}", (x, y), textcoords="offset points", xytext=(4, 4),
            fontsize=8, color=INK2,
        )
        ax.plot([x], [y], marker="o", ms=4, mfc="none", mec=INK, mew=0.8)


# ---------------------------------------------------------------------------
# 图 1：官方定义复现 —— 4 通道功率平均 RVimage（dB）
# ---------------------------------------------------------------------------

def fig_accumulated_rv(frames, cfgs, peaks_by_ds, out_path):
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), constrained_layout=True)
    for ax, ds in zip(axes, ["far", "near"]):
        img = to_db(frames[ds]["rv_image"])
        im = _imshow_db(ax, img, cfgs[ds])
        _mark_peaks(ax, peaks_by_ds[ds], cfgs[ds])
        ax.set_title(DS_TITLE[ds])
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("功率 dB（色标下限 = 图中位数）")
    fig.suptitle("官方定义复现：RVimage = Σ通道功率 / 4（非相干积累 = 功率平均）", y=1.06)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 2：单通道画像
# ---------------------------------------------------------------------------

def fig_single_channels(frames, cfgs, out_path):
    set_style()
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), constrained_layout=True)
    for row, ds in enumerate(["far", "near"]):
        cfg = cfgs[ds]
        img4 = to_db(frames[ds]["rv_image"])
        vmin, vmax = np.median(img4), img4.max()
        for ch in range(4):
            ax = axes[row, ch]
            img = to_db(frames[ds]["power"][:, :, ch])
            im = _imshow_db(ax, img, cfg, vmin=vmin, vmax=vmax)
            ax.set_title(f"{ds} RX{ch}（max {img.max():.1f} dB）")
            if ch > 0:
                ax.set_ylabel("")
        cb = fig.colorbar(im, ax=axes[row, :], pad=0.015, shrink=0.9)
        cb.set_label("单通道功率 dB（色标与同行 4 通道平均图一致）")
    fig.suptitle("单通道功率图：色标统一到 4 通道平均图的水平，直观看通道差异", y=1.03)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 3：留一消融差值图
# ---------------------------------------------------------------------------

def fig_leave_one_out(diffs_by_ds, cfgs, out_path):
    set_style()
    fig, axes = plt.subplots(2, 4, figsize=(13.5, 6.6), constrained_layout=True)
    for row, ds in enumerate(["far", "near"]):
        cfg = cfgs[ds]
        lims = [
            max(3.0, min(20.0, np.percentile(np.abs(d), 99.5)))
            for d in diffs_by_ds[ds].values()
        ]
        vmax = max(lims)
        for ch in range(4):
            ax = axes[row, ch]
            d = diffs_by_ds[ds][ch]
            im = _imshow_db(ax, d, cfg, vmin=-vmax, vmax=vmax, cmap=DIV_CMAP)
            ax.set_title(f"{ds} 去掉 RX{ch}")
            if ch > 0:
                ax.set_ylabel("")
        cb = fig.colorbar(im, ax=axes[row, :], pad=0.015, shrink=0.9)
        cb.set_label("ΔdB = 10log10(全4通道平均 / 留一平均)")
    fig.suptitle("留一消融：暖色 = 去掉该通道后此处减弱（该通道贡献大），冷色 = 增强", y=1.03)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 4：子集扫描（15 个子集）
# ---------------------------------------------------------------------------

def fig_subset_scan(results_by_ds, out_path):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), constrained_layout=True)
    panels = [
        ("最强峰位置抖动（Chebyshev bin）", "cheb"),
        ("与全集检测集的 Jaccard", "jaccard"),
        ("最强峰 SNR（峰 dB − 噪声底 dB）", "snr"),
        ("积累图总功率（功率平均定义）", "power"),
    ]
    for ds in ["far", "near"]:
        res = results_by_ds[ds]
        full_power = [r.total_power for r in res if r.k == 4][0]
        by_k: Dict[int, List] = {}
        for r in res:
            by_k.setdefault(r.k, []).append(r)
        color = DATASET_COLORS[ds]
        for ax, (title, what) in zip(axes.flat, panels):
            ks = sorted(by_k)
            if what == "cheb":
                vals = [metrics.chebyshev_jitter((r.jitter_rng, r.jitter_vel)) for r in res]
                per_k = [np.mean([metrics.chebyshev_jitter((r.jitter_rng, r.jitter_vel)) for r in by_k[k]]) for k in ks]
                ylab = "bin"
            elif what == "jaccard":
                vals = [r.jaccard for r in res]
                per_k = [np.mean([r.jaccard for r in by_k[k]]) for k in ks]
                ylab = "Jaccard"
            elif what == "snr":
                vals = [r.snr_db for r in res]
                per_k = [np.mean([r.snr_db for r in by_k[k]]) for k in ks]
                ylab = "dB"
            else:
                vals = [r.total_power / full_power for r in res]
                per_k = [np.mean([r.total_power / full_power for r in by_k[k]]) for k in ks]
                ylab = "归一化总功率"
            xs = [r.k + (0.08 if ds == "near" else -0.08) for r in res]
            ax.scatter(xs, vals, s=22, color=color, alpha=0.55, zorder=2,
                       label=f"{ds} 各子集" if ax is axes.flat[0] else None)
            ax.plot(ks, per_k, color=color, lw=2, marker="o", ms=5,
                    label=f"{ds} 均值" if ax is axes.flat[0] else None)
            ax.set_title(title)
            ax.set_xlabel("子集通道数 k")
            ax.set_ylabel(ylab)
            ax.set_xticks([1, 2, 3, 4])
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(0.99, 0.99),
               frameon=False)
    fig.suptitle("C(4,k) 子集扫描（15 个子集 × 远/近两组）：通道数与峰稳定性的关系", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 5：平均 vs 求和
# ---------------------------------------------------------------------------

def fig_avg_vs_sum(frames, out_path):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.8), constrained_layout=True)
    for row, ds in enumerate(["far", "near"]):
        power = frames[ds]["power"]
        delta = to_db(power.sum(axis=2)) - to_db(accumulate_power(power))
        expect = 10 * np.log10(4)
        ax = axes[row, 0]
        im = ax.imshow(
            delta, origin="lower", aspect="auto", cmap=SEQ_CMAP,
            vmin=expect - 1e-6, vmax=expect + 1e-6, interpolation="nearest",
        )
        ax.set_title(f"{ds}：10log10(Σ) − 10log10(mean) 全图")
        ax.set_xlabel("Doppler bin")
        ax.set_ylabel("距离 bin")
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label(f"ΔdB ≈ {expect:.4f}")
        ax2 = axes[row, 1]
        resid = (delta - expect).ravel() * 1e12     # 单位 1e-12 dB
        ax2.plot(np.arange(resid.size), np.sort(resid), lw=2,
                 color=DATASET_COLORS[ds])
        ax2.annotate(
            f"最大 |残差| = {np.max(np.abs(resid)):.2f}e-12 dB",
            (0.03, 0.92), xycoords="axes fraction", fontsize=9, color=INK2,
        )
        ax2.set_title(f"{ds}：ΔdB − 10log10(4) 残差（排序后）")
        ax2.set_xlabel("bin 秩")
        ax2.set_ylabel("残差（1e-12 dB）")
    fig.suptitle("平均 vs 求和：只差常数 10log10(4) dB，图结构完全相同", y=1.03)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 6：通道相关性 + 功率直方图
# ---------------------------------------------------------------------------

def fig_channel_corr(frames, out_path):
    set_style()
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.6), constrained_layout=True)
    for row, ds in enumerate(["far", "near"]):
        power = frames[ds]["power"]
        corr = metrics.channel_correlation(power, domain="db")
        ax = axes[row, 0]
        im = ax.imshow(corr, vmin=0.5, vmax=1.0, cmap=SEQ_CMAP)
        ax.set_xticks(range(4), [f"RX{k}" for k in range(4)])
        ax.set_yticks(range(4), [f"RX{k}" for k in range(4)])
        for i in range(4):
            for j in range(4):
                ax.text(j, i, f"{corr[i, j]:.3f}", ha="center", va="center",
                        fontsize=8, color=INK if corr[i, j] < 0.85 else "#cde2fb")
        ax.set_title(f"{ds}：通道间逐 bin 功率相关（dB 域）")
        ax.grid(False)
        cb = fig.colorbar(im, ax=ax, pad=0.02)
        cb.set_label("Pearson r")
        ax2 = axes[row, 1]
        for ch in range(4):
            ax2.hist(to_db(power[:, :, ch]).ravel(), bins=80, histtype="step",
                     lw=2, color=RX_COLORS[ch], label=f"RX{ch}")
        ax2.set_title(f"{ds}：各通道功率直方图（dB）")
        ax2.set_xlabel("dB")
        ax2.set_ylabel("bin 数")
        ax2.legend(frameon=False, fontsize=8.5)
    fig.suptitle("通道相关性与差异：结构高度相关，但能量分布不同", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 图 7：官方 JPG 定性对照
# ---------------------------------------------------------------------------

def fig_official_compare(jpg_by_ds, frames, cfgs, peaks_by_ds, out_path):
    set_style()
    from PIL import Image
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), constrained_layout=True)
    for row, ds in enumerate(["far", "near"]):
        ax = axes[row, 0]
        ax.imshow(np.asarray(Image.open(jpg_by_ds[ds]).convert("RGB")))
        ax.set_title(f"官方结果图（3D mesh 渲染）：{os.path.basename(jpg_by_ds[ds])}")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax2 = axes[row, 1]
        img = to_db(frames[ds]["rv_image"])
        im = _imshow_db(ax2, img, cfgs[ds])
        _mark_peaks(ax2, peaks_by_ds[ds], cfgs[ds])
        ax2.set_title(f"本项目复现（4 通道功率平均，dB 热图）")
        cb = fig.colorbar(im, ax=ax2, pad=0.02)
        cb.set_label("功率 dB")
    fig.suptitle("与官方 JPG 的定性对照：同一数据 + 同一官方链路定义，仅可视化方式不同", y=1.02)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
