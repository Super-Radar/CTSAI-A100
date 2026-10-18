# -*- coding: utf-8 -*-
"""消融实验用指标：峰检测、峰稳定性、能量守恒校验、通道相关性。

约定：
- 图像 shape = (rng_bin, vel_bin)，bin 索引一律 **0 基**（MATLAB 官方脚本
  为 1 基，换算时 +1；本模块所有输出均按 0 基报告）。
- 峰检测是“局部极大 + 噪声底加阈值”的简单检测器（非 CFAR），阈值与
  分离参数全部显式传参，结论不依赖隐藏默认值。
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import maximum_filter

from .pipeline import to_db


# ---------------------------------------------------------------------------
# 峰检测
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Peak:
    rng_bin: int
    vel_bin: int
    db: float

    @property
    def xy(self) -> Tuple[int, int]:
        return (self.rng_bin, self.vel_bin)


def noise_floor_db(power2d: np.ndarray) -> float:
    """鲁棒噪声底：dB 图的中位数（与官方 calculate_global_noise.m 思路
    一致的“全图统计”，此处取中位数以抗强峰影响）。"""
    return float(np.median(to_db(power2d)))


def detect_peaks(
    power2d: np.ndarray,
    margin_db: float = 15.0,
    min_rng_sep: int = 4,
    min_vel_sep: int = 4,
    top_n: int = 64,
    floor_db: float | None = None,
) -> List[Peak]:
    """局部极大 + 阈值 + 最小分离的峰检测。

    - ``margin_db``：检测阈值 = 噪声底 + margin（dB）。
    - ``min_rng_sep / min_vel_sep``：贪心非极大值抑制的最小 bin 间距
      （两轴同时满足才允许共存）。
    - 返回按 dB 降序的前 ``top_n`` 个峰。
    """
    img = to_db(power2d)
    if floor_db is None:
        floor_db = float(np.median(img))
    thr = floor_db + margin_db
    mx = maximum_filter(img, size=3, mode="nearest")
    cand = np.argwhere((img >= thr) & (img == mx))
    if cand.size == 0:
        return []
    vals = img[cand[:, 0], cand[:, 1]]
    order = np.argsort(vals)[::-1]
    peaks: List[Peak] = []
    for idx in order:
        r, v = int(cand[idx, 0]), int(cand[idx, 1])
        if all(
            abs(r - p.rng_bin) >= min_rng_sep or abs(v - p.vel_bin) >= min_vel_sep
            for p in peaks
        ):
            peaks.append(Peak(r, v, float(vals[idx])))
        if len(peaks) >= top_n:
            break
    return peaks


# ---------------------------------------------------------------------------
# 峰稳定性
# ---------------------------------------------------------------------------

def _match(
    a: Sequence[Peak], b: Sequence[Peak], rng_tol: int, vel_tol: int
) -> Tuple[int, int]:
    """贪心双向匹配：返回 (匹配数, 并集大小)。"""
    used_b = set()
    matched = 0
    for pa in sorted(a, key=lambda p: -p.db):
        for jb, pb in enumerate(sorted(b, key=lambda p: -p.db)):
            if jb in used_b:
                continue
            if abs(pa.rng_bin - pb.rng_bin) <= rng_tol and abs(
                pa.vel_bin - pb.vel_bin
            ) <= vel_tol:
                used_b.add(jb)
                matched += 1
                break
    return matched, len(a) + len(b) - matched


def peak_jaccard(
    a: Sequence[Peak],
    b: Sequence[Peak],
    rng_tol: int = 3,
    vel_tol: int = 3,
) -> float:
    """检测集 Jaccard：|匹配| / |并集|（容差内视为同一峰）。"""
    if not a and not b:
        return 1.0
    matched, union = _match(a, b, rng_tol, vel_tol)
    return matched / union if union else 1.0


def strongest_peak_jitter(
    ref: Sequence[Peak], test: Sequence[Peak]
) -> Tuple[int, int]:
    """最强峰位置抖动：返回 (Δrng_bin, Δvel_bin)（test 相对 ref，0 基）。"""
    if not ref or not test:
        raise ValueError("需要非空峰列表")
    ra = max(ref, key=lambda p: p.db)
    rb = max(test, key=lambda p: p.db)
    return (rb.rng_bin - ra.rng_bin, rb.vel_bin - ra.vel_bin)


def chebyshev_jitter(jitter: Tuple[int, int]) -> int:
    return max(abs(jitter[0]), abs(jitter[1]))


# ---------------------------------------------------------------------------
# 能量守恒校验（功率域恒等式）
# ---------------------------------------------------------------------------

def power_identity(channel_powers: np.ndarray) -> Dict[str, float]:
    """校验 Σ_ch P_ch == 全通道求和图（功率域线性恒等式）。

    该恒等式成立的充要条件是积累对象是**功率**而非复幅度：若误用
    |Σ_ch X_ch|^2 则会出现通道间交叉项，恒等式被破坏。
    """
    total = channel_powers.sum(axis=2)
    recon = np.zeros_like(total)
    for ch in range(channel_powers.shape[2]):
        recon += channel_powers[:, :, ch]
    return {
        "total": float(total.sum()),
        "sum_of_singles": float(recon.sum()),
        "max_abs_err": float(np.max(np.abs(total - recon))),
        "max_rel_err": float(
            np.max(np.abs(total - recon)) / np.max(np.abs(total))
        ),
    }


# ---------------------------------------------------------------------------
# 通道相关性与能量
# ---------------------------------------------------------------------------

def channel_correlation(
    channel_powers: np.ndarray, domain: str = "db"
) -> np.ndarray:
    """通道间逐 bin 功率相关系数矩阵（默认 dB 域）。"""
    x = channel_powers.reshape(channel_powers.shape[2], -1)
    if domain == "db":
        x = to_db(channel_powers).reshape(channel_powers.shape[2], -1)
    return np.corrcoef(x)


def channel_energy(channel_powers: np.ndarray) -> np.ndarray:
    """各通道总能量（功率图全 bin 求和）。"""
    return channel_powers.sum(axis=(0, 1))


def channel_power_histogram(
    power2d: np.ndarray, bins: int = 60
) -> Tuple[np.ndarray, np.ndarray]:
    return np.histogram(to_db(power2d).ravel(), bins=bins)


def peak_channel_shares(
    channel_powers: np.ndarray, peaks: Sequence[Peak]
) -> List[Dict[str, float]]:
    """给定峰位，计算各通道在该 bin 的功率占比（谁在主导该峰）。"""
    out = []
    nch = channel_powers.shape[2]
    for p in peaks:
        vals = np.array(
            [channel_powers[p.rng_bin, p.vel_bin, ch] for ch in range(nch)]
        )
        shares = vals / vals.sum() if vals.sum() > 0 else np.zeros(nch)
        out.append(
            {
                "rng_bin": p.rng_bin,
                "vel_bin": p.vel_bin,
                "db": p.db,
                **{f"rx{k}_share": float(shares[k]) for k in range(nch)},
            }
        )
    return out


def subset_labels(n_channels: int = 4) -> List[Tuple[Tuple[int, ...], str]]:
    """所有 C(4,k) 通道子集（k=1..4），按 k 升序、字典序。"""
    out = []
    for k in range(1, n_channels + 1):
        for combo in combinations(range(n_channels), k):
            out.append((combo, "+".join(f"RX{c}" for c in combo)))
    return out
