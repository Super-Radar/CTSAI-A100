# -*- coding: utf-8 -*-
"""消融实验：单通道画像、C(4,k) 子集扫描、留一消融、平均 vs 求和对照。

全部实验都在功率域进行（诚实红线：不做通道同步性 / 相位标定断言）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

from . import metrics
from .pipeline import RadarConfig, accumulate_power, to_db


@dataclass
class SubsetResult:
    label: str
    k: int
    channels: Tuple[int, ...]
    strongest_rng_bin: int
    strongest_vel_bin: int
    strongest_db: float
    floor_db: float
    snr_db: float
    jitter_rng: int
    jitter_vel: int
    jaccard: float
    total_power: float
    n_peaks: int


def scan_subsets(
    power: np.ndarray,
    margin_db: float,
    min_rng_sep: int,
    min_vel_sep: int,
    match_rng_tol: int = 3,
    match_vel_tol: int = 3,
) -> Tuple[List[SubsetResult], Dict[Tuple[int, ...], List[metrics.Peak]]]:
    """所有 C(4,k) 子集的官方定义积累（功率平均）+ 与全集对照的稳定性。

    返回 (结果列表, 各子集峰列表)。全集 (0,1,2,3) 自身作为参照
    （jitter=0, jaccard=1）。
    """
    full = (0, 1, 2, 3)
    full_peaks = metrics.detect_peaks(
        accumulate_power(power, list(full)),
        margin_db=margin_db, min_rng_sep=min_rng_sep, min_vel_sep=min_vel_sep,
    )
    peaks_cache: Dict[Tuple[int, ...], List[metrics.Peak]] = {full: full_peaks}
    results: List[SubsetResult] = []
    for combo, label in metrics.subset_labels(power.shape[2]):
        acc = accumulate_power(power, list(combo))
        peaks = metrics.detect_peaks(
            acc,
            margin_db=margin_db,
            min_rng_sep=min_rng_sep,
            min_vel_sep=min_vel_sep,
        )
        peaks_cache[combo] = peaks
        if combo == full:
            jitter = (0, 0)
            jac = 1.0
        else:
            jitter = metrics.strongest_peak_jitter(peaks_cache[full], peaks)
            jac = metrics.peak_jaccard(
                peaks_cache[full], peaks, match_rng_tol, match_vel_tol
            )
        strongest = max(peaks, key=lambda p: p.db) if peaks else None
        floor = float(np.median(to_db(acc)))
        results.append(
            SubsetResult(
                label=label,
                k=len(combo),
                channels=combo,
                strongest_rng_bin=strongest.rng_bin if strongest else -1,
                strongest_vel_bin=strongest.vel_bin if strongest else -1,
                strongest_db=strongest.db if strongest else float("nan"),
                floor_db=floor,
                snr_db=(strongest.db - floor) if strongest else float("nan"),
                jitter_rng=jitter[0],
                jitter_vel=jitter[1],
                jaccard=jac,
                total_power=float(acc.sum()),
                n_peaks=len(peaks),
            )
        )
    return results, peaks_cache


@dataclass
class LeaveOneOutResult:
    dropped: int
    label: str
    n_peaks: int
    jaccard: float
    jitter_rng: int
    jitter_vel: int
    top_peak_deltas_db: List[float]
    diff_max_db: float
    diff_min_db: float


def leave_one_out(
    power: np.ndarray,
    margin_db: float,
    min_rng_sep: int,
    min_vel_sep: int,
    match_rng_tol: int = 3,
    match_vel_tol: int = 3,
    top_n_peaks: int = 5,
) -> Tuple[List[LeaveOneOutResult], Dict[int, np.ndarray]]:
    """留一消融：全 4 通道平均 vs 去掉通道 ch 的 3 通道平均。

    差值图定义（dB）：delta = 10log10(mean_full) - 10log10(mean_loo)。
    delta > 0 表示“去掉该通道后此处减弱”，即该通道在此 bin 贡献大。
    """
    full = accumulate_power(power)
    full_peaks = metrics.detect_peaks(
        full, margin_db=margin_db, min_rng_sep=min_rng_sep, min_vel_sep=min_vel_sep
    )
    top_peaks = sorted(full_peaks, key=lambda p: -p.db)[:top_n_peaks]
    diffs: Dict[int, np.ndarray] = {}
    results: List[LeaveOneOutResult] = []
    for ch in range(power.shape[2]):
        keep = [c for c in range(power.shape[2]) if c != ch]
        loo = accumulate_power(power, keep)
        loo_peaks = metrics.detect_peaks(
            loo, margin_db=margin_db, min_rng_sep=min_rng_sep, min_vel_sep=min_vel_sep
        )
        diff = to_db(full) - to_db(loo)
        diffs[ch] = diff
        jitter = metrics.strongest_peak_jitter(full_peaks, loo_peaks)
        results.append(
            LeaveOneOutResult(
                dropped=ch,
                label=f"drop RX{ch}",
                n_peaks=len(loo_peaks),
                jaccard=metrics.peak_jaccard(
                    full_peaks, loo_peaks, match_rng_tol, match_vel_tol
                ),
                jitter_rng=jitter[0],
                jitter_vel=jitter[1],
                top_peak_deltas_db=[
                    float(diff[p.rng_bin, p.vel_bin]) for p in top_peaks
                ],
                diff_max_db=float(diff.max()),
                diff_min_db=float(diff.min()),
            )
        )
    return results, diffs


def avg_vs_sum(power: np.ndarray, channels: Sequence[int] | None = None) -> Dict[str, float]:
    """平均 vs 求和对照：dB 域差 = 10log10(k) 常数（k = 通道数）。"""
    sel = power if channels is None else power[:, :, list(channels)]
    k = sel.shape[2]
    mean_db = to_db(accumulate_power(sel))
    sum_db = to_db(sel.sum(axis=2))
    delta = sum_db - mean_db
    expect = 10.0 * np.log10(k)
    return {
        "k": k,
        "expected_db": float(expect),
        "delta_min_db": float(delta.min()),
        "delta_max_db": float(delta.max()),
        "max_abs_dev_db": float(np.max(np.abs(delta - expect))),
    }


def single_channel_profiles(
    power: np.ndarray, cfg: RadarConfig, margin_db: float, min_rng_sep: int,
    min_vel_sep: int, top_n: int = 5,
) -> List[Dict[str, object]]:
    """各单通道画像：最强 N 峰（bin + 物理量）、总能量、能量占比。"""
    energies = metrics.channel_energy(power)
    total = float(energies.sum())
    out = []
    for ch in range(power.shape[2]):
        peaks = metrics.detect_peaks(
            power[:, :, ch], margin_db=margin_db,
            min_rng_sep=min_rng_sep, min_vel_sep=min_vel_sep,
        )[:top_n]
        out.append(
            {
                "channel": ch,
                "energy": float(energies[ch]),
                "energy_share": float(energies[ch] / total) if total else 0.0,
                "max_db": float(to_db(power[:, :, ch]).max()),
                "n_peaks": len(
                    metrics.detect_peaks(
                        power[:, :, ch], margin_db=margin_db,
                        min_rng_sep=min_rng_sep, min_vel_sep=min_vel_sep,
                    )
                ),
                "peaks": peaks,
            }
        )
    return out


def subset_energy_identity(power: np.ndarray) -> Dict[str, float]:
    """能量守恒校验（metrics.power_identity 的封装）。"""
    return metrics.power_identity(power)
