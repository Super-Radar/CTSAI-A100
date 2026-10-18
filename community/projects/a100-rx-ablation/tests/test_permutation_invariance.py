# -*- coding: utf-8 -*-
"""合成数据上的通道置换不变性测试。

官方非相干积累是“通道功率图求平均”，对通道顺序天然不变；
这保证消融结论与通道文件命名顺序无关（本文全部指标都继承该性质）。
"""
import itertools

import numpy as np

from rx_ablation import metrics
from rx_ablation.ablation import leave_one_out, scan_subsets
from rx_ablation.pipeline import accumulate_power


def _synth_power(seed=7):
    rng = np.random.default_rng(seed)
    base = rng.lognormal(1.0, 2.0, size=(32, 24, 1))
    gains = rng.uniform(0.5, 2.0, size=4)
    return np.repeat(base, 4, axis=2) * gains[None, None, :]


def test_accumulate_power_permutation_invariant():
    p = _synth_power()
    ref = accumulate_power(p)
    for perm in itertools.permutations(range(4)):
        np.testing.assert_allclose(accumulate_power(p[:, :, list(perm)]), ref, rtol=1e-12)


def test_peak_detection_permutation_invariant():
    p = _synth_power(seed=8)
    ref = metrics.detect_peaks(accumulate_power(p), margin_db=8.0)
    for perm in itertools.permutations(range(4)):
        got = metrics.detect_peaks(accumulate_power(p[:, :, list(perm)]), margin_db=8.0)
        assert [(q.rng_bin, q.vel_bin) for q in got] == [
            (q.rng_bin, q.vel_bin) for q in ref
        ]


def test_subset_scan_invariant_under_channel_relabeling():
    """子集按通道集合（而非顺序）定义：重命名通道后，同集合子集结果一致。"""
    p = _synth_power(seed=9)
    res_a, _ = scan_subsets(p, margin_db=8.0, min_rng_sep=2, min_vel_sep=2)
    perm = (2, 0, 3, 1)
    p2 = p[:, :, list(perm)]
    res_b, _ = scan_subsets(p2, margin_db=8.0, min_rng_sep=2, min_vel_sep=2)
    by_a = {frozenset(r.channels): r for r in res_a}
    by_b = {frozenset(r.channels): r for r in res_b}
    for key, ra in by_a.items():
        rb = by_b[frozenset(perm.index(c) for c in key)]
        assert ra.strongest_rng_bin == rb.strongest_rng_bin
        assert ra.strongest_vel_bin == rb.strongest_vel_bin
        assert abs(ra.jaccard - rb.jaccard) < 1e-12
        assert abs(ra.total_power - rb.total_power) < 1e-6 * abs(ra.total_power)


def test_leave_one_out_matches_direct_computation():
    p = _synth_power(seed=10)
    res, diffs = leave_one_out(p, margin_db=8.0, min_rng_sep=2, min_vel_sep=2)
    full = accumulate_power(p)
    for ch in range(4):
        keep = [c for c in range(4) if c != ch]
        expect = 10 * np.log10(full) - 10 * np.log10(accumulate_power(p, keep))
        np.testing.assert_allclose(diffs[ch], expect, rtol=1e-12)
        assert res[ch].dropped == ch
