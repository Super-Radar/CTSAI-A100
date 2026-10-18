# -*- coding: utf-8 -*-
"""功率恒等式与平均/求和换算测试（合成数据，无需仓库数据）。"""
import numpy as np
import pytest

from rx_ablation import metrics
from rx_ablation.ablation import avg_vs_sum, subset_energy_identity
from rx_ablation.pipeline import accumulate_power


def _random_powers(ch=4, rng_bin=24, vel_bin=20, seed=0):
    rng = np.random.default_rng(seed)
    # 对数正态功率，模拟强峰 + 宽动态范围
    return rng.lognormal(mean=2.0, sigma=3.0, size=(rng_bin, vel_bin, ch))


def test_power_identity_sum_of_singles_equals_total():
    p = _random_powers()
    ident = subset_energy_identity(p)
    assert ident["max_abs_err"] == 0.0
    assert ident["max_rel_err"] < 1e-12
    assert np.isclose(ident["total"], ident["sum_of_singles"])


def test_power_identity_partition_over_subsets():
    """互斥子集的功率和相加 == 全通道总和（功率域线性性）。"""
    p = _random_powers(seed=1)
    total = p.sum(axis=2)
    parts = p[:, :, 0] + p[:, :, 1] + p[:, :, [2, 3]].sum(axis=2)
    np.testing.assert_allclose(parts, total, rtol=1e-12)


def test_accumulate_power_is_mean_of_selected():
    p = _random_powers(seed=2)
    np.testing.assert_allclose(accumulate_power(p, [0, 1, 2, 3]), p.mean(axis=2))
    np.testing.assert_allclose(
        accumulate_power(p, [1, 3]), p[:, :, [1, 3]].mean(axis=2)
    )
    # 单通道：积累图就是该通道功率图
    np.testing.assert_allclose(accumulate_power(p, [2]), p[:, :, 2])


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_avg_vs_sum_constant_offset(k):
    p = _random_powers(seed=3)
    res = avg_vs_sum(p, list(range(k)))
    assert res["k"] == k
    assert np.isclose(res["expected_db"], 10 * np.log10(k))
    assert res["delta_max_db"] - res["delta_min_db"] < 1e-9
    assert res["max_abs_dev_db"] < 1e-9


def test_avg_vs_sum_db_values_match_direct_computation():
    p = _random_powers(seed=4)
    sel = p[:, :, [0, 2, 3]]
    delta = 10 * np.log10(sel.sum(axis=2)) - 10 * np.log10(sel.mean(axis=2))
    np.testing.assert_allclose(delta, 10 * np.log10(3), rtol=1e-12)
