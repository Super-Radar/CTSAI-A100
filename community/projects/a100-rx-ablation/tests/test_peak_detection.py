# -*- coding: utf-8 -*-
"""峰检测小回归 + 稳定性指标。"""
import numpy as np

from rx_ablation import metrics


def _img_with_bumps(shape=(40, 30), bumps=((5, 6, 100.0), (20, 25, 60.0), (30, 3, 30.0))):
    img = np.ones(shape)
    for r, v, h in bumps:
        img[r, v] = h
        img[r, v + 1] = h / 2.0      # 相邻次高点：应被 NMS 压掉（同一峰）
    return img


def test_detects_known_local_maxima():
    peaks = metrics.detect_peaks(
        _img_with_bumps(), margin_db=10.0, min_rng_sep=2, min_vel_sep=2
    )
    got = {(p.rng_bin, p.vel_bin) for p in peaks}
    assert got == {(5, 6), (20, 25), (30, 3)}


def test_threshold_filters_weak_bumps():
    # 噪声底 0 dB；30 倍功率 = 14.8 dB < margin=16 dB -> 不检出
    peaks = metrics.detect_peaks(
        _img_with_bumps(), margin_db=16.0, min_rng_sep=2, min_vel_sep=2
    )
    got = {(p.rng_bin, p.vel_bin) for p in peaks}
    assert got == {(5, 6), (20, 25)}


def test_nms_merges_close_peaks():
    img = np.ones((20, 20))
    img[10, 10] = 100.0
    img[11, 11] = 80.0          # 距离 (1,1) < 最小分离 (3,3) -> 被压掉
    peaks = metrics.detect_peaks(img, margin_db=10.0, min_rng_sep=3, min_vel_sep=3)
    assert [(p.rng_bin, p.vel_bin) for p in peaks] == [(10, 10)]


def test_peaks_sorted_by_db_and_capped():
    img = _img_with_bumps()
    peaks = metrics.detect_peaks(img, margin_db=10.0, min_rng_sep=2, min_vel_sep=2, top_n=2)
    assert len(peaks) == 2
    assert peaks[0].db >= peaks[1].db


def test_jaccard_identity_and_disjoint():
    img = _img_with_bumps()
    a = metrics.detect_peaks(img, margin_db=10.0, min_rng_sep=2, min_vel_sep=2)
    assert metrics.peak_jaccard(a, a) == 1.0
    shifted = [type(a[0])(p.rng_bin + 20, p.vel_bin + 20, p.db) for p in a]
    assert metrics.peak_jaccard(a, shifted) == 0.0
    one_off = [type(a[0])(p.rng_bin + 1, p.vel_bin, p.db) for p in a]
    assert metrics.peak_jaccard(a, one_off, rng_tol=2, vel_tol=2) == 1.0


def test_strongest_peak_jitter_axes():
    img = _img_with_bumps()
    a = metrics.detect_peaks(img, margin_db=10.0, min_rng_sep=2, min_vel_sep=2)
    b = [type(a[0])(a[0].rng_bin + 3, a[0].vel_bin - 2, a[0].db)] + a[1:]
    assert metrics.strongest_peak_jitter(a, b) == (3, -2)
    assert metrics.chebyshev_jitter((3, -2)) == 3


def test_noise_floor_is_median():
    img = np.ones((10, 10))
    img[5, 5] = 1000.0
    assert np.isclose(metrics.noise_floor_db(img), 0.0, atol=1e-9)
