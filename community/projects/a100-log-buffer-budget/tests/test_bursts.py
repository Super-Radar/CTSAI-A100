"""Burst-metric tests with hand-computed expectations."""

import numpy as np

from buffer_budget import bursts


def test_sliding_window_counts_exact():
    t = np.array([0.00, 0.04, 0.10, 0.14, 0.20])
    b = np.full(5, 8, dtype=int)
    centers, mr, br = bursts.sliding_window_rates(t, b, window_s=0.1, hop_s=0.1)
    assert np.allclose(centers, [0.05, 0.15])
    assert np.allclose(mr, [20.0, 20.0])  # 2 messages / 0.1 s
    assert np.allclose(br, [160.0, 160.0])  # 2 x 8 bytes / 0.1 s


def test_windows_are_half_open():
    # A message exactly at a window edge belongs to the next window: the
    # timestamps 0.1, 0.1 fall into [0.1, 0.2), not into [0, 0.1); the
    # message at 0.25 lies beyond the last full window and is not counted.
    t = np.array([0.0, 0.1, 0.1, 0.25])
    b = np.ones(4, dtype=int)
    _, mr, _ = bursts.sliding_window_rates(t, b, window_s=0.1, hop_s=0.1)
    assert mr.tolist() == [10.0, 20.0]  # [0,0.1): 1 msg; [0.1,0.2): 2 msgs


def test_short_log_uses_single_anchored_window():
    t = np.array([1.0, 1.0, 1.0])
    b = np.ones(3, dtype=int)
    centers, mr, br = bursts.sliding_window_rates(t, b, window_s=1.0, hop_s=0.1)
    assert centers.tolist() == [1.5]
    assert mr.tolist() == [3.0]


def test_window_scale_stats_and_burst_factor():
    t = np.array([0.00, 0.04, 0.10, 0.14, 0.20])
    b = np.full(5, 8, dtype=int)
    scale = bursts.window_scale_stats(t, b, windows=(0.1,))
    assert scale[0.1]["peak_msg_s"] == 20.0  # any 0.1 s window holds <= 2 msgs
    mean_msg, mean_bytes, duration = bursts.mean_rates(t, b)
    assert duration == 0.2
    assert mean_msg == 25.0  # 5 msgs / 0.2 s
    assert mean_bytes == 200.0
    assert bursts.burst_factor(20.0, 25.0) == 0.8


def test_per_id_counts_sorted_and_normalized():
    ids, counts, fracs = bursts.per_id_counts(np.array([5, 5, 5, 2, 9]))
    assert ids.tolist() == [5, 2, 9]
    assert counts.tolist() == [3, 1, 1]
    assert fracs.sum() == 1.0
    assert np.all(np.diff(counts) <= 0)


def test_batch_stats():
    n_batches, max_batch, mean_batch = bursts.batch_stats(
        np.array([0.0, 0.0, 0.0, 1.0, 1.0, 2.0]))
    assert n_batches == 3
    assert max_batch == 3
    assert mean_batch == 2.0


def test_cv_1s_near_zero_for_flat_stream_and_large_for_bursts():
    # Perfectly uniform one-message-per-0.1 s stream: all 1 s window rates are
    # 10 msg/s up to +-1 message of float-boundary noise in searchsorted.
    t = np.arange(100) * 0.1
    b = np.ones(100, dtype=int)
    assert bursts.cv_1s(t, b) < 0.05
    # Alternating 5/15-message batches, one batch per second -> CV ~ 0.5.
    sizes = np.tile([5, 15], 5)
    t2 = np.repeat(np.arange(10) * 1.0, sizes)
    assert bursts.cv_1s(t2, np.ones(t2.size, dtype=int)) > 0.3
