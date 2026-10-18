"""Bootstrap tests: reference block length, quantile monotonicity, block bounds, statuses."""

import numpy as np

from temporal_ci.bootstrap import (
    block_length_reference,
    iid_bootstrap_means,
    mbb_indices,
    moving_block_bootstrap_means,
    percentile_ci,
    stationary_block_bootstrap_means,
    summarize_ci,
)


def test_block_length_reference_cubic_root_rule():
    assert block_length_reference(64) == 4
    assert block_length_reference(2456) == 13  # 2456^(1/3) = 13.48...
    assert block_length_reference(8000) == 20
    assert block_length_reference(10) == 2  # minimum of 2


def test_percentile_ci_monotone_in_level():
    rng = np.random.default_rng(21)
    x = rng.standard_normal(400) + 5.0
    means = iid_bootstrap_means(x, 2000, rng)
    center = float(np.median(means))
    prev = None
    for level in (0.50, 0.80, 0.95, 0.99):
        lo, hi = percentile_ci(means, level)
        assert lo <= center <= hi
        if prev is not None:
            assert lo <= prev[0]  # lower bound non-increasing as level grows
            assert hi >= prev[1]  # upper bound non-decreasing
        prev = (lo, hi)
    lo50, hi50 = percentile_ci(means, 0.50)
    lo99, hi99 = percentile_ci(means, 0.99)
    assert (hi50 - lo50) < (hi99 - lo99)


def test_mbb_indices_stay_in_bounds_and_form_blocks():
    rng = np.random.default_rng(33)
    n, b, B = 101, 7, 64
    idx = mbb_indices(n, b, B, rng)
    assert idx.shape == (B, n)
    assert idx.min() >= 0 and idx.max() <= n - 1  # blocks never cross the boundary
    nb = int(np.ceil(n / b))
    jumps = (np.diff(idx, axis=1) != 1).sum(axis=1)
    assert np.all(jumps <= nb - 1)  # at most one break per block boundary
    # breaks may only occur at block boundaries (positions b-1, 2b-1, ...)
    jump_cols = np.where(np.diff(idx, axis=1) != 1)[1]
    assert np.all(jump_cols % b == b - 1)


def test_mbb_rejects_invalid_block_length():
    rng = np.random.default_rng(1)
    try:
        mbb_indices(50, 51, 4, rng)
    except ValueError:
        pass
    else:
        raise AssertionError("block length > n must raise ValueError")


def test_sbb_means_finite_and_in_data_range():
    rng = np.random.default_rng(44)
    x = np.arange(50, dtype=float)
    m = stationary_block_bootstrap_means(x, 5.0, 32, rng)
    assert m.shape == (32,)
    assert np.all(np.isfinite(m))
    assert m.min() >= 0.0 and m.max() <= 49.0


def test_zero_variance_series_gets_status_not_ci():
    r = summarize_ci(np.full(100, 3.0), "iid", B=200, rng=np.random.default_rng(1))
    assert r["status"] == "zero_variance"
    assert r["lo"] is None and r["width"] is None


def test_insufficient_samples_below_30_frames():
    r = summarize_ci(np.arange(25, dtype=float), "mbb", B=100, rng=np.random.default_rng(2))
    assert r["status"] == "insufficient_samples"
    assert r["width"] is None


def test_iid_ci_covers_known_mean_around_nominal():
    rng = np.random.default_rng(55)
    hits = 0
    reps = 200
    for _ in range(reps):
        x = rng.standard_normal(300) + 2.0
        r = summarize_ci(x, "iid", B=399, rng=rng)
        hits += int(r["lo"] <= 2.0 <= r["hi"])
    assert 0.88 <= hits / reps <= 1.0  # deterministic under the fixed seed


def test_mbb_wider_than_iid_for_positively_correlated_series():
    rng = np.random.default_rng(66)
    x = np.empty(1200)
    x[0] = rng.standard_normal()
    for t in range(1, 1200):
        x[t] = 0.8 * x[t - 1] + rng.standard_normal()
    wi = percentile_ci(iid_bootstrap_means(x, 2000, rng))
    wm = percentile_ci(moving_block_bootstrap_means(x, 11, 2000, rng))
    assert (wm[1] - wm[0]) > 1.3 * (wi[1] - wi[0])
