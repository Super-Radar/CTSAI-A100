"""ECDF inversion and Clopper-Pearson utilities."""

import numpy as np
from scipy.stats import beta as sdbeta

from cfar_calib.calibration import alpha_at_pfa, clopper_pearson


def test_alpha_at_pfa_exponential_sample():
    rng = np.random.default_rng(61)
    sample = rng.exponential(1.0, 400_000)
    # P(X > x) = exp(-x) = p  ->  x = -ln(p)
    for p in (1e-2, 1e-3):
        est = alpha_at_pfa(sample, p)
        assert abs(est - (-np.log(p))) / (-np.log(p)) < 0.03


def test_alpha_at_pfa_subset_denominator():
    rng = np.random.default_rng(62)
    full = rng.exponential(1.0, 200_000)
    subset = full[::2]                 # half the cells are "local maxima"
    n_total = full.size
    p = 1e-2
    a = alpha_at_pfa(subset, p, n_total)
    # require: #{subset > a} / n_total == p
    assert abs(np.count_nonzero(subset > a) / n_total - p) < 1e-3


def test_clopper_pearson_edges_and_midpoint():
    lo, hi = clopper_pearson(0, 1000)
    assert lo == 0.0
    assert np.isclose(hi, 1 - 0.025 ** (1 / 1000), rtol=1e-6)
    lo, hi = clopper_pearson(1000, 1000)
    assert hi == 1.0
    k, n, conf = 37, 1000, 0.95
    lo, hi = clopper_pearson(k, n, conf)
    ref_lo = sdbeta.ppf(0.025, k, n - k + 1)
    ref_hi = sdbeta.ppf(0.975, k + 1, n - k)
    assert np.isclose(lo, ref_lo) and np.isclose(hi, ref_hi)
    assert lo <= k / n <= hi
