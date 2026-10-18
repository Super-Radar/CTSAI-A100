"""On iid exponential cells the threshold rule must hit the exact formula."""

import numpy as np

from cfar_calib.calibration import N_TRAIN
from cfar_calib.cfar import (
    alpha_from_pfa_single_look,
    ca_noise_estimate,
    interior_mask,
    local_max_mask,
    pfa_single_look,
)
from cfar_calib.calibration import clopper_pearson


def collect_iid_ratios(seed, frames, shape=(220, 160)):
    rng = np.random.default_rng(seed)
    parts = []
    for _ in range(frames):
        power = rng.exponential(1.0, shape)
        ratio = power / ca_noise_estimate(power)
        parts.append(ratio[interior_mask(shape)])
    return np.concatenate(parts)


def test_threshold_rule_pfa_within_exact_interval():
    ratios = collect_iid_ratios(21, frames=10)
    n = ratios.size
    assert n > 300_000
    for pfa in (1e-2, 1e-3):
        alpha = alpha_from_pfa_single_look(pfa, N_TRAIN)
        k = int(np.count_nonzero(ratios > alpha))
        lo, hi = clopper_pearson(k, n, conf=0.999)
        exact = pfa_single_look(alpha, N_TRAIN)
        assert lo <= exact <= hi, (
            f"pfa={pfa}: empirical {k}/{n} = {k/n:.2e}, "
            f"CP99.9% [{lo:.2e}, {hi:.2e}], exact {exact:.2e}")
        assert abs(k / n - exact) / exact < 0.12


def test_local_max_never_increases_alarms():
    rng = np.random.default_rng(22)
    power = rng.exponential(1.0, (120, 90))
    valid = interior_mask(power.shape)
    lm = local_max_mask(power)
    alpha = alpha_from_pfa_single_look(1e-3, N_TRAIN)
    above = power > alpha * ca_noise_estimate(power)
    assert int((above & lm & valid).sum()) <= int((above & valid).sum())
