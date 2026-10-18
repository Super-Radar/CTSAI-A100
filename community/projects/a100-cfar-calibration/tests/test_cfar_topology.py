"""The vectorised CFAR operators must reproduce the validator's own loops."""

import numpy as np
import pytest

from cfar_calib.cfar import ca_noise_estimate, cfar_detect


def validator_reference(power, training=(8, 6), guard=(2, 2), pfa=1e-4):
    """Verbatim logic of run_reference_validation.py::cfar (loops kept)."""
    tr, td = training
    gr, gd = guard
    kernel = np.ones((2 * (tr + gr) + 1, 2 * (td + gd) + 1))
    kernel[tr:tr + 2 * gr + 1, td:td + 2 * gd + 1] = 0
    count = kernel.sum()
    padded = np.pad(power, ((tr + gr, tr + gr), (0, 0)), mode="constant")
    padded = np.pad(padded, ((0, 0), (td + gd, td + gd)), mode="wrap")
    noise = np.zeros_like(power)
    for r, d in np.argwhere(kernel):
        noise += padded[r:r + power.shape[0], d:d + power.shape[1]]
    noise /= count
    alpha = count * (pfa ** (-1 / count) - 1)
    mask = power > alpha * noise
    local = np.ones_like(mask)
    for dr in (-1, 0, 1):
        for dd in (-1, 0, 1):
            if dr or dd:
                local &= power >= np.roll(power, (dr, dd), axis=(0, 1))
    mask &= local
    border_r = tr + gr
    mask[:border_r] = mask[-border_r:] = False
    return mask, noise, alpha


@pytest.mark.parametrize("training,guard",
                         [((8, 6), (2, 2)), ((4, 3), (1, 2)), ((2, 2), (0, 1))])
def test_noise_estimate_matches_validator(training, guard):
    rng = np.random.default_rng(11)
    power = rng.exponential(1.0, (60, 40))
    ref = validator_reference(power, training, guard)[1]
    mine = ca_noise_estimate(power, training, guard)
    np.testing.assert_allclose(mine, ref, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("pfa", [1e-2, 1e-4])
def test_detection_matches_validator(pfa):
    rng = np.random.default_rng(12)
    power = rng.exponential(1.0, (80, 50))
    ref_mask, ref_noise, alpha = validator_reference(power, pfa=pfa)
    mask, noise = cfar_detect(power, alpha)
    np.testing.assert_array_equal(mask, ref_mask)


def test_target_at_doppler_wrap_boundary_detected():
    # same probe the validator regression-checks: periodic Doppler boundary
    probe = np.ones((64, 64))
    probe[32, 0] = 1000.0
    from cfar_calib.cfar import alpha_from_pfa_single_look, kernel_count
    alpha = alpha_from_pfa_single_look(1e-3, kernel_count(4, 4, 1, 1))
    mask, _ = cfar_detect(probe, alpha, training=(4, 4), guard=(1, 1))
    assert mask[32, 0]


def test_range_border_decisions_excluded():
    rng = np.random.default_rng(13)
    power = rng.exponential(1.0, (40, 30)) * 0 + 1.0
    power[0, 5] = 1e6       # inside excluded border rows
    power[20, 5] = 1e6      # interior
    mask, _ = cfar_detect(power, 1.0)
    assert not mask[0, 5]
    assert mask[20, 5]


def test_db_ring_stats_match_brute_force():
    from cfar_calib.cfar import TRAINING, GUARD, db_ring_stats, kernel_count

    rng = np.random.default_rng(14)
    tr, td = TRAINING
    gr, gd = GUARD
    orr, od = tr + gr, td + gd
    n = kernel_count()
    db = 10 * np.log10(rng.exponential(1.0, (80, 60)))
    mu, sd = db_ring_stats(db)
    # brute-force ring statistics on a few interior cells
    for r in (20, 40, 55):
        for d in (10, 30, 50):
            vals = []
            for dr in range(-orr, orr + 1):
                for dd in range(-od, od + 1):
                    if abs(dr) <= gr and abs(dd) <= gd:
                        continue
                    vals.append(db[r + dr, (d + dd) % db.shape[1]])
            vals = np.array(vals)
            assert len(vals) == n
            assert np.isclose(mu[r, d], vals.mean(), rtol=1e-10)
            assert np.isclose(sd[r, d], vals.std(), rtol=1e-8, atol=1e-9)
