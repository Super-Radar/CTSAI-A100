"""Zero-perturbation regression tests and metric sanity checks.

These are the pytest counterparts of experiment 1: any corruption model at
severity 0 must leave the word stream untouched (exact identity) and every
spectral metric must read exactly 0 / 1 against the clean baseline.
"""

from dataclasses import replace

import numpy as np
import pytest

from dropout_study import metrics as M
from dropout_study.corruption import MODELS, STRATEGIES, apply_strategy, corrupt_words
from dropout_study.pipeline import INIT0, rv_power_map, synth_single_target_words, unpack_words

# Reduced chirp count keeps the identity regression fast; results are exact
# regardless of the frame size because nothing is corrupted at all.
FAST0 = replace(INIT0, nchirp=64)


@pytest.fixture(scope="module")
def synth_words():
    return synth_single_target_words(FAST0, 5.0, 12 * FAST0.vel_per_bin_mps)


@pytest.fixture(scope="module")
def ref_map(synth_words):
    return rv_power_map(unpack_words(synth_words), FAST0)


@pytest.mark.parametrize("strategy", STRATEGIES)
@pytest.mark.parametrize("model", MODELS)
def test_zero_severity_all_metrics_identity(synth_words, ref_map, model, strategy):
    w2, mask = corrupt_words(synth_words, model, 0, np.random.default_rng(1))
    assert np.array_equal(w2, synth_words)
    assert not mask.any()
    frame, _ = apply_strategy(w2, mask, strategy)
    mm = M.compare_maps(ref_map, rv_power_map(frame, FAST0))
    assert mm["drift_range_bins"] == 0.0
    assert mm["drift_doppler_bins"] == 0.0
    assert mm["drift_max_bins"] == 0.0
    assert mm["peak_power_error_db"] == pytest.approx(0.0, abs=1e-9)
    assert mm["total_energy_error_db"] == pytest.approx(0.0, abs=1e-9)
    assert mm["spectral_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert mm["doppler_spurious_delta"] == 0


def test_scaled_map_moves_only_power(ref_map):
    mm = M.compare_maps(ref_map, 4.0 * ref_map)
    assert mm["peak_power_error_db"] == pytest.approx(10.0 * np.log10(4.0), abs=1e-9)
    assert mm["total_energy_error_db"] == pytest.approx(10.0 * np.log10(4.0), abs=1e-9)
    assert mm["drift_max_bins"] == pytest.approx(0.0, abs=1e-9)
    assert mm["spectral_correlation"] == pytest.approx(1.0, abs=1e-12)
    assert mm["doppler_spurious_delta"] == 0


def test_shifted_map_moves_only_position(ref_map):
    shifted = np.roll(ref_map, 3, axis=0)
    mm = M.compare_maps(ref_map, shifted)
    assert mm["drift_range_bins"] == pytest.approx(3.0, abs=0.6)
    assert abs(mm["peak_power_error_db"]) < 1e-6


def test_refined_peak_recovers_subbin_position(ref_map):
    # A pure 3-bin roll must refine to a drift of almost exactly 3 bins.
    r0, d0 = M.refined_peak(ref_map)
    r1, d1 = M.refined_peak(np.roll(ref_map, 3, axis=0))
    assert (r1 - r0) == pytest.approx(3.0, abs=0.35)
    assert (d1 - d0) == pytest.approx(0.0, abs=1e-9)


def test_synthetic_frame_peaks_on_target(ref_map):
    # The synthetic single target must dominate the map at the theory bins.
    r, d = M.refined_peak(ref_map)
    assert abs(r - FAST0.range_bin_of(5.0)) <= 1.0
    assert abs(d - 12.0) <= 1.0
