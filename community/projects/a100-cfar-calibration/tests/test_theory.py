"""Threshold-factor <-> Pfa conversion tests (iid cell model)."""

import numpy as np
import pytest

from cfar_calib.calibration import N_TRAIN
from cfar_calib.cfar import (
    alpha_from_pfa_single_look,
    alpha_k_look,
    kernel_count,
    pfa_k_look,
    pfa_single_look,
)

PFAS = (1e-2, 1e-3, 1e-4)


def test_kernel_count_matches_validator_defaults():
    # (2*(8+2)+1)*(2*(6+2)+1) - (2*2+1)*(2*2+1) = 21*17 - 25 = 332
    assert kernel_count(8, 6, 2, 2) == 332
    assert N_TRAIN == 332


@pytest.mark.parametrize("pfa", PFAS)
def test_single_look_roundtrip(pfa):
    alpha = alpha_from_pfa_single_look(pfa, N_TRAIN)
    assert np.isclose(pfa_single_look(alpha, N_TRAIN), pfa, rtol=1e-12)


def test_single_look_closed_form():
    # independent recomputation of the textbook formula
    pfa, n = 1e-4, 332
    expected = n * (pfa ** (-1.0 / n) - 1.0)
    assert np.isclose(alpha_from_pfa_single_look(pfa, n), expected, rtol=1e-12)
    assert 8.0 < alpha_from_pfa_single_look(pfa, n) < 11.0  # sanity magnitude


def test_k_look_reduces_to_single_look():
    a = 7.3
    assert np.isclose(pfa_k_look(a, N_TRAIN, 1), pfa_single_look(a, N_TRAIN),
                      rtol=1e-10)


def test_pfa_monotone_decreasing_in_alpha_and_k():
    for k in (1, 2, 3, 4):
        alphas = np.geomspace(0.5, 15, 25)
        vals = np.array([pfa_k_look(a, N_TRAIN, k) for a in alphas])
        diffs = np.diff(vals)
        # strictly decreasing until the tail underflows to exactly 0
        assert all((diffs < 0) | (vals[1:] == 0.0))
    a = 7.3
    vals = [pfa_k_look(a, N_TRAIN, k) for k in (1, 2, 3, 4)]
    assert all(np.diff(vals) < 0)


@pytest.mark.parametrize("k", (1, 2, 3, 4))
@pytest.mark.parametrize("pfa", PFAS)
def test_alpha_k_look_roundtrip(k, pfa):
    a = alpha_k_look(pfa, N_TRAIN, k)
    assert np.isclose(pfa_k_look(a, N_TRAIN, k), pfa, rtol=1e-6, atol=1e-12)


@pytest.mark.parametrize("pfa", PFAS)
def test_alpha_needed_shrinks_with_k(pfa):
    alphas = [alpha_k_look(pfa, N_TRAIN, k) for k in (1, 2, 3, 4)]
    assert all(np.diff(alphas) < 0)
    # K=4 vs K=1 is a large correction (the headline of the study)
    assert alphas[3] < 0.6 * alphas[0]
