"""Unit tests for the controlled corruption models and mitigation strategies."""

import numpy as np
import pytest

from dropout_study.corruption import (
    MODELS,
    STRATEGIES,
    apply_strategy,
    chirp_keep_mask,
    corrupt_words,
)

N_CHIRP, WPC = 8, 16  # 128 words total


@pytest.fixture
def words():
    vals = (np.arange(N_CHIRP * WPC, dtype=np.uint64) * 4099 + 12345) % (2**32)
    return vals.astype(np.uint32).reshape(N_CHIRP, WPC)


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("severity_zero", [0, 0.0])
def test_severity_zero_is_exact_identity(words, model, severity_zero):
    w2, mask = corrupt_words(words, model, severity_zero, np.random.default_rng(7))
    assert np.array_equal(w2, words)
    assert not mask.any()
    assert w2.dtype == np.uint32 and mask.dtype == np.bool_


def test_random_drop_count_positions_and_values(words):
    w2, mask = corrupt_words(words, "random_drop", 0.1, np.random.default_rng(11))
    assert int(mask.sum()) == round(0.1 * words.size)
    assert (w2[mask] == 0).all()
    assert np.array_equal(w2[~mask], words[~mask])


@pytest.mark.parametrize(
    "position,expected_start",
    [("head", 0), ("middle", (128 - 32) // 2), ("tail", 128 - 32)],
)
def test_zero_segment_contiguous_at_position(words, position, expected_start):
    w2, mask = corrupt_words(words, "zero_segment", 32, np.random.default_rng(3),
                             position=position)
    flat = mask.reshape(-1)
    assert int(flat.sum()) == 32
    assert flat[expected_start:expected_start + 32].all()
    assert flat[:expected_start].sum() + flat[expected_start + 32:].sum() == 0


def test_chirp_zero_zeroes_whole_chirps(words):
    w2, mask = corrupt_words(words, "chirp_zero", 3, np.random.default_rng(5))
    assert int(mask.sum()) == 3 * WPC
    affected = np.where(mask.any(axis=1))[0]
    assert len(affected) == 3
    assert mask[affected, :].all()


def test_tail_trunc_marks_only_the_tail(words):
    w2, mask = corrupt_words(words, "tail_trunc", 0.25, np.random.default_rng(6))
    flat = mask.reshape(-1)
    k = round(0.25 * words.size)
    assert int(flat.sum()) == k
    assert flat[-k:].all()
    assert flat[:-k].sum() == 0


def test_same_seed_same_corruption(words):
    a = corrupt_words(words, "random_drop", 0.05, np.random.default_rng(42))
    b = corrupt_words(words, "random_drop", 0.05, np.random.default_rng(42))
    c = corrupt_words(words, "random_drop", 0.05, np.random.default_rng(43))
    assert np.array_equal(a[1], b[1])
    assert not np.array_equal(a[1], c[1])


def test_apply_strategy_zerofill_keeps_all_chirps(words):
    w2, mask = corrupt_words(words, "chirp_zero", 2, np.random.default_rng(9))
    frame, info = apply_strategy(w2, mask, "zerofill")
    assert frame.shape == (N_CHIRP, WPC * 2)  # int16 samples
    assert info == {"nchirp_used": N_CHIRP, "n_chirps_rejected": 0}


def test_apply_strategy_reject_drops_affected_chirps(words):
    w2, mask = corrupt_words(words, "chirp_zero", 2, np.random.default_rng(9))
    frame, info = apply_strategy(w2, mask, "reject")
    assert info == {"nchirp_used": N_CHIRP - 2, "n_chirps_rejected": 2}
    assert frame.shape == (N_CHIRP - 2, WPC * 2)


def test_apply_strategy_reject_returns_none_when_nothing_survives(words):
    all_missing = np.ones_like(words, dtype=bool)
    frame, info = apply_strategy(words, all_missing, "reject")
    assert frame is None
    assert info["nchirp_used"] == 0
    assert info["n_chirps_rejected"] == N_CHIRP


def test_apply_strategy_reject_single_word_in_a_chirp_is_enough(words):
    _, mask = corrupt_words(words, "random_drop", 1.0 / words.size, np.random.default_rng(2))
    assert int(mask.sum()) == 1
    frame, info = apply_strategy(words, mask, "reject")
    assert info["nchirp_used"] == N_CHIRP - 1


def test_chirp_keep_mask_intersects_channels():
    m1 = np.zeros((4, 8), dtype=bool)
    m1[1, 0] = True
    m2 = np.zeros((4, 8), dtype=bool)
    m2[2, 5] = True
    m2[3, :] = True
    keep = chirp_keep_mask([m1, m2])
    assert keep.tolist() == [True, False, False, False]


def test_unknown_inputs_raise(words):
    with pytest.raises(ValueError):
        corrupt_words(words, "no-such-model", 1, np.random.default_rng(1))
    with pytest.raises(ValueError):
        corrupt_words(words, "zero_segment", 4, np.random.default_rng(1), position="nowhere")
    with pytest.raises(ValueError):
        apply_strategy(words, np.zeros(words.shape, dtype=bool), "no-such-strategy")


def test_strategies_constant_matches_docs():
    assert STRATEGIES == ("zerofill", "reject")
