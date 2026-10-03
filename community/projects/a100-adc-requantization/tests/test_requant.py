"""Tests for the a100-adc-requantization study.

Covers the theoretical anchors (half-step bound, k=16 identity, truncation
bias, Delta^2/12 noise power), the unpack/pack round trip, the explicit
excess-value report of the loader, and end-to-end spectral behaviour
(tone bin placement, band-energy retention, weak-tone visibility loss).

Run: /home/cnj/openSource/.venv-a100/bin/python -m pytest tests/ -q
"""

from __future__ import annotations

import numpy as np
import pytest

from requant import SEED
from requant.config import INIT0, INIT1, SYNTH
from requant import fidelity as fid
from requant.pipeline import (LoadResult, get_windows, load_adc_txt,
                              pack_samples, spectral_chain, unpack_words)
from requant.requantize import quantization_error, requantize, step_for_k
from requant.synthetic import (TONE_RNG_BIN, TONE_VEL_BIN, WEAK_RNG_BIN,
                               WEAK_VEL_BIN, single_tone, two_tone)

VISIBLE_DB = 10.0


def _mc_samples(n=200_000, lo=-30_000, hi=30_000):
    rng = np.random.default_rng(SEED)
    return rng.integers(lo, hi, size=n, dtype=np.int64)


# ----------------------------------------------------------- configuration
def test_adc_sample_num_formula():
    """init0 -> 1020 valid samples (of 1024), init1 -> capped to 2048."""
    assert INIT0.adc_sample_num == 1020
    assert INIT1.adc_sample_num == 2048
    assert INIT0.words_per_channel == 512 * 256 == 131072
    assert INIT1.words_per_channel == 1024 * 128 == 131072


def test_hann_symmetric_matches_matlab():
    """The chain uses MATLAB hann(N): 0.5*(1-cos(2*pi*n/(N-1))), symmetric."""
    w, vw = get_windows(INIT0)
    n = np.arange(INIT0.adc_sample_num)
    ref = 0.5 * (1.0 - np.cos(2.0 * np.pi * n / (INIT0.adc_sample_num - 1)))
    assert np.allclose(w, ref, atol=1e-12)
    assert w.size == 1020 and vw.size == INIT0.nchirp == 256


# ------------------------------------------------------------ unpack/pack
def test_unpack_pack_roundtrip():
    """Random int16 frames survive pack -> unpack bit-exactly."""
    rng = np.random.default_rng(SEED)
    s = rng.integers(-32768, 32768, size=(SYNTH.rng_nfft, SYNTH.nchirp),
                     dtype=np.int64)
    words = pack_samples(s, SYNTH)
    assert words.dtype == np.uint32
    back = unpack_words(words, SYNTH)
    assert np.array_equal(back.astype(np.int64), s)


def test_unpack_word_split_order():
    """H16 first, L16 second, two's complement on both halves."""
    words = np.array([0x00010005, 0xFFFF8000, 0x7FFFFFF8], dtype=np.uint32)
    # SYNTH layout: nchirp=128 rows x half=512 -> pad to full matrix.
    full = np.zeros(SYNTH.words_per_channel, dtype=np.uint32)
    full[:3] = words
    s = unpack_words(full, SYNTH)
    col = s[:, 0]
    assert col[0] == 1 and col[1] == 5        # 0x0001, 0x0005
    assert col[2] == -1 and col[3] == -32768  # 0xFFFF, 0x8000
    assert col[4] == 32767 and col[5] == -8   # 0x7FFF, 0xFFF8


def test_load_reports_excess_and_pads_short(tmp_path):
    """Excess values are reported (not silently dropped); short files pad."""
    w = SYNTH.words_per_channel
    long_file = tmp_path / "long.txt"
    body = ",".join(["4267507368"] * w + ["7", "9", "11"])
    long_file.write_text(body, encoding="utf-8")
    lr = load_adc_txt(long_file, SYNTH)
    assert lr.had_excess and lr.excess_count == 3
    assert lr.excess_head[0] == 7          # first excess value reported
    assert lr.words.size == w              # truncated to reshape target
    short_file = tmp_path / "short.txt"
    short_file.write_text(",".join(["5"] * 100), encoding="utf-8")
    ls = load_adc_txt(short_file, SYNTH)
    assert ls.padded == w - 100 and ls.words.size == w
    assert ls.words[0] == 5 and ls.words[99] == 5 and ls.words[100] == 0


# ----------------------------------------------------------- requantize
@pytest.mark.parametrize("k", [4, 8, 12])
def test_round_error_bound_half_step(k):
    """No-clip region: |x - Q(x)| <= Delta/2 for round-half-away."""
    d = step_for_k(k)
    x = _mc_samples(lo=-30000, hi=30000)   # far from the +FS clip zone
    e = quantization_error(x, requantize(x, k, "round"))
    assert np.max(np.abs(e)) <= d / 2


def test_round_half_away_ties():
    """Exactly half a step rounds away from zero in both directions."""
    for k, d in ((15, 2), (10, 64), (4, 4096)):
        x = np.array([d // 2, -(d // 2), 0], dtype=np.int64)
        q = requantize(x, k, "round")
        assert q[0] == d and q[1] == -d and q[2] == 0


def test_k16_identity_both_modes():
    x = _mc_samples(lo=-32768, hi=32767)
    x[:3] = [-32768, 32767, 0]
    assert np.array_equal(requantize(x, 16, "round"), x)
    assert np.array_equal(requantize(x, 16, "trunc"), x)


def test_trunc_error_bounded_and_biased():
    """trunc (LSB drop = floor): error in [0, Delta), mean ~ +Delta/2."""
    k, d = 8, step_for_k(8)
    x = _mc_samples()
    e = quantization_error(x, requantize(x, k, "trunc"))
    assert np.min(e) >= 0.0 and np.max(e) < d
    assert abs(np.mean(e) - d / 2) < 0.02 * d    # +Delta/2 bias


def test_round_error_unbiased():
    k, d = 8, step_for_k(8)
    x = _mc_samples()
    e = quantization_error(x, requantize(x, k, "round"))
    assert abs(np.mean(e)) < 0.1 * d             # no DC bias


def test_round_noise_power_delta2_over_12():
    """MC (fixed seed): round-error power ~ Delta^2/12 within 3 %."""
    rng = np.random.default_rng(SEED)
    for k in (6, 10):
        d = step_for_k(k)
        x = rng.integers(-30000, 30000, size=400_000, dtype=np.int64)
        e = quantization_error(x, requantize(x, k, "round"))
        assert abs(np.mean(e ** 2) - d ** 2 / 12.0) <= 0.03 * d ** 2 / 12.0


def test_requant_never_leaves_int16_range():
    x = np.array([-32768, -32767, 32766, 32767] * 100, dtype=np.int64)
    for k in range(1, 17):
        for mode in ("round", "trunc"):
            q = requantize(x, k, mode)
            assert q.min() >= -32768 and q.max() <= 32767


def test_grid_points_are_fixed_points():
    """Values already on the k-bit grid pass through both laws unchanged.

    This is why k=14 requantisation is a bit-exact identity on the measured
    dumps: every sample of all 8 files is a multiple of 4 (the two least
    significant stored bits are always zero), i.e. the capture's storage
    granularity is already <= 14 bits.
    """
    rng = np.random.default_rng(SEED)
    for k in (4, 8, 12, 14, 16):
        d = step_for_k(k)
        x = rng.integers(-8, 8, size=1000, dtype=np.int64) * d
        assert np.array_equal(requantize(x, k, "round"), x)
        assert np.array_equal(requantize(x, k, "trunc"), x)


def test_requantize_rejects_bad_arguments():
    with pytest.raises(ValueError):
        requantize(np.zeros(4), 0)
    with pytest.raises(ValueError):
        requantize(np.zeros(4), 17)
    with pytest.raises(ValueError):
        requantize(np.zeros(4), 3.5)
    with pytest.raises(ValueError):
        requantize(np.zeros(4), 8, "floor")


# ------------------------------------------------------- spectral fidelity
def test_spectral_chain_places_tone_at_expected_bin():
    frame = single_tone(SYNTH.rng_nfft, SYNTH.nchirp, 12000)
    rng_spec, rd = spectral_chain(frame, SYNTH)
    assert rng_spec.shape == (SYNTH.rng_nfft // 2,)
    assert rd.shape == (SYNTH.rng_nfft // 2, SYNTH.vel_nfft)
    pk = np.unravel_index(int(np.argmax(rd)), rd.shape)
    assert pk == (TONE_RNG_BIN, TONE_VEL_BIN)


def test_identity_requantization_preserves_spectra_and_band_energy():
    frame = single_tone(SYNTH.rng_nfft, SYNTH.nchirp, 12000)
    q = requantize(frame, 16, "round")
    r0, m0 = spectral_chain(frame, SYNTH)
    r1, m1 = spectral_chain(q, SYNTH)
    assert np.array_equal(r0, r1) and np.array_equal(m0, m1)
    band = fid.predefined_band(m0, frac=0.01)
    assert band.sum() == max(1, int(round(0.01 * m0.size)))
    assert fid.band_energy_retention(m1, m0, band) == 1.0


def test_peak_stable_at_high_k():
    frame = single_tone(SYNTH.rng_nfft, SYNTH.nchirp, 12000)
    _, base = spectral_chain(frame, SYNTH)
    for mode in ("round", "trunc"):
        _, mk = spectral_chain(requantize(frame, 12, mode), SYNTH)
        drift = fid.peak_drift(mk, base)
        assert drift["rng_bin_shift"] == 0 and drift["vel_bin_shift"] == 0
        assert 0.98 <= drift["peak_amp_ratio"] <= 1.02


def test_two_tone_visibility_degrades_monotonically_with_k():
    """Visibility margin shrinks as k drops (68 dB class -> 34 dB class)."""
    frame = two_tone(SYNTH.rng_nfft, SYNTH.nchirp, SEED)
    vis = []
    for k in (16, 12, 10, 8, 6, 4):
        _, m = spectral_chain(requantize(frame, k, "round"), SYNTH)
        vis.append(fid.weak_tone_visibility(m, WEAK_RNG_BIN, WEAK_VEL_BIN)
                   ["ratio_db"])
    assert vis[0] >= 60.0                       # clearly visible @ 16 bit
    assert vis[-1] <= vis[0] - 20.0             # >= 20 dB margin lost @ 4 bit
    assert all(vis[i] + 1.0 >= vis[i + 1] for i in range(len(vis) - 1))


def test_weak_tone_alone_annihilated_below_half_step():
    """Without a carrier a sub-half-step tone is rounded to exactly zero.

    A=38: k<=8 (Delta>=256, Delta/2 >= 128 > 38) -> all samples 0 -> the
    tone bin is empty; k=10 (Delta=64 < 2*38) -> it survives.  With the
    two-tone carrier present the same weak tone instead survives via
    level-crossing modulation (PWM-like sub-LSB transfer) -- see
    test_two_tone_visibility_degrades_monotonically_with_k and summary.md.
    """
    frame = single_tone(SYNTH.rng_nfft, SYNTH.nchirp, 38,
                        rng_bin=WEAK_RNG_BIN, vel_bin=WEAK_VEL_BIN)
    q8 = requantize(frame, 8, "round")
    assert np.count_nonzero(q8) == 0
    _, m8 = spectral_chain(q8, SYNTH)
    assert m8[WEAK_RNG_BIN, WEAK_VEL_BIN] == 0.0
    q10 = requantize(frame, 10, "round")
    _, m10 = spectral_chain(q10, SYNTH)
    assert m10[WEAK_RNG_BIN, WEAK_VEL_BIN] > 0.0


def test_band_retention_cliff_and_overshoot():
    """Signal-band retention: hard 0 below half step, >1 overshoot just above.

    A=188: k=6 (Delta=1024, Delta/2=512 > 188) -> R == 0; k=8 (Delta=256,
    Delta/2=128 < 188) the tone is barely above half a step, converts
    towards a square wave and the band energy overshoots (R ~ 1.6,
    distortion not preservation); at k=14 it is back within a few percent.
    """
    frame = single_tone(SYNTH.rng_nfft, SYNTH.nchirp, 188)
    _, base = spectral_chain(frame, SYNTH)
    band = fid.signal_band(base.shape,
                           [(TONE_RNG_BIN, TONE_VEL_BIN, 2, 2)])
    ret = {}
    for k in (6, 8, 14):
        _, mk = spectral_chain(requantize(frame, k, "round"), SYNTH)
        ret[k] = fid.band_energy_retention(mk, base, band)
    assert ret[6] == 0.0
    assert ret[8] > 1.2
    assert 0.98 <= ret[14] <= 1.05
