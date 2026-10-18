"""Wordlength and behaviour tests for the fixed-point FFT simulator.

Every test asserts numerical properties of the bit-exact integer simulation;
no timing or memory properties are tested anywhere (machine independence).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from fixed_fft import budget, inputs as fx_inputs
from fixed_fft.fft_fixed import (
    SCALING_NONE,
    SCALING_SCALE_1BIT,
    clip_to_word,
    fft_fixed,
    fit_input,
    twiddle_quantization_error,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
ADC_FILE = REPO_ROOT / "ADC数据采集" / "示例adc数据和结果" / fx_inputs.DATA_FILES[0]


# ---------------------------------------------------------------------------
# DC / scaling structure
# ---------------------------------------------------------------------------

def test_dc_concentrates_in_bin0_exactly():
    n, a = 256, 30000
    r = fft_fixed(fx_inputs.dc_input(n, a), n_fft=n, B=24, Q=16,
                  scaling=SCALING_NONE)
    assert r.re[0] == a * n          # 7,680,000: exact integer DFT of DC
    assert np.count_nonzero(r.re) == 1 and np.all(r.im == 0)
    assert r.total_overflow == 0


def test_dc_with_per_stage_scaling_keeps_amplitude():
    n, a = 256, 30000
    r = fft_fixed(fx_inputs.dc_input(n, a), n_fft=n, B=24, Q=16,
                  scaling=SCALING_SCALE_1BIT)
    # every stage doubles then halves: the stored DC value stays at a,
    # i.e. the 1-bit-per-stage schedule exactly compensates the stage gain
    assert r.re[0] == a
    assert np.all(r.re[1:] == 0) and np.all(r.im == 0)
    assert r.effective_gain() == float(n)


# ---------------------------------------------------------------------------
# tone placement
# ---------------------------------------------------------------------------

def test_quarter_bin_tone_lands_exactly():
    n, k, a = 256, 64, 10000            # k = n/4: cos values are integers
    x = np.rint(a * np.cos(2.0 * np.pi * k * np.arange(n) / n)).astype(np.int64)
    r = fft_fixed(x, n_fft=n, B=24, Q=16, scaling=SCALING_NONE)
    assert r.re[k] == a * n // 2        # exact magnitude, calibrated in LSB
    assert r.re[n - k] == a * n // 2    # conjugate bin
    assert np.count_nonzero(r.re) == 2 and np.all(r.im == 0)


def test_generic_integer_bin_tone_places_peak():
    n, k, a = 1024, 37, 20000
    x = np.rint(a * np.cos(2.0 * np.pi * k * np.arange(n) / n)).astype(np.int64)
    r = fft_fixed(x, n_fft=n, B=26, Q=16, scaling=SCALING_NONE)
    xfix = r.spectrum() * r.effective_gain()
    power = np.abs(xfix[: n // 2 + 1])
    i = int(np.argmax(power))
    assert i == k                                        # exact bin placement
    ref = np.abs(np.fft.fft(x.astype(float), n))[i]
    assert abs(power[i] - ref) / ref < 0.01              # amplitude fidelity
    assert power[np.arange(len(power)) != i].max() / power[i] < 0.02  # leakage


# ---------------------------------------------------------------------------
# float64 convergence and identity
# ---------------------------------------------------------------------------

def test_high_wordlength_converges_to_float_reference():
    x = fx_inputs.bandlimited_noise(1024, seed=20261003)
    row = budget.run_case("noise", x, B=28, Q=24, scaling=SCALING_NONE)
    assert row["rel_l2_power"] < 1e-3
    assert row["input_shift"] == 0 and row["total_overflow"] == 0


def test_identity_special_grid_matches_exact_integer_dft():
    n = 16
    x = np.rint(1000 * np.cos(2.0 * np.pi * 4 * np.arange(n) / n) + 500).astype(np.int64)
    r = fft_fixed(x, n_fft=n, B=20, Q=8, scaling=SCALING_NONE)
    expected = np.zeros(n, dtype=np.int64)
    expected[0] = 500 * 16      # DC part: N * mean
    expected[4] = 8000          # 1000 * N/2, bin +N/4
    expected[12] = 8000         # conjugate bin
    assert np.array_equal(r.re, expected)
    assert np.all(r.im == 0)


def test_generic_small_identity_against_numpy():
    rng = np.random.default_rng(7)
    x = rng.integers(-32768, 32768, size=32).astype(np.int64)
    r = fft_fixed(x, n_fft=32, B=32, Q=28, scaling=SCALING_NONE)
    got = r.spectrum()
    ref = np.fft.fft(x.astype(float), 32)
    # residual is the <=1-LSB-per-multiply truncation of the integer datapath
    assert np.max(np.abs(got - ref)) / np.max(np.abs(ref)) < 1e-4


# ---------------------------------------------------------------------------
# overflow accounting
# ---------------------------------------------------------------------------

def test_overflow_counted_and_saturated():
    n, a = 256, 32767                     # exact X[0] = 2^23.0 - 256: needs 24 bits
    r = fft_fixed(fx_inputs.dc_input(n, a), n_fft=n, B=20, Q=16, scaling=SCALING_NONE)
    assert r.total_overflow > 0
    assert len(r.stage_overflow) == 8
    assert r.re[0] == 2 ** 19 - 1         # saturated rail


def test_wrap_mode_matches_two_complement_modular_result():
    n, a = 256, 32767
    exact = a * n                          # 8,388,352
    u = exact % (1 << 20)
    wrapped = u - (1 << 20) if u >= (1 << 19) else u   # sign-extended residue
    r = fft_fixed(fx_inputs.dc_input(n, a), n_fft=n, B=20, Q=16,
                  scaling=SCALING_NONE, overflow="wrap")
    assert r.re[0] == wrapped
    assert r.total_overflow > 0


# ---------------------------------------------------------------------------
# scaling invariants
# ---------------------------------------------------------------------------

def test_scaling_energy_invariant_without_saturation():
    x = fx_inputs.bandlimited_noise(512, seed=11, level=0.06)   # ~2000 LSB peak
    n = 512
    r = fft_fixed(x, n_fft=n, B=24, Q=16, scaling=SCALING_SCALE_1BIT)
    assert r.total_overflow == 0
    xfix = r.spectrum() * r.effective_gain()
    ratio = float(np.sum(np.abs(xfix) ** 2) / np.sum(np.abs(np.fft.fft(x.astype(float), n)) ** 2))
    assert 0.99 < ratio < 1.01            # energy conserved once the N gain is undone


def test_scaling_keeps_dc_stage_peak_constant():
    n, a = 1024, 4096
    r = fft_fixed(fx_inputs.dc_input(n, a), n_fft=n, B=24, Q=16,
                  scaling=SCALING_SCALE_1BIT)
    assert all(p == a for p in r.stage_peak)   # halving cancels doubling exactly


# ---------------------------------------------------------------------------
# symbol boundaries and input fitting
# ---------------------------------------------------------------------------

def test_symbol_boundary_values_representable_and_clipped():
    sb = fx_inputs.symbol_boundary(64)
    xq, shift, clipped = fit_input(sb, 16)
    assert shift == 0 and clipped == 0      # -32768 needs no shift
    assert xq.min() == -32768 and xq.max() == 32767
    clipped_vals = clip_to_word(np.array([32768, -32769, 32767, -32768]), 16)
    assert list(clipped_vals) == [32767, -32768, 32767, -32768]


def test_input_pre_shift_recorded_for_small_b():
    xq, shift, clipped = fit_input(np.array([16000, -32000, 7]), 8)
    assert shift == 8                      # 32000 needs 16 magnitude bits
    assert clipped == 0 and np.all(np.abs(xq) <= 127)


def test_symbol_boundary_fft_overflows_at_native_width():
    r = fft_fixed(fx_inputs.symbol_boundary(64), n_fft=64, B=16, Q=16,
                  scaling=SCALING_NONE)
    assert r.total_overflow > 0            # alternating +/- full scale needs ~7 headroom bits


# ---------------------------------------------------------------------------
# twiddle quantization
# ---------------------------------------------------------------------------

def test_twiddle_quantization_roundtrip_error_bounded():
    assert twiddle_quantization_error(1024, 16) < 2.0 ** -16
    assert twiddle_quantization_error(1024, 24) < 2.0 ** -24


# ---------------------------------------------------------------------------
# parameter validation
# ---------------------------------------------------------------------------

def test_parameter_validation_rejects_out_of_range():
    with pytest.raises(ValueError):
        fft_fixed(np.zeros(8), B=33)
    with pytest.raises(ValueError):
        fft_fixed(np.zeros(8), B=32, Q=32)   # B+Q exceeds int64 headroom guard
    with pytest.raises(ValueError):
        fft_fixed(np.zeros(8), scaling="sometimes")


# ---------------------------------------------------------------------------
# official data chain
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not ADC_FILE.is_file(), reason="official ADC dump not present")
def test_official_adc_chain_shapes():
    samples = fx_inputs.adc_words_to_samples(fx_inputs.load_adc_words(ADC_FILE))
    assert samples.shape == (256, 1024)
    assert samples.dtype == np.int64
    assert np.abs(samples).max() <= 32768   # int16 code range
    row = fx_inputs.official_range_fft_input(samples[0])
    assert row.shape == (1024,)
    assert np.count_nonzero(row[1020:]) == 0  # hann(1020) + 4 zero pads
