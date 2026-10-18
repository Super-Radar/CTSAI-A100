"""Window-function unit tests: CG/ENBW vs closed-form theory, sidelobes, widths."""

import numpy as np

from window_study import windows

N = 1021  # the real A100 far-wave valid-sample count (odd on purpose)


def test_cg_enbw_match_theory_periodic():
    for name in windows.WINDOW_NAMES:
        w = windows.make_window(name, N)
        th = windows.analytic_metrics(name)
        assert abs(windows.coherent_gain(w) - th["cg"]) < 1e-6
        assert abs(windows.enbw_bins(w) - th["enbw_bins"]) < 1e-6


def test_rect_is_identity():
    w = windows.make_window("rect", N)
    assert windows.coherent_gain(w) == 1.0
    assert abs(windows.enbw_bins(w) - 1.0) < 1e-12


def test_symmetric_differs_but_close():
    per = windows.make_window("hann", N, sym=False)
    sym = windows.make_window("hann", N, sym=True)
    assert not np.allclose(per, sym)
    # symmetric == MATLAB hann(N): 0.5 - 0.5*cos(2*pi*n/(N-1))
    nn = np.arange(N)
    assert np.allclose(sym, 0.5 - 0.5 * np.cos(2.0 * np.pi * nn / (N - 1)))
    assert abs(windows.coherent_gain(sym) - windows.coherent_gain(per)) < 5e-4


def test_kernel_sidelobe_close_to_theory():
    # finite-N and periodic-definition effects shift the classical table by
    # at most a few tenths of a dB
    for name in windows.WINDOW_NAMES:
        w = windows.make_window(name, N)
        W = windows.window_kernel(w, 64 * N)
        meas = windows.kernel_sidelobe_db(W)
        th = windows.analytic_metrics(name)["sidelobe_db"]
        assert abs(meas - th) < 0.5, (name, meas, th)


def test_3db_width_ordering():
    widths = {
        name: windows.kernel_3db_width_bins(windows.make_window(name, N), 64 * N)
        for name in windows.WINDOW_NAMES
    }
    assert widths["rect"] < widths["hamming"] < widths["hann"] < widths["blackman"]
    assert 0.7 < widths["rect"] < 1.0
    assert 1.3 < widths["hann"] < 1.6


def test_measured_width_matches_kernel_prediction():
    # the -3 dB width of a bin-centred tone, measured on an 8x zero-padded
    # spectrum (frequency interpolation), equals the window-kernel prediction.
    # 1-bin sampling cannot do this: a bin-aligned rect tone has exactly zero
    # leakage at every unpadded DFT bin (see README, experiment 1).
    nfft = 1024
    n = np.arange(nfft)
    x = np.cos(2.0 * np.pi * 100 * n / nfft)
    for name in windows.WINDOW_NAMES:
        w = windows.make_window(name, nfft)
        xw = x * w
        X8 = np.fft.fft(xw, 8 * nfft)
        db8 = 20 * np.log10(np.abs(X8[: 4 * nfft]) + 1e-300)
        meas = windows.measured_3db_width_bins(db8, 800) / 8.0
        pred = windows.kernel_3db_width_bins(w, 8 * nfft)
        assert abs(meas - pred) < 0.05, (name, meas, pred)
