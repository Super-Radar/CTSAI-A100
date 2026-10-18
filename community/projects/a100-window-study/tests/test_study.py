"""Study-level tests: synthetic tone recovery, Parseval identities, dB-convention
offsets, and a small end-to-end smoke run of experiment 1."""

import numpy as np

from window_study import study, windows


def test_dual_tone_amplitude_recovery():
    nfft = 1024
    n = np.arange(nfft)
    tones = ((100, 1.0), (300, 0.25))
    x = np.zeros(nfft)
    for k, a in tones:
        x += a * np.cos(2.0 * np.pi * k * n / nfft)
    for name in windows.WINDOW_NAMES:
        w = windows.make_window(name, nfft)
        X = np.fft.fft(x * w)
        for k, a in tones:
            a_est = 2.0 * abs(X[k]) / w.sum()
            assert abs(a_est - a) / a < 1e-6, (name, k, abs(a_est - a) / a)


def test_parseval_complex_exponential_exact():
    nfft = 1024
    n = np.arange(nfft)
    z = np.exp(2j * np.pi * 77 * n / nfft)  # constant modulus
    for name in windows.WINDOW_NAMES:
        w = windows.make_window(name, nfft)
        xw = z * w
        ratio = np.sum(np.abs(xw) ** 2) / np.sum(np.abs(z) ** 2)
        expect = np.sum(w**2) / nfft
        assert abs(ratio - expect) < 1e-12, name
        X = np.fft.fft(xw)
        assert abs(np.sum(np.abs(X) ** 2) / (nfft * np.sum(np.abs(xw) ** 2)) - 1.0) < 1e-12


def test_db_convention_offsets_are_constants():
    # the same spectrum presented under different conventions differs by fixed dB
    rng = np.random.default_rng(7)
    X = rng.normal(size=512) + 1j * rng.normal(size=512)
    nfft, sum_w = 1024, 510.5
    db1 = study.db20(X)
    assert np.allclose(study.db20(X / nfft), db1 - 20 * np.log10(nfft))
    assert np.allclose(study.db20(X / sum_w), db1 - 20 * np.log10(sum_w))
    assert np.allclose(10 * np.log10(np.abs(X) ** 2 + 1e-300), db1)  # 10log10|X|^2 == 20log10|X|


def test_injection_changes_only_the_intended_region():
    # the injected cosine must not move the strongest real peak
    cfg = study.pipeline.FAR_F0
    rng = np.random.default_rng(3)
    x = rng.normal(size=cfg.rng_nfft) * 50
    db_h = study.db20(study.pipeline.range_fft(x, cfg, "hann")[0])
    floor = float(np.median(db_h))
    strong = int(np.argmax(db_h))
    x2, _amp = study.inject_weak_target(x, cfg, strong + 60, floor, margin_db=8.0)
    db2 = study.db20(study.pipeline.range_fft(x2, cfg, "hann")[0])
    assert int(np.argmax(db2[: strong + 40])) == strong
    assert db2[strong + 60] - db_h[strong + 60] > 6.0  # tone actually appears


def test_exp1_smoke(tmp_path):
    summary = study.exp1_synthetic(tmp_path, nfft=256, seed=42)
    figs = sorted((tmp_path / "figures").glob("*.png"))
    csvs = sorted((tmp_path / "results").glob("*.csv"))
    assert len(figs) == 2 and all(f.stat().st_size < 300_000 for f in figs)
    assert len(csvs) == 3
    assert summary["worst_cg_abs_err"] < 1e-6
    assert summary["worst_enbw_abs_err"] < 1e-6
    assert summary["worst_amp_rel_err"] < 1e-6
    assert summary["worst_parseval_err"] < 1e-12
