"""Pipeline tests: uint32 unpacking, MATLAB-compatible reshape, truncation policy,
configuration-derived sample counts, tone peak/amplitude on the A100 geometry."""

import numpy as np

from window_study import pipeline


def test_unpack_word_order_and_sign():
    # earlier sample = high 16 bits, later sample = low 16 bits, two's complement
    words = np.array(
        [[0x00010002, 0xFFFF8000, 0x7FFF0000, 0x0000FFFF, 0x00000000]], dtype=np.uint32
    )
    out = pipeline.unpack_words(words)
    assert out.shape == (1, 10)
    assert out[0, 0] == 1.0 and out[0, 1] == 2.0
    assert out[0, 2] == -1.0 and out[0, 3] == -32768.0
    assert out[0, 4] == 32767.0 and out[0, 5] == 0.0
    assert out[0, 6] == 0.0 and out[0, 7] == -1.0
    assert out[0, 8] == 0.0 and out[0, 9] == 0.0


def test_reshape_matches_matlab_column_major():
    rng = np.random.default_rng(0)
    cfg = pipeline.FAR_F0
    payload = rng.integers(0, 2**32, size=cfg.expected_words, dtype=np.uint64).astype(np.uint32)
    m = pipeline.reshape_words_column_major(payload, cfg)
    assert m.shape == (cfg.nchirp, cfg.words_per_chirp)
    # MATLAB reshape(x,[512,256]) then transpose: chirp c is the contiguous block c*512..
    assert np.array_equal(m[0], payload[: cfg.words_per_chirp])
    assert np.array_equal(m[7], payload[7 * cfg.words_per_chirp : 8 * cfg.words_per_chirp])
    assert np.array_equal(m, np.reshape(payload, (cfg.nchirp, cfg.words_per_chirp)))


def test_load_raw_words_truncates_and_records(tmp_path):
    f = tmp_path / "mini.txt"
    f.write_text("1,2,3,4,5,6,7,8,9", encoding="utf-8")
    payload, info = pipeline.load_raw_words(f, expected_len=8)
    assert payload.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert info["dropped_tail"] == [9]
    # whitespace/newline separated input behaves the same
    f2 = tmp_path / "mini2.txt"
    f2.write_text("1 2\n3,4 5\n6 7 8 9", encoding="utf-8")
    payload2, info2 = pipeline.load_raw_words(f2, expected_len=8)
    assert payload2.tolist() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert info2["dropped_tail"] == [9]


def test_adc_sample_num():
    assert pipeline.FAR_F0.adc_sample_num == 1021  # round((42.8-2)*25)+1
    assert pipeline.FAR_F0.zero_fill == 3
    assert pipeline.NEAR_F1.adc_sample_num == 2048  # 2561 clamped to rng_nfft
    assert pipeline.NEAR_F1.zero_fill == 0
    # c/(2B) with exact c is 0.4997 m; the usual "0.5 m" uses c = 3e8
    assert abs(pipeline.FAR_F0.nominal_range_res_m - 0.5) < 1e-3


def test_range_fft_shape_and_tone_peak():
    cfg = pipeline.FAR_F0
    n = np.arange(cfg.rng_nfft)
    k = 37
    a = 1000.0
    x = a * np.cos(2.0 * np.pi * k * n / cfg.rng_nfft)
    X, w = pipeline.range_fft(x, cfg, "hann")
    assert X.shape == (cfg.rng_nfft // 2,)
    assert len(w) == cfg.adc_sample_num  # window length = valid samples, not nfft
    assert int(np.argmax(np.abs(X))) == k

    # complex exponential: exact coherent gain A*sum(w) at the bin
    z = a * np.exp(2j * np.pi * k * n / cfg.rng_nfft)
    Z, w2 = pipeline.range_fft(z, cfg, "hann")
    assert abs(abs(Z[k]) / w2.sum() - a) < 1e-9

    # real cosine: estimator 2|X[k]|/sum(w); residual = negative-image leakage
    a_est = 2.0 * abs(X[k]) / w.sum()
    assert abs(a_est - a) / a < 2e-6


def test_official_chain_divisor_exposed():
    assert pipeline.OFFICIAL_RNG_FFT_SCALAR == 255.0  # .rng_fft_scalar = 0xff
