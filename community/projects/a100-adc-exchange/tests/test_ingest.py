"""Ingest unit tests on synthetic captures (no repository data needed)."""

import numpy as np
import pytest

from adc_exchange.ingest import (
    IngestError,
    detect_layout,
    ingest,
    parse_values,
    profile_from_name,
    rx_index_from_name,
    unpack_words,
)

RNG, NCHIRP = 64, 8  # payload = 32 * 8 = 256 words


def make_txt(path, *, header=(), trailing=(), seed=7):
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 2**32, size=(RNG // 2) * NCHIRP, dtype=np.uint32)
    values = [int(v) for v in header] + [int(v) for v in words] + [int(v) for v in trailing]
    path.write_text(",".join(str(v) for v in values), encoding="ascii")
    return words


def matlab_reference_unpack(words):
    """Straight transcription of adc_mem_2_real_fixed.m."""
    out = []
    for w in words:
        h = int(np.floor(w / 2**16))
        low = int(np.mod(w, 2**16))
        out.append(h if h < 2**15 else h - 2**16)
        out.append(low if low < 2**15 else low - 2**16)
    return np.array(out, dtype=np.int16)


class TestParseValues:
    def test_plain(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_bytes(b"1,2,4294967295")
        assert parse_values(f.read_bytes()).tolist() == [1, 2, 2**32 - 1]

    def test_empty_file_rejected(self, tmp_path):
        f = tmp_path / "a.txt"
        f.write_bytes(b"")
        with pytest.raises(IngestError, match="empty"):
            parse_values(f.read_bytes())

    def test_newlines_rejected(self, tmp_path):
        with pytest.raises(IngestError, match="newline"):
            parse_values(b"1,2\n3")

    def test_negative_rejected(self):
        with pytest.raises(IngestError, match="unsigned"):
            parse_values(b"1,-2,3")

    def test_above_uint32_rejected(self):
        with pytest.raises(IngestError, match="uint32"):
            parse_values(b"4294967296")

    def test_empty_token_rejected(self):
        with pytest.raises(IngestError, match="token"):
            parse_values(b"1,,3")


class TestDetectLayout:
    def test_exact_payload(self):
        v = np.arange((RNG // 2) * NCHIRP, dtype=np.uint64)
        assert detect_layout(v, RNG, NCHIRP) == ((), 0, "headerless")

    def test_payload_plus_one_trailing(self):
        v = np.arange((RNG // 2) * NCHIRP + 1, dtype=np.uint64)
        assert detect_layout(v, RNG, NCHIRP) == ((), 1, "headerless")

    def test_triple_header_plus_trailing(self):
        v = np.array([3, RNG, NCHIRP] + list(range((RNG // 2) * NCHIRP + 1)), dtype=np.uint64)
        header, trailing, variant = detect_layout(v, RNG, NCHIRP)
        assert header == (3, RNG, NCHIRP) and trailing == 1 and variant == "triple_header"

    def test_triple_header_without_trailing(self):
        v = np.array([0, RNG, NCHIRP] + list(range((RNG // 2) * NCHIRP)), dtype=np.uint64)
        assert detect_layout(v, RNG, NCHIRP)[2] == "triple_header"

    def test_fake_header_rejected(self):
        v = np.array([3, 999, NCHIRP] + list(range((RNG // 2) * NCHIRP)), dtype=np.uint64)
        with pytest.raises(IngestError, match="header signature"):
            detect_layout(v, RNG, NCHIRP)

    def test_below_payload_rejected(self):
        v = np.arange(7, dtype=np.uint64)
        with pytest.raises(IngestError, match="below the expected payload"):
            detect_layout(v, RNG, NCHIRP)


class TestNameHints:
    def test_f_tokens(self):
        assert profile_from_name("adc_远波_20260512135827_f0_2_13.txt") == "init0"
        assert profile_from_name("adc_test_20260512141857_f1_0_14.txt") == "init1"
        assert rx_index_from_name("adc_远波_20260512135827_f0_2_13.txt") == 2

    def test_pf_tokens(self):
        assert profile_from_name("adc_test_20260527115050_Pf0_Rx2.txt") == "init0"
        assert rx_index_from_name("adc_test_20260527115050_Pf0_Rx2.txt") == 2

    def test_no_hint(self):
        assert profile_from_name("random.txt") is None
        assert rx_index_from_name("random.txt") is None


class TestUnpackOfficialSemantics:
    def test_matches_matlab_reference(self):
        rng = np.random.default_rng(3)
        words = rng.integers(0, 2**32, size=5000, dtype=np.uint32)
        np.testing.assert_array_equal(unpack_words(words), matlab_reference_unpack(words))

    def test_extremes(self):
        words = np.array(
            [0, 2**32 - 1, 0x80000000, 0x00007FFF, 0xFFFF8000, 0x12345678],
            dtype=np.uint32,
        )
        expected = np.array(
            [
                0, 0,           # 0x00000000
                -1, -1,         # 0xFFFFFFFF
                -32768, 0,      # 0x80000000
                0, 32767,       # 0x00007FFF
                -1, -32768,     # 0xFFFF8000
                0x1234, 0x5678,  # 0x12345678 (both halves positive)
            ],
            dtype=np.int16,
        )
        np.testing.assert_array_equal(unpack_words(words), expected)


class TestIngestSynthetic:
    def test_headerless_with_trailing(self, tmp_path):
        f = tmp_path / "cap_f0_0_x.txt"
        words = make_txt(f, trailing=[0])
        res = ingest(f, rng_nfft=RNG, nchirp=NCHIRP)
        assert res.format_variant == "headerless"
        assert res.header_fields == ()
        assert np.array_equal(res.words, words)
        assert res.trailing.tolist() == [0]
        assert res.total_values == words.size + 1
        assert res.rx_index == 0
        assert res.source_sha256 and res.source_size_bytes == f.stat().st_size

    def test_triple_header_with_trailing(self, tmp_path):
        f = tmp_path / "cap.txt"
        words = make_txt(f, header=[5, RNG, NCHIRP], trailing=[0])
        res = ingest(f, rng_nfft=RNG, nchirp=NCHIRP)
        assert res.format_variant == "triple_header"
        assert res.header_fields == (5, RNG, NCHIRP)
        assert np.array_equal(res.words, words)
        assert res.trailing.tolist() == [0]
        assert res.rx_index == 5

    def test_custom_params_keep_label(self, tmp_path):
        f = tmp_path / "cap.txt"
        make_txt(f)
        res = ingest(f, rng_nfft=64, nchirp=8)
        assert res.rng_nfft == 64 and res.nchirp == 8

    def test_unknown_profile_rejected(self, tmp_path):
        f = tmp_path / "cap.txt"
        make_txt(f)
        with pytest.raises(IngestError, match="unknown profile"):
            ingest(f, profile="init9")

    def test_single_dim_param_rejected(self, tmp_path):
        f = tmp_path / "cap.txt"
        make_txt(f)
        with pytest.raises(IngestError, match="both"):
            ingest(f, rng_nfft=64)
