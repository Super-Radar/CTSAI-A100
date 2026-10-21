"""解析器测试：双布局识别、异常 token 报告、int16 解包语义。"""

import hashlib

import numpy as np
import pytest

from adc_audit.loader import load_adc_file, unpack_int16
from conftest import make_words, write_adc


def test_headerless_detection(tmp_path):
    words = make_words(64, seed=1)
    p = write_adc(tmp_path / "adc_test_20260512141857_f1_0_1_2_3.txt", words)
    lf = load_adc_file(p)
    assert lf.header is None
    assert lf.format_name == "headerless"
    assert lf.payload_count == 64
    assert lf.token_count == 64
    assert lf.bad_tokens == []
    assert lf.sha256 == hashlib.sha256(p.read_bytes()).hexdigest()


def test_header3_detection(tmp_path):
    words = make_words(64, seed=2)
    p = write_adc(tmp_path / "adc_test_20260527115033_Pf0_Rx0.txt", words, header=(0, 1024, 256))
    lf = load_adc_file(p)
    assert lf.header == (0, 1024, 256)
    assert lf.format_name == "header3"
    assert lf.payload_count == 64
    assert lf.token_count == 67


def test_header_requires_three_plausible_fields(tmp_path):
    # 仅首值小不足以判定为头：后续两值不是 FFT/chirp 数
    words = np.concatenate([[0], make_words(64, seed=3)])
    p = write_adc(tmp_path / "adc_test_x_f1_0.txt", words)
    lf = load_adc_file(p)
    assert lf.header is None
    assert lf.payload_count == 65


def test_bad_tokens_non_numeric(tmp_path):
    vals = [str(w) for w in make_words(32, seed=4)]
    vals[10] = "NaN"
    vals[20] = "12.5"
    p = tmp_path / "adc_test_x_f1_0.txt"
    p.write_text(",".join(vals), encoding="utf-8")
    lf = load_adc_file(p)
    assert len(lf.bad_tokens) == 2
    assert lf.bad_tokens[0].index == 10 and lf.bad_tokens[0].reason == "non-numeric"
    assert lf.bad_tokens[1].index == 20 and lf.bad_tokens[1].reason == "non-numeric"
    assert lf.payload_count == 30


def test_bad_tokens_out_of_range(tmp_path):
    vals = [str(w) for w in make_words(32, seed=5)]
    vals[3] = "4294967296"  # 2^32
    vals[7] = "-7"
    p = tmp_path / "adc_test_x_f1_0.txt"
    p.write_text(",".join(vals), encoding="utf-8")
    lf = load_adc_file(p)
    assert [b.index for b in lf.bad_tokens] == [3, 7]
    assert lf.bad_tokens[0].reason == "out-of-range(>2^32-1)"
    assert lf.bad_tokens[1].reason == "out-of-range(<0)"
    assert lf.payload_count == 30


def test_unpack_int16_semantics():
    # 0x00010002 -> hi=+1, lo=+2；0xFFFF8000 -> hi=-1, lo=-32768
    words = np.array([0x00010002, 0xFFFF8000, 0x7FFFFF87], dtype=np.uint32)
    samples = unpack_int16(words, rng_nfft=6, nchirp=1)
    assert samples.shape == (1, 6)
    assert samples[0].tolist() == [1, 2, -1, -32768, 32767, -121]


def test_unpack_chirp_major_order():
    # 第 c 个 chirp 对应载荷中第 c*wpc..(c+1)*wpc 个连续字（MATLAB 列主序等价）
    wpc = 2
    words = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
    samples = unpack_int16(words, rng_nfft=2 * wpc, nchirp=4)
    assert samples[0].tolist() == [0, 1, 0, 2]  # word 1 -> hi 0, lo 1; word 2 -> hi 0, lo 2
    assert samples[3].tolist() == [0, 7, 0, 8]


def test_unpack_rejects_misaligned():
    with pytest.raises(ValueError):
        unpack_int16(np.zeros(5, dtype=np.uint32), rng_nfft=8, nchirp=1)
