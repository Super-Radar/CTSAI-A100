"""CLI export/rebuild end-to-end tests plus one real B-group capture."""

import csv

import numpy as np
import pytest

from conftest import DATA_B_DIR

from adc_exchange.cli import main
from adc_exchange.pack import pack_file

RNG, NCHIRP = 64, 8


def make_txt(path, *, header=(), trailing=(), seed=9):
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 2**32, size=(RNG // 2) * NCHIRP, dtype=np.uint32)
    values = [int(v) for v in header] + [int(v) for v in words] + [int(v) for v in trailing]
    path.write_bytes(",".join(str(v) for v in values).encode("ascii"))
    return words


@pytest.fixture()
def packed(tmp_path):
    # NB: name deliberately free of f0/f1 markers so CLI profile inference is
    # not triggered; the fixture passes explicit dimensions.
    src = tmp_path / "capture_probe.txt"
    words = make_txt(src, trailing=[0])
    npz = tmp_path / "cap.npz"
    pack_file(src, npz, rng_nfft=RNG, nchirp=NCHIRP)
    return src, npz, words


def test_cli_pack_rebuild_roundtrip(packed, tmp_path):
    src, _, _ = packed
    npz2 = tmp_path / "again.npz"
    out = tmp_path / "again.txt"
    rc = main(
        ["pack", str(src), "-o", str(npz2), "--rng-nfft", str(RNG), "--nchirp", str(NCHIRP)]
    )
    assert rc == 0
    assert main(["rebuild", str(npz2), "-o", str(out)]) == 0
    assert out.read_bytes() == src.read_bytes()


def test_csv_export_full_and_truncated(packed, tmp_path):
    _, npz, words = packed
    full = tmp_path / "full.csv"
    assert main(["export", str(npz), "-o", str(full)]) == 0
    with full.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == words.size
    assert list(rows[0].keys()) == [
        "word_index", "chirp_index", "word_in_chirp", "adc_word", "sample_hi", "sample_lo",
    ]
    assert int(rows[0]["adc_word"]) == int(words[0])
    assert int(rows[0]["chirp_index"]) == 0
    # word 33 (words_per_chirp=32) belongs to chirp 1
    assert int(rows[33]["chirp_index"]) == 1 and int(rows[33]["word_in_chirp"]) == 1
    cut = tmp_path / "cut.csv"
    assert main(["export", str(npz), "-o", str(cut), "--max-rows", "10"]) == 0
    with cut.open() as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 10


def test_mat_export_roundtrip_via_scipy(packed, tmp_path):
    scipy_io = pytest.importorskip("scipy.io")
    _, npz, _ = packed
    mat = tmp_path / "out.mat"
    rc = main(["export", str(npz), "-o", str(tmp_path / "x.csv"), "--mat", str(mat)])
    assert rc == 0
    m = scipy_io.loadmat(mat)
    with np.load(npz) as z:
        np.testing.assert_array_equal(m["raw_words"].ravel(), z["raw_words"])
        np.testing.assert_array_equal(m["frames"], z["frames"])
        assert len(str(m["source_sha256"][0])) == 64
        import json as _json

        meta = _json.loads(str(m["metadata_json"][0]))
        assert meta["profile"]["rng_nfft"] == RNG


def test_real_b_group_triple_header_roundtrip(tmp_path):
    src = DATA_B_DIR / "adc_test_20260527115033_Pf0_Rx0.txt"
    if not src.is_file():
        pytest.skip(f"repository capture not present: {src}")
    npz = tmp_path / "pf0.npz"
    out = tmp_path / "pf0_rebuilt.txt"
    meta = pack_file(src, npz)
    assert meta["format"]["variant"] == "triple_header"
    assert meta["format"]["header_fields"] == [0, 1024, 256]
    assert meta["trailing"]["values"] == [0]
    assert meta["payload_words"] == 131072
    assert meta["rx_index"] == 0
    assert main(["rebuild", str(npz), "-o", str(out)]) == 0
    assert out.read_bytes() == src.read_bytes()
    assert main(["verify", "--npz", str(npz), "--txt", str(src)]) == 0
