"""Roundtrip pack -> rebuild tests on synthetic and repository data."""

from pathlib import Path

import numpy as np
import pytest

from conftest import DATA_A_DIR, PROJECT_DIR

from adc_exchange.ingest import IngestError
from adc_exchange.pack import pack_file
from adc_exchange.rebuild import rebuild_bytes, rebuild_file

RNG, NCHIRP = 64, 8


def make_txt(path, *, header=(), trailing=(), seed=11):
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 2**32, size=(RNG // 2) * NCHIRP, dtype=np.uint32)
    values = [int(v) for v in header] + [int(v) for v in words] + [int(v) for v in trailing]
    path.write_bytes(",".join(str(v) for v in values).encode("ascii"))
    return words


@pytest.mark.parametrize(
    "header,trailing",
    [
        ((), ()),          # exact payload only
        ((), (0,)),        # headerless + extra trailing value (A-group layout)
        ((2, RNG, NCHIRP), ()),      # triple header, no extra
        ((7, RNG, NCHIRP), (0, 0)),  # triple header + two extras
    ],
)
def test_synthetic_roundtrip_byte_exact(tmp_path, header, trailing):
    src = tmp_path / "cap_f1_3_x.txt"
    make_txt(src, header=header, trailing=trailing)
    npz = tmp_path / "cap.npz"
    out = tmp_path / "cap_rebuilt.txt"
    meta = pack_file(src, npz, rng_nfft=RNG, nchirp=NCHIRP)
    assert meta["format"]["header_fields"] == list(header)
    assert meta["trailing"]["values"] == list(trailing)
    assert meta["is_complete"] is True
    rebuild_file(npz, out)
    assert out.read_bytes() == src.read_bytes()


def test_compact_package_roundtrip_and_views(tmp_path):
    src = tmp_path / "cap_probe.txt"
    make_txt(src, header=(1, RNG, NCHIRP), trailing=[0])
    npz = tmp_path / "compact.npz"
    out = tmp_path / "compact_rebuilt.txt"
    meta = pack_file(src, npz, rng_nfft=RNG, nchirp=NCHIRP, compact=True)
    assert meta["package_mode"] == "compact"
    with np.load(npz) as z:
        assert set(z.files) == {"raw_words", "trailing_words", "metadata"}
    rebuild_file(npz, out)
    assert out.read_bytes() == src.read_bytes()
    # derived views materialize on load and pass consistency checks
    from adc_exchange.pack import check_package, load_package
    from adc_exchange.ingest import unpack_words

    pkg = load_package(npz)
    check_package(pkg)
    assert np.array_equal(pkg["samples"], unpack_words(pkg["raw_words"]))
    assert pkg["frames"].shape == (NCHIRP, RNG)


def test_partial_package_refuses_rebuild(tmp_path):
    src = tmp_path / "cap_f0_0_x.txt"
    make_txt(src, trailing=[0])
    npz = tmp_path / "partial.npz"
    meta = pack_file(src, npz, rng_nfft=RNG, nchirp=NCHIRP, limit_chirps=2)
    assert meta["is_complete"] is False
    assert meta["chirps_included"] == 2
    with pytest.raises(Exception, match="partial"):
        rebuild_file(npz, tmp_path / "never.txt")


def test_partial_package_keeps_first_chirps(tmp_path):
    src = tmp_path / "cap_f1_0_x.txt"
    words = make_txt(src, trailing=[0])
    npz = tmp_path / "partial.npz"
    pack_file(src, npz, rng_nfft=RNG, nchirp=NCHIRP, limit_chirps=3)
    with np.load(npz) as z:
        kept = z["raw_words"]
        assert np.array_equal(kept, words[: 3 * (RNG // 2)])
        assert z["frames"].shape == (3, RNG)
        assert z["trailing_words"].size == 0


def test_limit_chirps_too_large_rejected(tmp_path):
    src = tmp_path / "cap.txt"
    make_txt(src)
    with pytest.raises(IngestError, match="exceeds"):
        pack_file(src, tmp_path / "x.npz", rng_nfft=RNG, nchirp=NCHIRP, limit_chirps=99)


A_GROUP_FILES = [
    "adc_远波_20260512135827_f0_0_13_58_36_748.txt",
    "adc_近波_20260512141857_f1_0_14_19_05_965.txt",
]


@pytest.mark.parametrize("name", A_GROUP_FILES)
def test_real_capture_roundtrip(tmp_path, name):
    src = DATA_A_DIR / name
    if not src.is_file():
        pytest.skip(f"repository capture not present: {src}")
    npz = tmp_path / "real.npz"
    out = tmp_path / "real_rebuilt.txt"
    meta = pack_file(src, npz)  # profile inferred from the f0/f1 name token
    expected_chirps = 256 if "f0" in name else 128
    expected_nfft = 1024 if "f0" in name else 2048
    assert meta["frames_shape"] == [expected_chirps, expected_nfft]
    assert meta["payload_words"] == 131072
    assert meta["format"]["variant"] == "headerless"
    assert meta["trailing"]["values"] == [0]
    sha = rebuild_file(npz, out)
    assert out.read_bytes() == src.read_bytes()
    import hashlib

    assert sha == hashlib.sha256(src.read_bytes()).hexdigest()


def test_rebuild_bytes_rejects_multiline_metadata():
    meta = {
        "format": {
            "header_fields": [],
            "text_layout": {"has_newlines": True},
        }
    }
    from adc_exchange.pack import PackageError

    with pytest.raises(PackageError, match="multi-line"):
        rebuild_bytes(meta, np.zeros(4, dtype=np.uint32), np.zeros(0, dtype=np.uint32))


def test_repository_sample_package_is_consistent():
    sample = PROJECT_DIR / "results" / "sample_near_f1_rx0_first16chirps.npz"
    if not sample.is_file():
        pytest.skip("sample package not built yet")
    from adc_exchange.pack import check_package, load_package

    meta = check_package(load_package(sample))
    assert meta["profile"] == {"name": "init1", "rng_nfft": 2048, "nchirp": 128}
    assert meta["is_complete"] is False
    assert meta["chirps_included"] == 16
    with np.load(sample) as z:
        assert z["frames"].shape == (16, 2048)
    assert sample.stat().st_size < 1024 * 1024
