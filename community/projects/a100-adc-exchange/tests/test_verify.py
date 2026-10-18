"""Verify-subcommand tests: good packages pass, tampered packages fail."""

import json

import numpy as np
import pytest

from adc_exchange.cli import main
from adc_exchange.pack import pack_file

RNG, NCHIRP = 64, 8


def make_txt(path, *, header=(), trailing=(), seed=5):
    rng = np.random.default_rng(seed)
    words = rng.integers(0, 2**32, size=(RNG // 2) * NCHIRP, dtype=np.uint32)
    values = [int(v) for v in header] + [int(v) for v in words] + [int(v) for v in trailing]
    path.write_bytes(",".join(str(v) for v in values).encode("ascii"))
    return words


@pytest.fixture()
def good_pair(tmp_path):
    src = tmp_path / "cap_f0_1_x.txt"
    make_txt(src, header=(1, RNG, NCHIRP), trailing=[0])
    npz = tmp_path / "cap.npz"
    pack_file(src, npz, rng_nfft=RNG, nchirp=NCHIRP)
    return src, npz, tmp_path


def test_verify_passes_and_is_quiet_on_success(good_pair, capsys):
    src, npz, _ = good_pair
    assert main(["verify", "--npz", str(npz), "--txt", str(src)]) == 0
    out = capsys.readouterr().out
    assert "[PASS]" in out and "[FAIL]" not in out


def test_verify_without_txt_checks_consistency_only(good_pair):
    _, npz, _ = good_pair
    assert main(["verify", "--npz", str(npz)]) == 0


def _resave(npz, *, modify=None, meta_transform=None):
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    if modify:
        modify(arrays)
    if meta_transform:
        meta = json.loads(str(arrays["metadata"].item()))
        meta_transform(meta)
        arrays["metadata"] = np.array(json.dumps(meta, ensure_ascii=False))
    np.savez_compressed(npz, **arrays)


def test_tampered_samples_detected(good_pair, capsys):
    src, npz, _ = good_pair

    def flip(arrays):
        arrays["samples"][17] = np.int16(arrays["samples"][17] + 1)

    _resave(npz, modify=flip)
    assert main(["verify", "--npz", str(npz), "--txt", str(src)]) == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_tampered_raw_words_detected(good_pair, capsys):
    src, npz, _ = good_pair

    def flip(arrays):
        arrays["raw_words"][3] = np.uint32(arrays["raw_words"][3] + 7)

    _resave(npz, modify=flip)
    assert main(["verify", "--npz", str(npz), "--txt", str(src)]) == 1


def test_tampered_frames_detected(good_pair, capsys):
    src, npz, _ = good_pair

    def flip(arrays):
        arrays["frames"][2, 4] = np.int16(arrays["frames"][2, 4] - 3)

    _resave(npz, modify=flip)
    assert main(["verify", "--npz", str(npz), "--txt", str(src)]) == 1


def test_tampered_trailing_detected(good_pair, capsys):
    src, npz, _ = good_pair

    def append(arrays):
        arrays["trailing_words"] = np.array([0, 0], dtype=np.uint32)

    _resave(npz, modify=append)
    assert main(["verify", "--npz", str(npz), "--txt", str(src)]) == 1


def test_tampered_metadata_sha_detected(good_pair, capsys):
    src, npz, _ = good_pair
    _resave(npz, meta_transform=lambda m: m["source"].update(sha256="0" * 64))
    assert main(["verify", "--npz", str(npz), "--txt", str(src)]) == 1


def test_modified_source_text_detected(good_pair, capsys):
    src, npz, _ = good_pair
    original = src.read_bytes()
    # change one payload token, keeping the layout valid
    tokens = original.split(b",")
    tokens[5] = b"1234567890"
    src.write_bytes(b",".join(tokens))
    assert main(["verify", "--npz", str(npz), "--txt", str(src)]) == 1
    assert "[FAIL]" in capsys.readouterr().out


def test_missing_key_detected(good_pair, capsys):
    src, npz, tmp = good_pair
    broken = tmp / "broken.npz"
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files if k != "frames"}
    np.savez_compressed(broken, **arrays)
    rc = main(["verify", "--npz", str(broken), "--txt", str(src)])
    assert rc == 1


def test_corrupt_metadata_detected(good_pair):
    src, npz, tmp = good_pair
    broken = tmp / "broken.npz"
    with np.load(npz, allow_pickle=False) as z:
        arrays = {k: z[k].copy() for k in z.files}
    arrays["metadata"] = np.array("not json at all")
    np.savez_compressed(broken, **arrays)
    assert main(["verify", "--npz", str(broken), "--txt", str(src)]) == 1
