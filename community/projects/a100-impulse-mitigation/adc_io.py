"""Strict batch reader for the headed captures used in these experiments."""

import csv
import hashlib
import re
from pathlib import Path

import numpy as np


def sources(project):
    root = Path(project).parent / "a100-indoor-cart-mti"
    rows = list(csv.DictReader((root / "data/manifest.csv").open(encoding="utf-8")))
    return [
        (root / row["relative_path"], row)
        for row in rows
        if row["data_type"] == "adc_txt"
    ]


def load(path, expected_hash=None):
    raw = Path(path).read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    if expected_hash is not None and sha != expected_hash:
        raise ValueError("Source hash mismatch")
    text = raw.decode("ascii").strip()
    text = text.removesuffix(",")
    fields = text.split(",")
    if any(not re.fullmatch(r"[0-9]{1,10}", word.strip()) for word in fields):
        raise ValueError("Invalid uint32 token")
    values = np.array([int(word) for word in fields], dtype=np.uint64)
    if len(values) < 3 or np.any(values > 2**32 - 1):
        raise ValueError("Invalid header/word range")
    rx, n, c = map(int, values[:3])
    if rx > 3 or n < 2 or n % 2 or c < 1:
        raise ValueError("Invalid headed dimensions")
    expected = n * c // 2
    payload = values[3:]
    if len(payload) == expected + 1 and payload[-1] == 0:
        payload = payload[:-1]
    if len(payload) != expected:
        raise ValueError("Payload size mismatch")
    signed = payload.astype(">u4").view(">i2").astype(float).reshape(c, n)
    return signed, {
        "rx": rx,
        "samples": n,
        "chirps": c,
        "sha256": sha,
        "source": Path(path).name,
    }
