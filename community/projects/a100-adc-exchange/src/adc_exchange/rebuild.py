"""Rebuild the original capture text from an exchange package, byte-for-byte.

The rebuild concatenates ``header fields + payload words + trailing values``
with the same single-line comma layout the official captures use (verified:
no newline, no carriage return, no trailing newline), so the output is
byte-identical to the source file.  Incomplete (chirp-limited) packages are
refused.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from .pack import PackageError, check_package, load_package


def rebuild_bytes(metadata: dict, words: np.ndarray, trailing: np.ndarray) -> bytes:
    """Render the exact original text bytes from package parts."""
    header = metadata["format"]["header_fields"]
    layout = metadata["format"].get("text_layout", {})
    if layout.get("has_newlines") or layout.get("has_carriage_returns"):
        raise PackageError("cannot rebuild multi-line captures (unsupported layout)")
    values = [int(v) for v in header]
    values.extend(int(v) for v in words)
    values.extend(int(v) for v in trailing)
    text = ",".join(str(v) for v in values)
    if layout.get("trailing_newline"):
        text += "\n"
    return text.encode("ascii")


def rebuild_file(npz_path: str | Path, out_txt_path: str | Path) -> str:
    """Rebuild the source text from ``npz_path``; returns the written sha256."""
    pkg = load_package(npz_path)
    meta = check_package(pkg)
    if not meta.get("is_complete", False):
        raise PackageError(
            "package is partial (chirps_included="
            f"{meta.get('chirps_included')}); the full source text cannot be "
            "restored from it"
        )
    data = rebuild_bytes(meta, pkg["raw_words"], pkg["trailing_words"])
    out = Path(out_txt_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(data)
    return hashlib.sha256(data).hexdigest()
