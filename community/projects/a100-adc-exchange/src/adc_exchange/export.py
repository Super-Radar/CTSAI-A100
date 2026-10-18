"""Export an exchange package to a long-table CSV and a MATLAB-readable .mat.

The CSV carries one row per 32-bit word:

``word_index, chirp_index, word_in_chirp, adc_word, sample_hi, sample_lo``

A full export of a 131072-word capture produces roughly 3.4 MB of CSV, so
``max_rows`` can truncate the table to keep artifacts small.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from .ingest import unpack_words
from .pack import PackageError, check_package, load_package

CSV_COLUMNS = [
    "word_index",
    "chirp_index",
    "word_in_chirp",
    "adc_word",
    "sample_hi",
    "sample_lo",
]


def export_csv(
    npz_path: str | Path,
    csv_path: str | Path,
    max_rows: int | None = None,
) -> dict:
    """Write the long-table CSV; returns ``{"rows": n, "truncated": bool}``."""
    pkg = load_package(npz_path)
    meta = check_package(pkg)
    words = pkg["raw_words"]
    samples = pkg["samples"]
    rng = int(meta["profile"]["rng_nfft"])
    words_per_chirp = rng // 2
    if max_rows is not None and max_rows < 0:
        raise ValueError("max_rows must be non-negative")
    limit = words.size if max_rows is None else min(int(max_rows), words.size)
    out = Path(csv_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(CSV_COLUMNS)
        for i in range(limit):
            writer.writerow(
                [
                    i,
                    i // words_per_chirp,
                    i % words_per_chirp,
                    int(words[i]),
                    int(samples[2 * i]),
                    int(samples[2 * i + 1]),
                ]
            )
    return {"rows": int(limit), "truncated": bool(limit < words.size)}


def export_mat(npz_path: str | Path, mat_path: str | Path) -> Path:
    """Write a MATLAB v5 .mat (via ``scipy.io.savemat``) with the same content."""
    try:
        from scipy.io import savemat
    except ImportError as exc:  # pragma: no cover - scipy is optional
        raise PackageError("scipy is required for .mat export: pip install scipy") from exc
    pkg = load_package(npz_path)
    meta = check_package(pkg)
    out = Path(mat_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    savemat(
        out,
        {
            "raw_words": pkg["raw_words"],
            "samples": pkg["samples"],
            "frames": pkg["frames"],
            "trailing_words": pkg["trailing_words"],
            "metadata_json": json.dumps(meta, ensure_ascii=False),
            "source_sha256": str(meta["source"]["sha256"]),
            "rng_nfft": float(meta["profile"]["rng_nfft"]),
            "nchirp": float(meta["profile"]["nchirp"]),
            "is_complete": float(bool(meta.get("is_complete", False))),
        },
        do_compression=True,
        oned_as="column",
    )
    return out
