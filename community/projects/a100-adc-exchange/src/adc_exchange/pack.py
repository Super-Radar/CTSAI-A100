"""Pack ingested captures into a self-describing ``.npz`` exchange package.

Package layout (see README for the full specification):

==================  =====================  ====================================
key                 dtype / shape          meaning
==================  =====================  ====================================
``raw_words``       uint32 (W,)            payload words exactly as in the text
``samples``         int16  (2*W,)          interleaved hi/lo unpack view
``frames``          int16  (nchirp, nfft)  per-chirp ADC frames
``trailing_words``  uint32 (T,)            extra values after the payload
``metadata``        <str>  0-d JSON        provenance + layout description
==================  =====================  ====================================
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

from . import __version__
from .ingest import (
    IngestError,
    IngestResult,
    frames_from_words,
    ingest,
    unpack_words,
)

TOOL_NAME = "adc_exchange"
NPZ_KEYS = ("raw_words", "samples", "frames", "trailing_words", "metadata")
NPZ_COMPACT_KEYS = ("raw_words", "trailing_words", "metadata")


class PackageError(ValueError):
    """Raised when an .npz package is malformed, inconsistent or tampered."""


def _utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_relative(path: str | Path) -> str | None:
    """Return the path relative to the repository root when discoverable."""
    p = Path(path).resolve()
    for parent in [p, *p.parents]:
        if (parent / ".git").exists() and (parent / "community").is_dir():
            try:
                return str(p.relative_to(parent))
            except ValueError:
                return None
    return None


def build_metadata(
    result: IngestResult,
    *,
    included_words: np.ndarray | None = None,
    is_complete: bool,
    chirps_included: int | None,
    notes: str | None = None,
    package_mode: str = "full",
) -> dict:
    rw = result.rng_nfft // 2
    words = result.words if included_words is None else included_words
    chirps = words.size // rw
    return {
        "tool": TOOL_NAME,
        "version": __version__,
        "created_utc": _utc_now(),
        "source": {
            "path": result.source_path,
            "repo_relative_path": repo_relative(result.source_path),
            "name": result.source_name,
            "sha256": result.source_sha256,
            "size_bytes": result.source_size_bytes,
        },
        "format": {
            "variant": result.format_variant,
            "header_fields": list(result.header_fields),
            "text_layout": result.text_layout.as_dict(),
        },
        "profile": {"name": result.profile, "rng_nfft": result.rng_nfft, "nchirp": result.nchirp},
        "payload_words": int(words.size),
        "source_payload_words": int(result.words.size),
        "sample_count": int(2 * words.size),
        "frames_shape": [int(chirps), int(result.rng_nfft)],
        "trailing": {
            "count": int(result.trailing.size),
            "position": "after_payload",
            "values": [int(v) for v in result.trailing],
        },
        "rx_index": result.rx_index,
        "total_values_in_source": int(result.total_values),
        "is_complete": bool(is_complete),
        "chirps_included": int(chirps_included) if chirps_included is not None else None,
        "package_mode": package_mode,
        "notes": notes,
    }


def pack_file(
    txt_path: str | Path,
    npz_path: str | Path,
    *,
    rng_nfft: int | None = None,
    nchirp: int | None = None,
    profile: str | None = None,
    limit_chirps: int | None = None,
    compact: bool = False,
    notes: str | None = None,
) -> dict:
    """Ingest ``txt_path`` and write the exchange package to ``npz_path``.

    With ``limit_chirps`` the package keeps only the first N chirps: it stays
    fully usable for analysis but ``is_complete`` is false and
    :func:`adc_exchange.rebuild.rebuild_file` refuses to restore the source
    text from it.  With ``compact=True`` only ``raw_words``/``trailing_words``
    /``metadata`` are stored (the samples/frames views are derived on load),
    which makes archiving packages roughly 4x smaller than the source text.
    """
    result = ingest(txt_path, rng_nfft=rng_nfft, nchirp=nchirp, profile=profile)
    rw = result.rng_nfft // 2
    words = result.words
    is_complete = True
    chirps_included = None
    if limit_chirps is not None:
        if limit_chirps <= 0:
            raise IngestError("limit_chirps must be a positive integer")
        total_chirps = words.size // rw
        if limit_chirps > total_chirps:
            raise IngestError(
                f"limit_chirps={limit_chirps} exceeds the {total_chirps} chirps available"
            )
        words = np.ascontiguousarray(words[: limit_chirps * rw])
        is_complete = limit_chirps == total_chirps
        chirps_included = min(limit_chirps, total_chirps)
    mode = "compact" if compact else "full"
    metadata = build_metadata(
        result,
        included_words=words,
        is_complete=is_complete,
        chirps_included=chirps_included,
        notes=notes,
        package_mode=mode,
    )
    trailing = result.trailing if is_complete else np.empty(0, dtype=np.uint32)
    out = Path(npz_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    common = {
        "raw_words": np.ascontiguousarray(words, dtype=np.uint32),
        "trailing_words": np.ascontiguousarray(trailing, dtype=np.uint32),
        "metadata": np.array(json.dumps(metadata, ensure_ascii=False)),
    }
    if compact:
        np.savez_compressed(out, **common)
    else:
        samples = unpack_words(words)
        frames = frames_from_words(words, result.rng_nfft, metadata["frames_shape"][0])
        np.savez_compressed(out, samples=samples, frames=frames, **common)
    return metadata


def load_package(npz_path: str | Path) -> dict:
    """Load a package into ``{raw_words, samples, frames, trailing_words, metadata}``.

    Compact packages (``package_mode: "compact"``, no stored views) get their
    ``samples``/``frames`` views derived on load, so consumers see a uniform
    structure either way.
    """
    p = Path(npz_path)
    if not p.is_file():
        raise PackageError(f"package not found: {p}")
    with np.load(p, allow_pickle=False) as z:
        try:
            meta_raw = str(z["metadata"].item())
        except (KeyError, ValueError) as exc:
            raise PackageError(f"package has no readable metadata: {exc}") from exc
        for key in ("raw_words", "trailing_words"):
            if key not in z.files:
                raise PackageError(f"package is missing key {key!r}")
        pkg = {
            "raw_words": z["raw_words"],
            "trailing_words": z["trailing_words"],
            "metadata_raw": meta_raw,
        }
        for key in ("samples", "frames"):
            pkg[key] = z[key] if key in z.files else None
    try:
        pkg["metadata"] = json.loads(meta_raw)
    except ValueError as exc:
        raise PackageError(f"metadata is not valid JSON: {exc}") from exc
    return pkg


def check_package(pkg: dict) -> dict:
    """Validate internal consistency of a loaded package.

    Raises :class:`PackageError` on the first problem found; tampered arrays
    (samples, words or frames edited after packing) are always detected.
    Compact packages store no views, so their views are derived here and only
    the metadata/shape arithmetic is checked.
    """
    meta = pkg.get("metadata")
    if not isinstance(meta, dict):
        raise PackageError("metadata is missing or not an object")
    for key in ("tool", "source", "format", "profile", "payload_words"):
        if key not in meta:
            raise PackageError(f"metadata is missing field {key!r}")
    words = pkg["raw_words"]
    trailing = pkg["trailing_words"]
    if words.dtype != np.uint32:
        raise PackageError(f"raw_words dtype is {words.dtype}, expected uint32")
    if trailing.dtype != np.uint32:
        raise PackageError(f"trailing_words dtype is {trailing.dtype}, expected uint32")
    compact = meta.get("package_mode") == "compact"
    samples = pkg["samples"]
    frames = pkg["frames"]
    if compact:
        if samples is not None or frames is not None:
            raise PackageError("compact packages must not carry samples/frames arrays")
        samples = unpack_words(words)
        frames = samples.reshape(-1, int(meta["profile"]["rng_nfft"]))
        pkg["samples"], pkg["frames"] = samples, frames
    else:
        if samples is None or frames is None:
            raise PackageError(
                "package is missing keys: "
                f"{[k for k in ('samples', 'frames') if pkg.get(k) is None]}"
            )
        if samples.dtype != np.int16:
            raise PackageError(f"samples dtype is {samples.dtype}, expected int16")
        if frames.dtype != np.int16:
            raise PackageError(f"frames dtype is {frames.dtype}, expected int16")
    n_words = int(words.size)
    if n_words != int(meta["payload_words"]):
        raise PackageError(
            f"raw_words holds {n_words} words, metadata claims {meta['payload_words']}"
        )
    rng = int(meta["profile"]["rng_nfft"])
    chirps = int(meta["frames_shape"][0])
    if rng % 2:
        raise PackageError("metadata rng_nfft is odd")
    if n_words != chirps * (rng // 2):
        raise PackageError(
            f"raw_words ({n_words}) does not fill frames_shape "
            f"{meta['frames_shape']} with {rng // 2} words per chirp"
        )
    if samples.size != 2 * n_words:
        raise PackageError(
            f"samples holds {samples.size} values, expected {2 * n_words}"
        )
    if tuple(frames.shape) != tuple(meta["frames_shape"]):
        raise PackageError(
            f"frames shape {frames.shape} does not match metadata {meta['frames_shape']}"
        )
    recomputed = unpack_words(words)
    if not np.array_equal(samples, recomputed):
        raise PackageError("samples does not equal the unpack view of raw_words")
    if not np.array_equal(frames.reshape(-1), samples):
        raise PackageError("frames does not equal samples reshaped to frames_shape")
    if meta.get("is_complete", False):
        if int(trailing.size) != int(meta["trailing"]["count"]):
            raise PackageError(
                f"trailing_words holds {trailing.size} values, metadata claims "
                f"{meta['trailing']['count']}"
            )
    elif trailing.size:
        raise PackageError("incomplete packages must not carry trailing_words")
    return meta
