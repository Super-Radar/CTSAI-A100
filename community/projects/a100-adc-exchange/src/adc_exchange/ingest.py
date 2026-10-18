"""Ingest CTSAI-A100 ADC text captures into verified in-memory structures.

Two on-disk text variants exist in the official repository data:

* ``headerless``
  ``ADC数据采集/示例adc数据和结果/adc_*.txt`` -- one single line of
  comma-separated ``uint32`` words holding ``rng_nfft/2 * nchirp`` payload
  words followed by exactly one extra trailing value (observed to be ``0``).
* ``triple_header``
  ``ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100/data/adc_test_*_Pf*_Rx*.txt``
  -- the same payload/trailing layout preceded by three header fields
  ``<rx_index>,<rng_nfft>,<nchirp>`` (measured example: ``0,1024,256,``).

The loader keeps header fields, payload words and extra trailing values apart
so that :mod:`adc_exchange.rebuild` can restore the original file
byte-for-byte.  Sample unpacking follows the official MATLAB semantics of
``init_function/adc_mem_2_real_fixed.m``: each ``uint32`` word carries one
signed int16 sample in its high 16 bits followed by another in its low 16
bits.

This module performs offline analysis of repository text captures only; it
does not read the live USB ADC stream.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

UINT32_MAX = 0xFFFFFFFF

#: Sensor profiles from
#: ``cfg/CTASI-A100配置/sensor_config_init{0,1}.hxx`` (rng_nfft / nchirp).
PROFILE_PARAMS: dict[str, dict[str, int]] = {
    "init0": {"rng_nfft": 1024, "nchirp": 256},  # far wave, f0
    "init1": {"rng_nfft": 2048, "nchirp": 128},  # near wave, f1
}

PROFILES_DIR_REL = "ADC数据采集/matlab_signal_processing_platform_231023_for_txt_A100"

_BYTES_DIGITS = frozenset(b"0123456789")

_F_TOKEN_RE = re.compile(r"_f([01])_(\d+)_")
_PF_TOKEN_RE = re.compile(r"pf([01])_rx(\d+)", re.IGNORECASE)


class IngestError(ValueError):
    """Raised when a text capture does not match any supported layout."""


@dataclass
class TextLayout:
    """Facts about the on-disk text layout that rebuild relies on."""

    separator: str = ","
    has_newlines: bool = False
    has_carriage_returns: bool = False
    trailing_newline: bool = False

    def as_dict(self) -> dict:
        return {
            "separator": self.separator,
            "has_newlines": self.has_newlines,
            "has_carriage_returns": self.has_carriage_returns,
            "trailing_newline": self.trailing_newline,
        }


@dataclass
class IngestResult:
    """Everything needed to describe, unpack and losslessly rebuild a capture."""

    words: np.ndarray  # uint32 payload words, word-major stream
    trailing: np.ndarray  # uint32 extra values recorded after the payload
    header_fields: tuple[int, ...]  # header ints (empty for headerless files)
    format_variant: str  # "headerless" | "triple_header"
    total_values: int  # values found in the source text (all parts included)
    rng_nfft: int
    nchirp: int
    profile: str  # "init0" | "init1" | "custom"
    rx_index: int | None  # receiver channel when identifiable
    source_path: str
    source_name: str
    source_sha256: str
    source_size_bytes: int
    text_layout: TextLayout = field(default_factory=TextLayout)

    @property
    def payload_words(self) -> int:
        return int(self.words.size)


def read_source(path: str | Path) -> tuple[bytes, str, int]:
    """Read the raw bytes of a capture plus its sha256 and size."""
    p = Path(path)
    raw = p.read_bytes()
    return raw, hashlib.sha256(raw).hexdigest(), p.stat().st_size


def parse_values(raw: bytes) -> np.ndarray:
    """Parse a single-line comma-separated uint32 text into a uint64 array.

    The official captures contain no newline characters and no negative
    values; anything else is rejected loudly instead of being silently
    reinterpreted.
    """
    if not raw:
        raise IngestError("file is empty")
    if b"\n" in raw or b"\r" in raw:
        raise IngestError(
            "file contains newline/carriage-return characters; only the "
            "single-line comma-separated layout of the official captures is "
            "supported"
        )
    tokens = raw.split(b",")
    values = np.empty(len(tokens), dtype=np.uint64)
    for i, token in enumerate(tokens):
        if not token or not _BYTES_DIGITS.issuperset(token):
            raise IngestError(f"token #{i} is not a plain unsigned integer: {token!r}")
        v = int(token)
        if v > UINT32_MAX:
            raise IngestError(f"token #{i} exceeds uint32 range: {v}")
        values[i] = v
    return values


def _looks_like_triple_header(candidate: np.ndarray, rng_nfft: int, nchirp: int) -> bool:
    if candidate.size < 3:
        return False
    rx, fft_len, chirps = (int(x) for x in candidate[:3])
    return rx < 64 and fft_len == rng_nfft and chirps == nchirp


def detect_layout(
    values: np.ndarray, rng_nfft: int, nchirp: int
) -> tuple[tuple[int, ...], int, str]:
    """Split a value stream into (header fields, trailing count, variant).

    ``rng_nfft``/``nchirp`` define the expected payload length
    ``rng_nfft // 2 * nchirp`` exactly like ``load_adc_data.m`` does.  Every
    value beyond the payload (and beyond the 3 header fields when a
    ``<rx_index>,<rng_nfft>,<nchirp>`` signature is present) is preserved as a
    trailing value; the official captures carry exactly one trailing ``0``.
    """
    payload = (rng_nfft // 2) * nchirp
    n = int(values.size)
    if n < payload:
        raise IngestError(
            f"unsupported value count {n} for profile rng_nfft={rng_nfft}, "
            f"nchirp={nchirp}: below the expected payload of {payload} words"
        )
    if n >= payload + 3 and _looks_like_triple_header(values, rng_nfft, nchirp):
        header3 = tuple(int(x) for x in values[:3])
        return header3, n - payload - 3, "triple_header"
    if n >= payload + 3:
        raise IngestError(
            f"value count {n} exceeds payload+3 but the leading values "
            f"{tuple(int(x) for x in values[:3])} do not look like the "
            f"<rx_index>,{rng_nfft},{nchirp} header signature"
        )
    return (), n - payload, "headerless"


def profile_from_name(name: str) -> str | None:
    """Best-effort profile hint from a capture file name (f0 -> init0...)."""
    lowered = name.lower()
    m = _PF_TOKEN_RE.search(lowered) or _F_TOKEN_RE.search(lowered)
    if m:
        return f"init{m.group(1)}"
    return None


def rx_index_from_name(name: str) -> int | None:
    m = _PF_TOKEN_RE.search(name.lower()) or _F_TOKEN_RE.search(name.lower())
    return int(m.group(2)) if m else None


def _resolve_profile(
    path: Path,
    rng_nfft: int | None,
    nchirp: int | None,
    profile: str | None,
) -> tuple[int, int, str]:
    if rng_nfft is not None and nchirp is not None:
        for label, params in PROFILE_PARAMS.items():
            if params["rng_nfft"] == rng_nfft and params["nchirp"] == nchirp:
                return rng_nfft, nchirp, label
        if rng_nfft % 2:
            raise IngestError("rng_nfft must be even (two int16 samples per word)")
        return rng_nfft, nchirp, "custom"
    if rng_nfft is not None or nchirp is not None:
        raise IngestError("pass both rng_nfft and nchirp, or neither")
    if profile is not None:
        if profile not in PROFILE_PARAMS:
            raise IngestError(f"unknown profile {profile!r}; expected one of {sorted(PROFILE_PARAMS)}")
        params = PROFILE_PARAMS[profile]
        return params["rng_nfft"], params["nchirp"], profile
    hint = profile_from_name(path.name)
    if hint is not None:
        params = PROFILE_PARAMS[hint]
        return params["rng_nfft"], params["nchirp"], hint
    raise IngestError(
        "cannot infer the sensor profile: pass --rng-nfft/--nchirp or --profile "
        f"(init0/init1), or use a file name containing an f0/f1 marker"
    )


def ingest(
    path: str | Path,
    *,
    rng_nfft: int | None = None,
    nchirp: int | None = None,
    profile: str | None = None,
) -> IngestResult:
    """Load a capture and separate header fields / payload / trailing values."""
    p = Path(path)
    if not p.is_file():
        raise IngestError(f"file not found: {p}")
    resolved_fft, resolved_chirps, label = _resolve_profile(p, rng_nfft, nchirp, profile)
    raw, sha, size = read_source(p)
    values = parse_values(raw)
    header, trailing_count, variant = detect_layout(values, resolved_fft, resolved_chirps)
    payload = (resolved_fft // 2) * resolved_chirps
    header_len = len(header)
    words = np.ascontiguousarray(values[header_len : header_len + payload], dtype=np.uint32)
    trailing = np.ascontiguousarray(
        values[header_len + payload : header_len + payload + trailing_count], dtype=np.uint32
    )
    rx_index = int(header[0]) if header else rx_index_from_name(p.name)
    return IngestResult(
        words=words,
        trailing=trailing,
        header_fields=header,
        format_variant=variant,
        total_values=int(values.size),
        rng_nfft=resolved_fft,
        nchirp=resolved_chirps,
        profile=label,
        rx_index=rx_index,
        source_path=str(p.resolve()),
        source_name=p.name,
        source_sha256=sha,
        source_size_bytes=size,
        text_layout=TextLayout(),
    )


def unpack_words(words: np.ndarray) -> np.ndarray:
    """Unpack uint32 words into an interleaved int16 sample stream.

    Mirrors ``adc_mem_2_real_fixed.m``: word ``w`` yields the signed sample
    ``floor(w / 2^16)`` (two's complement) followed by ``mod(w, 2^16)``.
    """
    w = np.ascontiguousarray(words, dtype=np.uint32)
    out = np.empty(2 * w.size, dtype=np.int16)
    out[0::2] = (w >> np.uint32(16)).astype(np.int16)
    out[1::2] = (w & np.uint32(0xFFFF)).astype(np.int16)
    return out


def frames_from_words(words: np.ndarray, rng_nfft: int, nchirp: int) -> np.ndarray:
    """Shape the sample stream into ``(nchirp, rng_nfft)`` int16 frames.

    Equivalent to the MATLAB platform path
    ``reshape(col, [rng_nfft/2, nchirp])'`` followed by per-word interleaved
    unpacking: ``frames.ravel()`` equals :func:`unpack_words` output.
    """
    samples = unpack_words(words)
    expected = nchirp * rng_nfft
    if samples.size != expected:
        raise IngestError(
            f"samples ({samples.size}) do not fill {nchirp}x{rng_nfft} frames"
        )
    return samples.reshape(nchirp, rng_nfft)
