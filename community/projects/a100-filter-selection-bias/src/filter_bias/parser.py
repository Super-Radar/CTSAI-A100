"""Multi-frame RawTarget CSV parser.

Frame log format (verified on 22 corpus files; named-column headers come in
57 / 58 / 59 / 62 field variants -- the main file's header has 57 names and
its data rows carry a trailing comma, i.e. 58 raw fields; the first five
fields are always ``ObjId,range,speed,angle,snr``)::

    ObjId,range,speed,angle,snr,...            <- file-level header (once)
    START,FrameNb:0,CameraNb:2029,...          <- frame start
    2026/03/10 15:34:39:090---->               <- frame timestamp
    heightOffset:0,1773128079090,...           <- frame metadata
    0,2.31,0,-22.45,36.5,...                   <- target rows (>= 5 fields)
    END,Total:16                               <- frame end

Columns are consumed positionally (never by header name, the header is only
counted). Rows in which any of range / speed / angle / snr is missing, empty
or non-finite are dropped entirely -- they are never zero-filled.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["PointCloud", "FileStats", "parse_text", "parse_file", "parse_files"]

_FRAME_RE = re.compile(r"FrameNb:(-?[0-9]+)")
_TOTAL_RE = re.compile(r"Total:([0-9]+)")
_TIMESTAMP_MARKER = "---->"
_META_PREFIX = "heightOffset:"

# Positional column indices (stable across the 58/59/62 column variants).
_COL_OBJID = 0
_COL_RANGE = 1
_COL_SPEED = 2
_COL_ANGLE = 3
_COL_SNR = 4
_MIN_FIELDS = 5


@dataclass(frozen=True)
class PointCloud:
    """Flat pool of raw target detections across frames and files."""

    file_id: np.ndarray  # int32, index into ``file_names``
    frame: np.ndarray  # int32, FrameNb within its file
    obj_id: np.ndarray  # int32, ObjId within its frame
    range_m: np.ndarray  # float64, m
    speed_ms: np.ndarray  # float64, m/s
    angle_deg: np.ndarray  # float64, deg
    snr_db: np.ndarray  # float64, dB
    file_names: tuple  # tuple[str, ...]

    def __len__(self) -> int:
        return int(self.range_m.shape[0])


@dataclass(frozen=True)
class FileStats:
    """Per-file parse summary (for the data provenance section)."""

    name: str
    header_columns: int
    frames: int
    targets: int
    declared_total_sum: int


def _parse_stream(fh) -> tuple[list, list, int]:
    """Parse one already-open RawTarget stream.

    Returns ``(rows, frames, header_columns)`` where ``rows`` is a list of
    ``(frame_nb, obj_id, range, speed, angle, snr)`` tuples and ``frames``
    is a list of ``(frame_nb, declared_total)``.
    """
    header = fh.readline().rstrip("\r\n")
    header_columns = len(header.split(",")) if header else 0

    rows: list[tuple] = []
    frames: list[tuple] = []
    inside = False
    frame_nb = -1
    for raw in fh:
        line = raw.rstrip("\r\n")
        if not line:
            continue
        if line.startswith("START"):
            match = _FRAME_RE.search(line)
            frame_nb = int(match.group(1)) if match else -1
            inside = True
            continue
        if not inside:
            continue
        if line.startswith("END"):
            match = _TOTAL_RE.search(line)
            frames.append((frame_nb, int(match.group(1)) if match else -1))
            inside = False
            continue
        if _TIMESTAMP_MARKER in line or line.startswith(_META_PREFIX):
            continue
        parts = line.split(",")
        if len(parts) < _MIN_FIELDS:
            continue
        try:
            obj_id = int(parts[_COL_OBJID])
            rng = float(parts[_COL_RANGE])
            spd = float(parts[_COL_SPEED])
            ang = float(parts[_COL_ANGLE])
            snr = float(parts[_COL_SNR])
        except ValueError:
            continue  # malformed numeric field -> row excluded, not zero-filled
        if not (np.isfinite(rng) and np.isfinite(spd) and np.isfinite(ang) and np.isfinite(snr)):
            continue  # NaN/inf field -> row excluded, not zero-filled
        rows.append((frame_nb, obj_id, rng, spd, ang, snr))
    return rows, frames, header_columns


def parse_text(text: str) -> tuple[list, list, int]:
    """Parse a RawTarget log held in a string (used by the test suite)."""
    return _parse_stream(io.StringIO(text))


def parse_file(path) -> tuple[list, list, int]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return _parse_stream(fh)


def _name_of(path) -> str:
    return str(path).replace(chr(92), "/")


def parse_files(paths) -> tuple[PointCloud, list]:
    """Parse several RawTarget logs and pool them into one :class:`PointCloud`."""
    file_names: list[str] = []
    stats: list[FileStats] = []
    file_id_parts, frame_parts, obj_parts = [], [], []
    range_parts, speed_parts, angle_parts, snr_parts = [], [], [], []

    for file_idx, path in enumerate(paths):
        path = Path(path)
        rows, frames, header_columns = parse_file(path)
        name = _name_of(path)
        file_names.append(name)
        stats.append(
            FileStats(
                name=name,
                header_columns=header_columns,
                frames=len(frames),
                targets=len(rows),
                declared_total_sum=sum(t for _, t in frames if t >= 0),
            )
        )
        if not rows:
            continue
        arr = np.asarray(rows, dtype=np.float64)
        file_id_parts.append(np.full(arr.shape[0], file_idx, dtype=np.int32))
        frame_parts.append(arr[:, 0].astype(np.int32))
        obj_parts.append(arr[:, 1].astype(np.int32))
        range_parts.append(arr[:, 2])
        speed_parts.append(arr[:, 3])
        angle_parts.append(arr[:, 4])
        snr_parts.append(arr[:, 5])

    def _cat(parts, dtype):
        if not parts:
            return np.zeros(0, dtype=dtype)
        return np.concatenate(parts)

    cloud = PointCloud(
        file_id=_cat(file_id_parts, np.int32),
        frame=_cat(frame_parts, np.int32),
        obj_id=_cat(obj_parts, np.int32),
        range_m=_cat(range_parts, np.float64),
        speed_ms=_cat(speed_parts, np.float64),
        angle_deg=_cat(angle_parts, np.float64),
        snr_db=_cat(snr_parts, np.float64),
        file_names=tuple(file_names),
    )
    return cloud, stats
