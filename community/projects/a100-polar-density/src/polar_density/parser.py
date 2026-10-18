"""Multi-frame RawTarget CSV parser for RadarTools host-tool capture logs.

Observed on-disk format (verified on the 22 ``*_RawTarget.csv`` captures that
ship with this repository):

* Line 1 is a single header row.  The column count varies between captures
  (58, 59 and 62 column variants were measured) but the first five columns are
  always ``ObjId,range,speed,angle,snr``.  Target rows are therefore decoded
  by *positional* indexing of the first five fields, never by column name.
* Every frame block then looks like::

      START,FrameNb:0,CameraNb:2029,...
      2026/03/10 15:34:39:090---->          <- timestamp line (may end with CR)
      heightOffset:0,1773128079090,...      <- frame metadata, 2nd field = epoch ms
      0,2.31,0,-22.45,36.5,...              <- one row per detected target
      END,Total:16                          <- frame terminator

* Line endings are mixed (some lines carry a trailing ``CR``), rows may end
  with a dangling comma, and blank lines / stray whitespace occur.  The parser
  tolerates all of that; anything that is not a ``START`` / timestamp /
  ``heightOffset:`` / ``END`` / numeric row is skipped (the header row falls
  into that bucket and is recorded for reference).

The parser never assumes a single file-wide table: files are decoded strictly
frame-by-frame as described above.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: Column order produced for every target row (positional, header-agnostic).
TARGET_COLUMNS = ("objid", "range_m", "speed_ms", "angle_deg", "snr_db")

_TIMESTAMP_SUFFIX = "---->"
_META_PREFIXES = ("heightOffset:",)
_START_PREFIX = "START,"
_END_PREFIX = "END,"


@dataclass
class RadarFrame:
    """One frame block (``START`` ... ``END``) of a capture log."""

    frame_nb: int
    epoch_ms: float | None
    targets: np.ndarray  # shape (n, 5), columns follow TARGET_COLUMNS

    def __len__(self) -> int:
        return int(self.targets.shape[0])


@dataclass
class CaptureLog:
    """All frames of one capture file."""

    path: Path
    header: str = ""
    frames: list[RadarFrame] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        """Frame count, including frames that contain zero targets."""
        return len(self.frames)

    @property
    def n_targets(self) -> int:
        """Total number of target rows across all frames."""
        return int(sum(len(f) for f in self.frames))

    def all_targets(self) -> np.ndarray:
        """Concatenated (N, 5) target array (objid, range, speed, angle, snr)."""
        blocks = [f.targets for f in self.frames if len(f)]
        if not blocks:
            return np.empty((0, len(TARGET_COLUMNS)), dtype=float)
        return np.vstack(blocks)


def _to_float(token: str) -> float | None:
    try:
        return float(token)
    except ValueError:
        return None


def _frame_nb(start_line: str) -> int:
    """Extract the integer after ``FrameNb:`` from a ``START,...`` line."""
    for field_str in start_line.split(","):
        field_str = field_str.strip()
        if field_str.startswith("FrameNb:"):
            value = _to_float(field_str[len("FrameNb:"):])
            if value is not None:
                return int(value)
    return -1


def _parse_row(line: str) -> list[float] | None:
    """Decode the first five comma-separated fields of a candidate target row.

    Returns ``None`` when any of those fields is not numeric (e.g. the header
    row ``ObjId,range,...`` or any other non-data line).
    """
    parts = line.split(",")
    if len(parts) < len(TARGET_COLUMNS):
        return None
    values: list[float] = []
    for token in parts[: len(TARGET_COLUMNS)]:
        value = _to_float(token.strip())
        if value is None:
            return None
        values.append(value)
    return values


def parse_capture(path: str | Path) -> CaptureLog:
    """Parse one ``*_RawTarget.csv`` capture into per-frame target arrays."""
    path = Path(path)
    log = CaptureLog(path=path)
    frame_nb = -1
    epoch_ms: float | None = None
    rows: list[list[float]] = []
    inside_frame = False

    def close_frame() -> None:
        nonlocal rows
        if rows:
            targets = np.asarray(rows, dtype=float).reshape(-1, len(TARGET_COLUMNS))
        else:
            targets = np.empty((0, len(TARGET_COLUMNS)), dtype=float)
        log.frames.append(
            RadarFrame(frame_nb=frame_nb, epoch_ms=epoch_ms, targets=targets)
        )
        rows = []

    with open(path, "r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue  # tolerate blank lines
            if line.startswith(_START_PREFIX):
                if inside_frame:
                    close_frame()  # unterminated frame: close it defensively
                frame_nb = _frame_nb(line)
                epoch_ms = None
                rows = []
                inside_frame = True
                continue
            if line.endswith(_TIMESTAMP_SUFFIX):
                continue  # "YYYY/MM/DD HH:MM:SS:mmm---->" wall-clock line
            if any(line.startswith(p) for p in _META_PREFIXES):
                if epoch_ms is None:
                    parts = line.split(",")
                    if len(parts) > 1:
                        epoch_ms = _to_float(parts[1].strip())
                continue
            if line.startswith(_END_PREFIX):
                if inside_frame:
                    close_frame()
                    inside_frame = False
                continue
            if not inside_frame:
                # Outside a frame block the only legal row is the file header.
                if not log.header and line.split(",")[0].strip() == "ObjId":
                    log.header = line
                continue
            values = _parse_row(line)
            if values is None:
                # Non-numeric row inside a frame: record header once, else skip.
                if not log.header and line.split(",")[0].strip() == "ObjId":
                    log.header = line
                continue
            rows.append(values)

    if inside_frame:
        close_frame()  # trailing frame without END line
    return log
