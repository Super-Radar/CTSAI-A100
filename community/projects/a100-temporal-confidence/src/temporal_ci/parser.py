"""Frame-block parser for RadarTools ``RawTarget`` CSV logs.

The log layout (observed for the CTLRR-300-V2_1 example assets shipped in this
repository) is one header row followed by frame blocks::

    ObjId,range,speed,angle,snr,...                       <- header row
    START,FrameNb:0,CameraNb:2029,HeSaiNb:0,ARS410Nb:0   <- frame block opens
    2026/03/10 15:34:39:090---->                          <- frame timestamp
    heightOffset:0,1773128079090,...                      <- metadata row (epoch ms in field 1)
    0,2.31,0,-22.45,36.5,...                              <- target rows
    END,Total:16                                          <- frame block closes

Observed header widths are 57 / 58 / 59 / 62 fields, and data rows may carry
one trailing empty field.  All values are therefore read positionally: column
0 ``ObjId``, 1 ``range`` [m], 2 ``speed`` [m/s], 3 ``angle`` [deg], 4 ``snr``.
Blank lines and trailing firmware banner lines are tolerated and counted.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_START_RE = re.compile(r"^START,FrameNb:\s*(\d+)")
_END_RE = re.compile(r"^END,Total:\s*(\d+)")
_TS_RE = re.compile(r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}:\d{3})---->")
_EPOCH_RE = re.compile(r"^heightOffset:[^,]*,(\d{12,15})")


@dataclass
class Frame:
    """One parsed frame block with its target rows."""

    frame_nb: int
    ts_text: str | None
    ts_epoch_ms: int | None
    obj_id: np.ndarray
    rng: np.ndarray  # range [m]
    speed: np.ndarray  # radial speed [m/s]
    angle: np.ndarray  # azimuth [deg]
    snr: np.ndarray

    @property
    def n_targets(self) -> int:
        return int(self.obj_id.size)


@dataclass
class ParseResult:
    """All frames of one log file plus parse-quality counters."""

    frames: list[Frame]
    header: list[str]
    n_unknown_lines: int = 0
    n_total_mismatch: int = 0

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    def target_counts(self) -> np.ndarray:
        return np.array([f.n_targets for f in self.frames], dtype=np.int64)

    def total_targets(self) -> int:
        return int(sum(f.n_targets for f in self.frames))

    def epoch_ms(self) -> np.ndarray:
        return np.array([f.ts_epoch_ms for f in self.frames], dtype=np.int64)


def _try_target_row(fields: list[str]):
    """Return (obj_id, range, speed, angle, snr) or None if not a target row."""
    if len(fields) < 5:
        return None
    try:
        obj = int(fields[0])
        vals = (float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]))
    except (TypeError, ValueError):
        return None
    return (obj, *vals)


def parse_raw_target_csv(path: str | Path) -> ParseResult:
    """Parse a multi-frame RawTarget CSV log into per-frame target arrays."""
    path = Path(path)
    with open(path, "r", encoding="utf-8-sig", errors="replace") as fh:
        lines = fh.read().splitlines()

    header: list[str] = []
    frames: list[Frame] = []
    n_unknown = 0
    n_mismatch = 0
    cur: dict | None = None

    for raw in lines:
        line = raw.strip("\r").strip()
        if not line:
            continue
        if not header:
            if line.startswith("ObjId"):
                header = [h.strip() for h in line.split(",")]
            else:
                n_unknown += 1
            continue
        m = _START_RE.match(line)
        if m:
            cur = {"frame_nb": int(m.group(1)), "ts_text": None, "ts_epoch_ms": None, "rows": []}
            continue
        if cur is not None:
            m = _TS_RE.match(line)
            if m:
                cur["ts_text"] = m.group(1)
                continue
            m = _EPOCH_RE.match(line)
            if m:
                cur["ts_epoch_ms"] = int(m.group(1))
                continue
            m = _END_RE.match(line)
            if m:
                total = int(m.group(1))
                rows = cur["rows"]
                if total != len(rows):
                    n_mismatch += 1
                arr = np.asarray(rows, dtype=np.float64).reshape(len(rows), 5)
                frames.append(
                    Frame(
                        frame_nb=cur["frame_nb"],
                        ts_text=cur["ts_text"],
                        ts_epoch_ms=cur["ts_epoch_ms"],
                        obj_id=arr[:, 0].astype(np.int64),
                        rng=arr[:, 1].copy(),
                        speed=arr[:, 2].copy(),
                        angle=arr[:, 3].copy(),
                        snr=arr[:, 4].copy(),
                    )
                )
                cur = None
                continue
            row = _try_target_row(line.split(","))
            if row is not None:
                cur["rows"].append(row)
                continue
        n_unknown += 1

    return ParseResult(
        frames=frames,
        header=header,
        n_unknown_lines=n_unknown,
        n_total_mismatch=n_mismatch,
    )
