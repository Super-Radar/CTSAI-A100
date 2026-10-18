"""Multi-frame parser for RadarTools RawTarget target logs (CTLRR-300-V2_1 sessions).

Log dialect (verified against the two session files shipped in this repository):

- Line 1 is a column header. Observed variants have 57 / 58 / 59 / 62 columns, but
  the first five columns are always ``ObjId,range,speed,angle,snr``; everything
  after them varies. The parser therefore never trusts header names and always
  reads target fields **by position**: range = column 1 [m], angle = column 3
  [deg], snr = column 4 [dB]. Data rows may carry a trailing comma, which only
  appends an empty field and is harmless for positional reads.
- Each frame is the block::

      START,FrameNb:N,...
      YYYY/MM/DD HH:MM:SS:mmm---->            (wall clock, millisecond field)
      heightOffset:0,<epoch_ms>,...            (device epoch in ms = field index 1)
      <target rows: ObjId,range,speed,angle,snr,...>   (zero or more)
      END,Total:M

- After the final ``END`` the file may carry non-frame trailing lines (e.g. a
  firmware signature block) and blank lines anywhere. Both are tolerated.

Timestamps: the epoch-ms field of the ``heightOffset`` metadata line is the
authoritative frame time (it matches the wall-clock line in both shipped
sessions). Frame times must be non-decreasing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

#: positional column indices inside a target row (0-based, after split on ',')
COL_RANGE = 1
COL_ANGLE = 3
COL_SNR = 4
_MIN_TARGET_FIELDS = 5


@dataclass
class Frame:
    """One parsed target frame."""

    index: int  # 0-based position inside the session
    frame_nb: int  # FrameNb reported by the START line
    epoch_ms: int  # device epoch timestamp [ms]
    wall_time: str  # raw wall-clock string from the "---->" line
    targets: np.ndarray  # shape (N, 3): range [m], angle [deg], snr [dB]

    @property
    def n_targets(self) -> int:
        return int(self.targets.shape[0])


@dataclass
class TargetLog:
    """A parsed multi-frame target log."""

    source: Path
    frames: list
    file_size_bytes: int
    header_columns: int
    trailing_lines: list = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def times_ms(self) -> np.ndarray:
        return np.array([f.epoch_ms for f in self.frames], dtype=np.int64)

    @property
    def duration_s(self) -> float:
        t = self.times_ms
        return float(t[-1] - t[0]) / 1000.0

    @property
    def counts(self) -> np.ndarray:
        """Per-frame target counts."""
        return np.array([f.n_targets for f in self.frames], dtype=np.int64)

    def check_times_monotonic(self) -> bool:
        t = self.times_ms
        return bool(np.all(np.diff(t) >= 0))


def _parse_frame_nb(start_line: str) -> int:
    # "START,FrameNb:12,CameraNb:0,..." -> 12
    payload = start_line[len("START,") :]
    for item in payload.split(","):
        if item.startswith("FrameNb:"):
            return int(item[len("FrameNb:") :])
    raise ValueError(f"START line without FrameNb: {start_line!r}")


def _parse_end_total(end_line: str) -> int:
    # "END,Total:32" -> 32
    payload = end_line[len("END,") :]
    for item in payload.split(","):
        if item.startswith("Total:"):
            return int(item[len("Total:") :])
    raise ValueError(f"END line without Total: {end_line!r}")


def parse_target_log(path, strict_total: bool = True) -> TargetLog:
    """Parse a RawTarget CSV into :class:`TargetLog`.

    Parameters
    ----------
    path : str or Path
        Path to the ``*RawTarget.csv`` file.
    strict_total : bool
        When True (default) an ``END,Total:M`` that disagrees with the number of
        parsed target rows raises ``ValueError``.
    """
    path = Path(path)
    file_size = path.stat().st_size

    frames: list[Frame] = []
    trailing: list[str] = []

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().rstrip("\r\n")
        header_columns = len(header.split(","))

        frame_nb = None
        epoch_ms = None
        wall_time = None
        targets: list[tuple] = []
        inside_frame = False

        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line:
                continue

            if line.startswith("START,"):
                if inside_frame:
                    raise ValueError(f"{path}: START without END for FrameNb {frame_nb}")
                frame_nb = _parse_frame_nb(line)
                epoch_ms = None
                wall_time = None
                targets = []
                inside_frame = True
                continue

            if not inside_frame:
                # Non-frame content (e.g. firmware signature after the last END).
                trailing.append(line)
                continue

            if line.endswith("---->"):
                wall_time = line[: -len("---->")]
                continue

            if line.startswith("heightOffset:"):
                epoch_ms = int(line.split(",")[1])
                continue

            if line.startswith("END,"):
                total = _parse_end_total(line)
                if strict_total and total != len(targets):
                    raise ValueError(
                        f"{path}: frame {frame_nb} END,Total:{total} but "
                        f"{len(targets)} target rows parsed"
                    )
                if epoch_ms is None:
                    raise ValueError(
                        f"{path}: frame {frame_nb} has no heightOffset epoch line"
                    )
                frames.append(
                    Frame(
                        index=len(frames),
                        frame_nb=frame_nb,
                        epoch_ms=epoch_ms,
                        wall_time=wall_time or "",
                        targets=np.asarray(targets, dtype=np.float64).reshape(-1, 3),
                    )
                )
                inside_frame = False
                continue

            # Candidate target row: read positionally, tolerate trailing comma.
            cols = line.split(",")
            if len(cols) >= _MIN_TARGET_FIELDS:
                try:
                    int(cols[0])
                    rng = float(cols[COL_RANGE])
                    ang = float(cols[COL_ANGLE])
                    snr = float(cols[COL_SNR])
                except (ValueError, IndexError):
                    if any(c.strip() for c in cols):
                        trailing.append(line)  # unexpected non-numeric payload
                    continue
                targets.append((rng, ang, snr))

    if inside_frame:
        raise ValueError(f"{path}: truncated frame (START without END) at end of file")

    log = TargetLog(
        source=path,
        frames=frames,
        file_size_bytes=file_size,
        header_columns=header_columns,
        trailing_lines=trailing,
    )
    if log.n_frames > 1 and not log.check_times_monotonic():
        raise ValueError(f"{path}: frame epoch timestamps are not non-decreasing")
    return log
