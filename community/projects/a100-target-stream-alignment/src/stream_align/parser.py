"""Parser for RadarTools target-log CSV files (RawTarget / TraTarget).

File layout (verified against ``RadarTools/RadarTools_Release/data/``):

* Line 1 is a 60-column header starting with ``ObjId,range,speed,...``.
* Every frame then consists of:
  - a ``START,FrameNb:<N>,CameraNb:...`` line,
  - a wall-clock line ``YYYY/MM/DD HH:MM:SS:mmm---->``,
  - a metadata line ``heightOffset:0,<epoch_ms>,...`` whose second field is
    an epoch-millisecond value expressing Beijing (UTC+8) wall-clock time
    (verified: converting the ``---->`` timestamp to epoch ms under UTC+8
    reproduces this field exactly on all sampled files),
  - one CSV row per detected/tracked target, until the next ``START`` line.

The parser is intentionally defensive: FrameNb resets / rollbacks split the
frame sequence into segments, and wall-clock vs epoch consistency is checked
per frame so that later alignment stages never rely on unverified time
sources.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

#: Timezone carried by the epoch_ms field (Beijing time, verified on sample data).
BEIJING_TZ = timezone(timedelta(hours=8))

_START_RE = re.compile(r"^START,FrameNb:(\d+)")
_TS_RE = re.compile(
    r"^(\d{4})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2}):(\d{3})---->"
)
_META_RE = re.compile(r"^heightOffset:(-?\d+),(\d+),")


def wallclock_to_epoch_ms(wallclock: datetime) -> int:
    """Interpret a naive wall-clock datetime as UTC+8 and return epoch ms.

    ``round`` (not ``int``) keeps millisecond precision despite the float
    second-count round trip on values near 1.8e12 ms.
    """
    return round(wallclock.replace(tzinfo=BEIJING_TZ).timestamp() * 1000)


@dataclass
class Frame:
    """One parsed frame of a target stream."""

    frame_idx: int  # 0-based position in file order
    frame_nb: int  # FrameNb field of the START line
    segment: int  # 0-based segment id; increments after every FrameNb reset/rollback
    wallclock: datetime | None = None  # parsed from the ----> line (UTC+8 wall clock)
    epoch_ms: int | None = None  # second field of the heightOffset metadata line
    n_targets: int = 0  # target rows until the next START line
    ts_epoch_consistent: bool | None = None  # wallclock (as UTC+8) == epoch_ms?


@dataclass
class StreamLog:
    """A parsed single-stream target log."""

    path: Path
    kind: str  # "raw" or "tra"
    frames: list[Frame] = field(default_factory=list)

    @property
    def n_frames(self) -> int:
        return len(self.frames)

    @property
    def n_segments(self) -> int:
        return (self.frames[-1].segment + 1) if self.frames else 0

    @property
    def n_resets(self) -> int:
        return max(0, self.n_segments - 1)

    @property
    def n_ts_epoch_mismatch(self) -> int:
        return sum(1 for f in self.frames if f.ts_epoch_consistent is False)

    @property
    def n_missing_epoch(self) -> int:
        return sum(1 for f in self.frames if f.epoch_ms is None)

    def duplicate_epoch_count(self) -> int:
        """Number of frames beyond the first for each repeated epoch_ms value."""
        seen: set[int] = set()
        dups = 0
        for f in self.frames:
            if f.epoch_ms is None:
                continue
            if f.epoch_ms in seen:
                dups += 1
            seen.add(f.epoch_ms)
        return dups


def parse_target_csv(path: str | Path, kind: str = "raw") -> StreamLog:
    """Parse one RawTarget/TraTarget CSV into a :class:`StreamLog`."""
    path = Path(path)
    log = StreamLog(path=path, kind=kind)
    cur: Frame | None = None
    segment = 0
    prev_frame_nb: int | None = None
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line:
                continue
            m = _START_RE.match(line)
            if m:
                if cur is not None:
                    log.frames.append(cur)
                frame_nb = int(m.group(1))
                if prev_frame_nb is not None and frame_nb <= prev_frame_nb:
                    # FrameNb reset or rollback: start a new segment.
                    segment += 1
                prev_frame_nb = frame_nb
                cur = Frame(
                    frame_idx=len(log.frames),
                    frame_nb=frame_nb,
                    segment=segment,
                )
                continue
            if cur is None:
                # Header line (ObjId,...) or stray content before the first START.
                continue
            m = _TS_RE.match(line)
            if m:
                y, mo, d, h, mi, s, ms = (int(g) for g in m.groups())
                cur.wallclock = datetime(y, mo, d, h, mi, s, ms * 1000)
                _refresh_consistency(cur)
                continue
            if line.startswith("heightOffset:"):
                m2 = _META_RE.match(line)
                if m2:
                    cur.epoch_ms = int(m2.group(2))
                    _refresh_consistency(cur)
                continue
            if line.startswith("ObjId"):
                continue
            # Candidate target row: first CSV field must be an integer ObjId.
            first = line.split(",", 1)[0]
            try:
                int(first)
            except ValueError:
                continue
            cur.n_targets += 1
    if cur is not None:
        log.frames.append(cur)
    return log


def _refresh_consistency(frame: Frame) -> None:
    """Mark ts_epoch_consistent once both time fields of a frame are known."""
    if frame.wallclock is not None and frame.epoch_ms is not None:
        frame.ts_epoch_consistent = (
            wallclock_to_epoch_ms(frame.wallclock) == frame.epoch_ms
        )
