"""Vector ASC CAN/CAN-FD log parser.

The RadarTools host tool records CAN traffic next to its target logs in
Vector ASC format (``// version 7.2.0``).  Message lines look like::

    0.000000 CANFD 1 740 Rx 1 0 d 15 64 02 06 ...

i.e. ``<relative-seconds> CANFD <channel> <id-hex> Rx <brs> <esi> d <dlc> <data bytes...>``.

Two facts measured on the shipped logs drive the parser design:

* The ``dlc`` field is a CAN-FD length *code* and regularly disagrees with the
  number of data-byte tokens actually present (e.g. ``d 15`` followed by 64
  hex bytes).  The byte count therefore always comes from counting the actual
  data tokens on the line, never from the DLC field.
* Header lines (``date ...``, ``base ...``), ``//`` comments and completely
  empty logs (header only) must be tolerated; 46 of the 56 ASC files in the
  repository contain no messages at all.

Only timing (``time_s``), channel, identifier and payload length are kept.
Payload values are intentionally ignored: this project does statistics on
logging load, not protocol decoding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_MESSAGE_RE = re.compile(
    r"^(?P<time>[0-9]+\.[0-9]+)\s+"
    r"(?P<kind>CANFD|CAN)\s+"
    r"(?P<channel>[0-9]+)\s+"
    r"(?P<can_id>[0-9a-fA-F]+)\s+"
    r"(?P<direction>Rx|Tx)\s+"
    r"(?:(?P<brs>[0-9]+)\s+(?P<esi>[0-9]+)\s+)?"  # CAN-FD flag pair, absent on classic CAN
    r"(?P<dlc_tok>[dD])\s+"
    r"(?P<dlc>[0-9]+)"
    r"(?P<payload>(?:\s+[0-9a-fA-F]{1,2})*)\s*$"
)


@dataclass
class AscLog:
    """Parsed contents of one ASC file (messages only; payload bytes counted)."""

    path: Path
    date_header: str | None
    base_header: str | None
    version_comment: str | None
    n_skipped_lines: int
    time_s: np.ndarray  # float64, relative seconds, sorted ascending
    channel: np.ndarray  # int32
    can_id: np.ndarray  # int32, numeric value of the hex identifier
    n_bytes: np.ndarray  # int32, actual data-byte token count per message

    def __len__(self) -> int:
        return int(self.time_s.size)

    @property
    def duration_s(self) -> float:
        if self.time_s.size == 0:
            return 0.0
        return float(self.time_s[-1] - self.time_s[0])

    @property
    def total_bytes(self) -> int:
        return int(self.n_bytes.sum())

    @property
    def mean_rate_msgs_per_s(self) -> float:
        dur = self.duration_s
        if self.time_s.size == 0 or dur <= 0.0:
            return float("nan")
        return float(self.time_s.size) / dur


def parse_asc(path: str | Path) -> AscLog:
    """Parse one ASC file.

    Lines that are neither header, ``//`` comment, nor a recognisable
    ``CANFD``/``CAN`` message line are counted in ``n_skipped_lines`` rather
    than raising, so that a partially unusual log still yields its usable
    messages.
    """
    path = Path(path)
    times: list[float] = []
    channels: list[int] = []
    ids: list[int] = []
    nbytes: list[int] = []
    date_header: str | None = None
    base_header: str | None = None
    version_comment: str | None = None
    skipped = 0

    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\r\n")
            if not line.strip():
                continue
            if line.startswith("//"):
                if version_comment is None and "version" in line:
                    version_comment = line
                continue
            if line.startswith("date "):
                date_header = line
                continue
            if line.startswith("base "):
                base_header = line
                continue
            match = _MESSAGE_RE.match(line)
            if match is None:
                skipped += 1
                continue
            times.append(float(match["time"]))
            channels.append(int(match["channel"]))
            ids.append(int(match["can_id"], 16))
            # DLC field is a length code and can mismatch; count real tokens.
            nbytes.append(len(match["payload"].split()))

    order = np.argsort(np.asarray(times, dtype=np.float64), kind="stable")
    return AscLog(
        path=path,
        date_header=date_header,
        base_header=base_header,
        version_comment=version_comment,
        n_skipped_lines=skipped,
        time_s=np.asarray(times, dtype=np.float64)[order],
        channel=np.asarray(channels, dtype=np.int32)[order],
        can_id=np.asarray(ids, dtype=np.int32)[order],
        n_bytes=np.asarray(nbytes, dtype=np.int32)[order],
    )


def parse_asc_dir(directory: str | Path, pattern: str = "*_CanData.asc") -> list[AscLog]:
    """Parse every ASC file matching ``pattern`` in ``directory``.

    Files are returned sorted by filename.  Empty logs (header only) are
    included so callers can account for them; check ``len(log)`` to filter.
    """
    directory = Path(directory)
    paths = sorted(directory.glob(pattern))
    return [parse_asc(p) for p in paths]


def sha256_of(path: str | Path, chunk_size: int = 1 << 20) -> str:
    """Hex SHA-256 of a file, streamed."""
    import hashlib

    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()
