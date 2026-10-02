"""Streaming parser for RadarTools ``*_RawTarget.csv`` / ``*_TraTarget.csv`` target logs.

File layout (verified against the 41 corpus files shipped in
``RadarTools/RadarTools_Release/data/``):

* line 1: comma separated column header (58/59/62 columns depending on the
  host-tool version; the key columns always keep the same indices);
* each frame starts with ``START,FrameNb:N,CameraNb:..,HeSaiNb:..,ARS410Nb:..``;
* next line: ``YYYY/MM/DD HH:MM:SS:mmm---->`` wall-clock timestamp;
* next line: ``heightOffset:0,<epoch_ms>,<frame_interval_us>,...`` metadata
  (the second comma field is the epoch-millisecond stamp);
* then one line per target (numeric CSV fields aligned with the header,
  rows may carry a trailing comma, may be shorter than the header, or may be
  blank junk such as ``,,,,,,,,"`);
* every frame block closes with an ``END,Total:N`` line (corpus files carry
  one END per frame), and the file ends with a firmware footer line starting
  with ``BB Hw:``.

The parser is pure standard library + numpy (no pandas), tolerant to blank
lines, stray whitespace and ragged rows.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

__all__ = ["Frame", "TargetLog", "parse_file", "parse_string", "to_float"]

# Columns this study relies on; all variants of the header carry them.
KEY_COLUMNS: Sequence[str] = (
    "ObjId", "range", "speed", "angle", "snr", "X", "Y", "RCS", "DbfMag",
)


def to_float(cell: str) -> float:
    """Parse a CSV cell to float; empty / non-numeric cells become NaN."""
    s = cell.strip()
    if not s:
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


@dataclass
class Frame:
    """One radar frame and its target rows."""

    frame_nb: Optional[int] = None
    start_fields: Dict[str, str] = field(default_factory=dict)
    timestamp: Optional[str] = None      # "YYYY/MM/DD HH:MM:SS:mmm"
    epoch_ms: Optional[float] = None     # epoch milliseconds from the meta line
    rows: List[List[str]] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.rows)


@dataclass
class TargetLog:
    """Parsed target log: header + all frames."""

    filename: str
    columns: List[str]
    kind: str                     # "Raw" or "Tra"
    frames: List[Frame] = field(default_factory=list)
    n_rows_total: int = 0
    n_rows_skipped: int = 0       # blank-junk / unparsable rows

    # ------------------------------------------------------------------ #
    def column(self, name: str) -> np.ndarray:
        """Stack one numeric column across every frame (NaN where missing)."""
        if name not in self.columns:
            raise KeyError(f"column {name!r} not present in {self.filename}")
        idx = self.columns.index(name)
        out = np.full(self.n_rows_total, math.nan)
        pos = 0
        for fr in self.frames:
            for row in fr.rows:
                cell = row[idx] if idx < len(row) else ""
                out[pos] = to_float(cell)
                pos += 1
        return out

    def column_str(self, name: str) -> List[str]:
        """Stack one column as raw strings ("" where missing)."""
        idx = self.columns.index(name)
        out: List[str] = []
        for fr in self.frames:
            for row in fr.rows:
                out.append(row[idx].strip() if idx < len(row) else "")
        return out

    def frame_index(self) -> np.ndarray:
        """Row -> frame ordinal map."""
        idx = np.empty(self.n_rows_total, dtype=np.int64)
        pos = 0
        for i, fr in enumerate(self.frames):
            for _ in fr.rows:
                idx[pos] = i
                pos += 1
        return idx

    def valid_mask(self, *names: str) -> np.ndarray:
        m = np.ones(self.n_rows_total, dtype=bool)
        for n in names:
            m &= np.isfinite(self.column(n))
        return m


# ---------------------------------------------------------------------- #
def _looks_like_data_row(line: str) -> bool:
    c = line[0]
    return c.isdigit() or c == "-" or c == "+"


def _parse_start(line: str) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for part in line.split(",")[1:]:
        k, _, v = part.partition(":")
        fields[k.strip()] = v.strip()
    return fields


def parse_string(text: str, filename: str = "<string>") -> TargetLog:
    """Parse target-log CSV text (LF or CRLF, blank lines allowed)."""
    kind = "Tra" if "TraTarget" in filename else "Raw"
    log = TargetLog(filename=filename, columns=[], kind=kind)

    current: Optional[Frame] = None
    saw_header = False
    ncols = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue  # tolerate blank lines anywhere

        if not saw_header:
            # First non-blank line is the column header (never a START line).
            log.columns = [c.strip() for c in line.split(",")]
            ncols = len(log.columns)
            saw_header = True
            continue

        if line.startswith("START"):
            sf = _parse_start(line)
            try:
                frame_nb = int(sf.get("FrameNb", ""))
            except ValueError:
                frame_nb = None
            current = Frame(frame_nb=frame_nb, start_fields=sf)
            log.frames.append(current)
            continue

        if line.startswith("END"):
            # ``END,Total:N`` closes a frame block; corpus files carry one
            # END per frame (and a firmware footer after the last one), so
            # parsing must continue with the next START.
            current = None
            continue

        if current is None:
            continue  # stray line before the first START / between frames

        if "---->" in line:
            current.timestamp = line.split("---->")[0].strip()
            continue

        if line.startswith("heightOffset"):
            parts = line.split(",")
            if len(parts) > 1:
                v = to_float(parts[1])
                if not math.isnan(v):
                    current.epoch_ms = v
            continue

        if not _looks_like_data_row(line):
            # junk row of empty fields, e.g. ",,,,,,,," seen in the corpus
            if line.split(",")[0] == "":
                log.n_rows_skipped += 1
            continue  # any other metadata-ish line

        cells = [c.strip() for c in line.split(",")]
        if not any(cells):
            log.n_rows_skipped += 1
            continue
        # A data row must start with a numeric ObjId.
        if math.isnan(to_float(cells[0])):
            log.n_rows_skipped += 1
            continue
        if len(cells) < ncols:
            cells = cells + [""] * (ncols - len(cells))  # pad ragged row
        current.rows.append(cells[:ncols])
        log.n_rows_total += 1

    return log


def parse_file(path: str) -> TargetLog:
    """Parse one target-log CSV file (utf-8, replacement on bad bytes)."""
    import os

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as fh:
        return parse_string(fh.read(), filename=os.path.basename(path))
