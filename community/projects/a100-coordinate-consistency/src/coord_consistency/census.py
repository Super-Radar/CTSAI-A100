"""Field-validity census over the target-log corpus.

For every file and every focus column we count, over all target rows:

* ``rows`` -- number of parsed target rows;
* ``nonmissing`` -- cells that parse as finite floats (empty / text -> missing);
* ``nonzero_effective`` -- cells whose magnitude exceeds
  :data:`EFFECTIVE_EPS` (this excludes the float-junk values, e.g. ``1.7e-31``,
  that the RawTarget X column is filled with);
* ``fill_rate`` = nonmissing / rows;
* ``effective_fill_rate`` = nonzero_effective / rows.

RawTarget and TraTarget files are summarised separately.  Expected (and
observed) outcome: the Raw X/Y columns are effectively empty, so they must not
be used as cartesian coordinates.
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np

from .parser import TargetLog, to_float

__all__ = ["FOCUS_FIELDS", "EFFECTIVE_EPS", "file_census", "corpus_census"]

# Columns the census focuses on (all header variants carry them).
FOCUS_FIELDS: Sequence[str] = (
    "ObjId", "range", "speed", "angle", "snr", "X", "Y",
    "heigh", "RCS", "DbfMag",
)

# |value| above this counts as an effective non-zero [units vary by column].
EFFECTIVE_EPS = 1e-6


def file_census(log: TargetLog, fields: Sequence[str] = FOCUS_FIELDS) -> List[Dict[str, object]]:
    """Per-column validity counts for one parsed log."""
    out: List[Dict[str, object]] = []
    for name in fields:
        if name not in log.columns:
            continue
        idx = log.columns.index(name)
        vals = np.array([
            to_float(row[idx]) if idx < len(row) else np.nan
            for fr in log.frames
            for row in fr.rows
        ], dtype=float)
        finite = np.isfinite(vals)
        effective = finite & (np.abs(vals) > EFFECTIVE_EPS)
        n = int(vals.size)
        out.append({
            "kind": log.kind,
            "file": log.filename,
            "column": name,
            "rows": n,
            "nonmissing": int(finite.sum()),
            "nonzero_effective": int(effective.sum()),
            "fill_rate": float(finite.sum()) / n if n else 0.0,
            "effective_fill_rate": float(effective.sum()) / n if n else 0.0,
        })
    return out


def aggregate_rows(rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    """Pool per-file census rows into per-kind AGGREGATE rows."""
    pooled: Dict[str, Dict[str, int]] = {}
    for rec in rows:
        key = f"{rec['kind']}|{rec['column']}"
        acc = pooled.setdefault(key, {"rows": 0, "nonmissing": 0,
                                      "nonzero_effective": 0})
        for k in acc:
            acc[k] += int(rec[k])
    out: List[Dict[str, object]] = []
    for key in sorted(pooled):
        kind, column = key.split("|", 1)
        acc = pooled[key]
        n = acc["rows"]
        out.append({
            "kind": kind,
            "file": "AGGREGATE",
            "column": column,
            "rows": n,
            "nonmissing": acc["nonmissing"],
            "nonzero_effective": acc["nonzero_effective"],
            "fill_rate": acc["nonmissing"] / n if n else 0.0,
            "effective_fill_rate": acc["nonzero_effective"] / n if n else 0.0,
        })
    return out


def corpus_census(logs: Sequence[TargetLog],
                  fields: Sequence[str] = FOCUS_FIELDS) -> List[Dict[str, object]]:
    """Per-file census rows plus per-kind AGGREGATE rows (pooled counts)."""
    rows: List[Dict[str, object]] = []
    for log in logs:
        rows.extend(file_census(log, fields))
    return rows + aggregate_rows(rows)
