"""Alignment of a RawTarget stream against a TraTarget stream.

Three pairing methods are provided.  All of them obey two hard invariants:

* **one-to-one** - every frame of either stream is used by at most one pair;
* **within-segment** - a pair never spans two FrameNb segments (segments are
  derived from FrameNb reset/rollback detection in :mod:`stream_align.parser`),
  so a FrameNb reset cannot silently glue unrelated frames together.

Methods:

* **A - FrameNb equality** (:func:`align_by_framenb`): pairs frames whose
  ``(segment, FrameNb)`` keys are equal and unique on both sides.  This is the
  "naive merge" baseline; it looks productive but, as the results show, its
  temporal residuals can reach seconds when the two streams are emitted at
  different rates.
* **B - exact timestamp** (:func:`align_exact_time`): pairs frames carrying an
  identical ``epoch_ms``.  Duplicate epochs on either side make the pairing
  ambiguous and are counted, not guessed.
* **C - tolerance matching** (:func:`align_tolerance`): within each segment,
  walks both time-sorted frame sequences with a two-pointer sweep and greedily
  accepts a pair when ``|t_raw - t_tra| <= tol_ms``, preferring a strictly
  closer upcoming Tra frame (one-step look-ahead).  The sweep yields
  time-monotonic, one-to-one pairs by construction; frames without a partner
  inside the tolerance abstain.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from stream_align.parser import StreamLog


@dataclass
class AlignResult:
    """Result of one alignment run."""

    method: str
    pairs: list[tuple[int, int]] = field(default_factory=list)
    #: pairs[i] = (raw frame_idx, tra frame_idx); residual = t_tra - t_raw in ms
    residuals_ms: list[int] = field(default_factory=list)
    ambiguous_keys: int = 0  # keys/timestamps rejected for ambiguity

    @property
    def n_pairs(self) -> int:
        return len(self.pairs)

    def coverage(self, raw: StreamLog, tra: StreamLog) -> tuple[float, float]:
        """(coverage of raw frames, coverage of tra frames)."""
        cov_raw = self.n_pairs / raw.n_frames if raw.n_frames else 0.0
        cov_tra = self.n_pairs / tra.n_frames if tra.n_frames else 0.0
        return cov_raw, cov_tra


def _residual(raw_frame, tra_frame) -> int | None:
    if raw_frame.epoch_ms is None or tra_frame.epoch_ms is None:
        return None
    return tra_frame.epoch_ms - raw_frame.epoch_ms


def _finalize(method, pairs, ambiguous) -> AlignResult:
    return AlignResult(method=method, pairs=pairs, ambiguous_keys=ambiguous)


def align_by_framenb(raw: StreamLog, tra: StreamLog) -> AlignResult:
    """Method A: pair frames with equal (segment, FrameNb), unique on both sides."""
    raw_map: dict[tuple[int, int], list[int]] = {}
    tra_map: dict[tuple[int, int], list[int]] = {}
    for f in raw.frames:
        raw_map.setdefault((f.segment, f.frame_nb), []).append(f.frame_idx)
    for f in tra.frames:
        tra_map.setdefault((f.segment, f.frame_nb), []).append(f.frame_idx)

    pairs: list[tuple[int, int]] = []
    residuals: list[int] = []
    ambiguous = 0
    for key in sorted(set(raw_map) & set(tra_map)):
        r_idx, t_idx = raw_map[key], tra_map[key]
        if len(r_idx) > 1 or len(t_idx) > 1:
            ambiguous += 1  # duplicate FrameNb within a segment: abstain
            continue
        pairs.append((r_idx[0], t_idx[0]))
        res = _residual(raw.frames[r_idx[0]], tra.frames[t_idx[0]])
        if res is not None:
            residuals.append(res)
    result = _finalize("A_framenb", pairs, ambiguous)
    result.residuals_ms = residuals
    return result


def align_exact_time(raw: StreamLog, tra: StreamLog) -> AlignResult:
    """Method B: pair frames carrying an identical epoch_ms (no tolerance)."""
    raw_map: dict[int, list[int]] = {}
    tra_map: dict[int, list[int]] = {}
    for f in raw.frames:
        if f.epoch_ms is None:
            continue
        raw_map.setdefault(f.epoch_ms, []).append(f.frame_idx)
    for f in tra.frames:
        if f.epoch_ms is None:
            continue
        tra_map.setdefault(f.epoch_ms, []).append(f.frame_idx)

    pairs: list[tuple[int, int]] = []
    residuals: list[int] = []
    ambiguous = 0
    for ts in sorted(set(raw_map) & set(tra_map)):
        r_idx, t_idx = raw_map[ts], tra_map[ts]
        if len(r_idx) > 1 or len(t_idx) > 1:
            ambiguous += 1  # duplicate timestamps: abstain instead of guessing
            continue
        if raw.frames[r_idx[0]].segment != tra.frames[t_idx[0]].segment:
            continue
        pairs.append((r_idx[0], t_idx[0]))
        residuals.append(0)
    result = _finalize("B_exact_time", pairs, ambiguous)
    result.residuals_ms = residuals
    return result


def align_tolerance(raw: StreamLog, tra: StreamLog, tol_ms: int) -> AlignResult:
    """Method C: per-segment two-pointer greedy matching within ``tol_ms``."""
    pairs: list[tuple[int, int]] = []
    residuals: list[int] = []
    seg_ids = sorted({f.segment for f in raw.frames} & {f.segment for f in tra.frames})
    for seg in seg_ids:
        a = [f for f in raw.frames if f.segment == seg and f.epoch_ms is not None]
        b = [f for f in tra.frames if f.segment == seg and f.epoch_ms is not None]
        i = j = 0
        while i < len(a) and j < len(b):
            d = a[i].epoch_ms - b[j].epoch_ms  # t_raw - t_tra
            if abs(d) <= tol_ms:
                nxt = a[i].epoch_ms - b[j + 1].epoch_ms if j + 1 < len(b) else None
                if nxt is not None and abs(nxt) <= tol_ms and abs(nxt) < abs(d):
                    j += 1  # a strictly closer Tra frame is imminent: skip ahead
                    continue
                pairs.append((a[i].frame_idx, b[j].frame_idx))
                residuals.append(-d)  # t_tra - t_raw
                i += 1
                j += 1
            elif d > 0:
                j += 1  # Tra frame lags behind: drop it
            else:
                i += 1  # Raw frame lags behind: drop it
    result = _finalize(f"C_tol{tol_ms}ms", pairs, 0)
    result.residuals_ms = residuals
    return result


def tolerance_sweep(
    raw: StreamLog, tra: StreamLog, tolerances: list[int]
) -> list[dict]:
    """Run method C over a list of tolerances and record match counts.

    Returns a list of ``{"tol_ms", "n_pairs", "cov_raw", "cov_tra"}`` dicts,
    one per tolerance, in the order given.
    """
    rows = []
    for tol in tolerances:
        res = align_tolerance(raw, tra, tol)
        cov_raw, cov_tra = res.coverage(raw, tra)
        rows.append(
            {
                "tol_ms": tol,
                "n_pairs": res.n_pairs,
                "cov_raw": cov_raw,
                "cov_tra": cov_tra,
            }
        )
    return rows
