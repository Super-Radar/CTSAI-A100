"""Tests for stream_align: synthetic dual streams + real-data smoke checks.

Synthetic cases cover the parser (FrameNb resets, cross-day timestamps,
epoch/wall-clock consistency) and alignment method C (known offset, different
output rates, duplicate timestamps, cross-segment abstention, one-to-one and
monotonicity invariants, greedy look-ahead, tolerance abstention). Real-data
tests are skipped automatically when the repository data directory is absent.
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from stream_align.align import (
    align_by_framenb,
    align_exact_time,
    align_tolerance,
    tolerance_sweep,
)
from stream_align.parser import parse_target_csv, wallclock_to_epoch_ms

HEADER = "ObjId,range,speed,angle,snr,Kstate\n"
BASE = datetime(2026, 5, 20, 15, 0, 0)


def build_csv(path, frames):
    """frames: list of (frame_nb, wallclock: datetime, epoch_ms, n_targets)."""
    lines = [HEADER]
    for frame_nb, wc, epoch, n in frames:
        ts = wc.strftime("%Y/%m/%d %H:%M:%S:") + f"{wc.microsecond // 1000:03d}"
        lines.append(f"START,FrameNb:{frame_nb},CameraNb:0,HeSaiNb:0,ARS410Nb:0")
        lines.append(ts + "---->")
        lines.append(f"heightOffset:0,{epoch},0,selfCaliBufCnt:0")
        for k in range(n):
            lines.append(f"{k},253.0,-1.3,1.1,32.3,0")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def consistent_frames(offsets_ms, frame_nbs=None, n_targets=2, base=BASE):
    """Frames built by integer arithmetic only: no float epoch round trip."""
    base_epoch = wallclock_to_epoch_ms(base)
    frames = []
    for i, off in enumerate(offsets_ms):
        wc = base + timedelta(milliseconds=off)
        nb = frame_nbs[i] if frame_nbs else i
        frames.append((nb, wc, base_epoch + off, n_targets))
    return frames


def parse_synth(tmp_path, name, frames, kind="raw"):
    return parse_target_csv(build_csv(tmp_path / name, frames), kind=kind)


# ---------------------------------------------------------------- parser ----

def test_parse_synthetic_fields(tmp_path):
    base = wallclock_to_epoch_ms(datetime(2026, 5, 20, 10, 0, 0, 123000))
    log = parse_synth(tmp_path, "raw.csv", [
        (0, datetime(2026, 5, 20, 10, 0, 0, 123000), base, 3),
        (1, datetime(2026, 5, 20, 10, 0, 0, 250000), base + 127, 0),
        (2, datetime(2026, 5, 20, 10, 0, 0, 400000), base + 277, 1),
    ])
    assert log.n_frames == 3
    assert [f.frame_idx for f in log.frames] == [0, 1, 2]
    assert [f.frame_nb for f in log.frames] == [0, 1, 2]
    assert [f.n_targets for f in log.frames] == [3, 0, 1]
    assert [f.epoch_ms for f in log.frames] == [base, base + 127, base + 277]
    assert all(f.ts_epoch_consistent for f in log.frames)
    assert log.frames[0].wallclock == datetime(2026, 5, 20, 10, 0, 0, 123000)
    assert log.n_segments == 1 and log.n_resets == 0


def test_timestamp_parsing_cross_day(tmp_path):
    # 23:59:59.999 -> next day 00:00:00.010 is an 11 ms step across midnight.
    a = datetime(2026, 5, 22, 23, 59, 59, 999000)
    b = datetime(2026, 5, 23, 0, 0, 0, 10000)
    ea, eb = wallclock_to_epoch_ms(a), wallclock_to_epoch_ms(b)
    log = parse_synth(tmp_path, "cross.csv", [(0, a, ea, 1), (1, b, eb, 1)])
    assert eb - ea == 11
    assert log.frames[0].wallclock.day == 22 and log.frames[1].wallclock.day == 23
    assert log.frames[1].epoch_ms - log.frames[0].epoch_ms == 11
    assert all(f.ts_epoch_consistent for f in log.frames)


def test_epoch_wallclock_consistency_flag(tmp_path):
    wc = datetime(2026, 5, 20, 12, 0, 0, 500000)
    good = wallclock_to_epoch_ms(wc)
    log = parse_synth(tmp_path, "cons.csv", [
        (0, wc, good, 1),
        (1, wc + timedelta(milliseconds=100), good + 100 + 5000, 1),  # +5 s skew
    ])
    assert log.frames[0].ts_epoch_consistent is True
    assert log.frames[1].ts_epoch_consistent is False
    assert log.n_ts_epoch_mismatch == 1


def test_framenb_reset_segmentation(tmp_path):
    log = parse_synth(tmp_path, "reset.csv", consistent_frames(
        [0, 100, 200, 1000, 1100], frame_nbs=[0, 1, 2, 0, 1]))
    assert log.n_frames == 5
    assert log.n_segments == 2
    assert log.n_resets == 1
    assert [f.segment for f in log.frames] == [0, 0, 0, 1, 1]
    # a repeated FrameNb also counts as a reset/rollback
    log2 = parse_synth(tmp_path, "repeat.csv", consistent_frames(
        [0, 100, 200, 300], frame_nbs=[0, 1, 1, 2]))
    assert log2.n_segments == 2 and log2.n_resets == 1


def test_duplicate_epoch_count(tmp_path):
    log = parse_synth(tmp_path, "dup.csv", consistent_frames(
        [0, 0, 100]))
    assert log.duplicate_epoch_count() == 1
    assert log.n_missing_epoch == 0


# ---------------------------------------------------------------- align C ----

def test_method_c_known_offset(tmp_path):
    offs = list(range(0, 2000, 100))
    raw = parse_synth(tmp_path, "raw.csv", consistent_frames(offs))
    tra = parse_synth(tmp_path, "tra.csv",
                      consistent_frames(offs, base=BASE + timedelta(milliseconds=7)), kind="tra")
    res = align_tolerance(raw, tra, 10)
    assert res.n_pairs == 20
    assert res.residuals_ms == [7] * 20
    cov = res.coverage(raw, tra)
    assert cov == (1.0, 1.0)


def test_method_c_abstains_outside_tolerance(tmp_path):
    offs = list(range(0, 1000, 100))
    raw = parse_synth(tmp_path, "raw.csv", consistent_frames(offs))
    tra = parse_synth(tmp_path, "tra.csv",
                      consistent_frames(offs, base=BASE + timedelta(milliseconds=7)), kind="tra")
    assert align_tolerance(raw, tra, 6).n_pairs == 0
    assert align_tolerance(raw, tra, 5).n_pairs == 0
    assert align_tolerance(raw, tra, 7).n_pairs == 10  # |offset| == tol: in


def test_method_c_different_output_rates(tmp_path):
    # Raw at 10 Hz vs Tra at 12.5 Hz: coincidences every lcm(100,80)=400 ms.
    raw = parse_synth(tmp_path, "raw.csv",
                      consistent_frames(list(range(0, 5000, 100))))
    tra = parse_synth(tmp_path, "tra.csv",
                      consistent_frames(list(range(0, 4730, 80))), kind="tra")
    res = align_tolerance(raw, tra, 15)
    assert res.n_pairs == 12  # coincidences t = 0, 400, ..., 4400 ms
    assert all(abs(r) <= 15 for r in res.residuals_ms)
    assert res.residuals_ms == [0] * 12
    cov = res.coverage(raw, tra)
    assert cov[0] == pytest.approx(12 / 50)
    assert cov[1] == pytest.approx(12 / 60)


def test_duplicate_timestamps_one_to_one(tmp_path):
    raw = parse_synth(tmp_path, "raw.csv", consistent_frames(
        [0, 0, 100, 200], frame_nbs=[0, 1, 2, 3]))
    tra = parse_synth(tmp_path, "tra.csv", consistent_frames(
        [0, 100, 100, 200]), kind="tra")
    res = align_tolerance(raw, tra, 0)
    assert res.n_pairs == 3
    used_raw = [p[0] for p in res.pairs]
    used_tra = [p[1] for p in res.pairs]
    assert len(set(used_raw)) == 3 and len(set(used_tra)) == 3  # one-to-one
    assert used_raw == sorted(used_raw) and used_tra == sorted(used_tra)
    assert 1 not in used_raw  # second raw dup abstains
    assert res.residuals_ms == [0, 0, 0]
    # method B abstains on every ambiguous epoch instead of guessing
    resb = align_exact_time(raw, tra)
    assert resb.n_pairs == 1  # only t=200 ms is unique on both sides
    assert resb.ambiguous_keys == 2  # t=0 ms and t=100 ms are duplicated


def test_cross_segment_no_match(tmp_path):
    raw = parse_synth(tmp_path, "raw.csv", consistent_frames(
        [0, 100, 1000, 1100], frame_nbs=[0, 1, 0, 1]))
    tra = parse_synth(tmp_path, "tra.csv", consistent_frames(
        [0, 100, 1000, 1100], frame_nbs=[0, 1, 2, 3]), kind="tra")
    assert raw.n_segments == 2 and tra.n_segments == 1
    t0 = wallclock_to_epoch_ms(BASE)
    resa = align_by_framenb(raw, tra)
    assert resa.n_pairs == 2
    matched_raw_epochs = {raw.frames[p[0]].epoch_ms for p in resa.pairs}
    assert matched_raw_epochs == {t0, t0 + 100}  # segment 2 stays unpaired
    resc = align_tolerance(raw, tra, 50)
    assert resc.n_pairs == 2
    assert {raw.frames[p[0]].epoch_ms for p in resc.pairs} == {t0, t0 + 100}


def test_greedy_lookahead_prefers_nearest(tmp_path):
    raw = parse_synth(tmp_path, "raw.csv", consistent_frames([1000]))
    tra = parse_synth(tmp_path, "tra.csv",
                      consistent_frames([990, 1005]), kind="tra")
    res = align_tolerance(raw, tra, 20)
    assert res.n_pairs == 1
    assert res.residuals_ms == [5]  # picks 1005, not the farther 990


def test_invariants_random_streams(tmp_path):
    import random

    rng = random.Random(20261003)
    for trial in range(20):
        offs_raw = sorted(rng.randrange(0, 5000) for _ in range(40))
        offs_tra = sorted(rng.randrange(0, 5000) for _ in range(45))
        raw = parse_synth(tmp_path, f"raw{trial}.csv",
                          consistent_frames(offs_raw))
        tra = parse_synth(tmp_path, f"tra{trial}.csv",
                          consistent_frames(offs_tra), kind="tra")
        for res, tol in ((align_tolerance(raw, tra, 12), 12),
                         (align_tolerance(raw, tra, 0), 0)):
            used_raw = [p[0] for p in res.pairs]
            used_tra = [p[1] for p in res.pairs]
            assert len(set(used_raw)) == len(used_raw)  # one-to-one ...
            assert len(set(used_tra)) == len(used_tra)
            assert used_raw == sorted(used_raw)  # ... and monotonic
            assert used_tra == sorted(used_tra)
            assert all(abs(r) <= tol for r in res.residuals_ms)
            for r_i, t_i in res.pairs:
                assert raw.frames[r_i].segment == tra.frames[t_i].segment
        resa = align_by_framenb(raw, tra)
        for r_i, t_i in resa.pairs:
            assert raw.frames[r_i].frame_nb == tra.frames[t_i].frame_nb
            assert raw.frames[r_i].segment == tra.frames[t_i].segment


def test_tolerance_sweep_monotone_in_pairs(tmp_path):
    offs = list(range(0, 1000, 50))
    raw = parse_synth(tmp_path, "raw.csv", consistent_frames(offs))
    tra = parse_synth(tmp_path, "tra.csv",
                      consistent_frames(offs, base=BASE + timedelta(milliseconds=3)), kind="tra")
    rows = tolerance_sweep(raw, tra, [0, 2, 3, 5, 10])
    assert [r["tol_ms"] for r in rows] == [0, 2, 3, 5, 10]
    counts = [r["n_pairs"] for r in rows]
    assert counts == sorted(counts)  # never decreases with tol
    assert counts == [0, 0, 20, 20, 20]


# ------------------------------------------------------------ real data ----

def test_parse_real_raw_smoke(real_data_dir):
    if real_data_dir is None:
        pytest.skip("repository data directory not available")
    p = real_data_dir / "2026_05_20_16_24_32_Tatget_CTLRR-300-V2_1_RawTarget.csv"
    log = parse_target_csv(p, kind="raw")
    assert log.n_frames == 2456
    f0 = log.frames[0]
    assert f0.frame_nb == 0 and f0.epoch_ms == 1779265472815
    assert f0.wallclock == datetime(2026, 5, 20, 16, 24, 32, 815000)
    assert log.n_segments == 1 and log.n_resets == 0
    assert log.n_ts_epoch_mismatch == 0
    assert sum(f.n_targets for f in log.frames) > 0
    assert all(f.n_targets >= 0 for f in log.frames)


def test_real_session_exact_pairing(real_data_dir):
    if real_data_dir is None:
        pytest.skip("repository data directory not available")
    raw = parse_target_csv(
        real_data_dir / "2025_11_20_14_39_07_Tatget_CTLRR-300-V2_1_RawTarget.csv",
        kind="raw")
    tra = parse_target_csv(
        real_data_dir / "2025_11_20_14_39_07_Tatget_CTLRR-300-V2_1_TraTarget.csv",
        kind="tra")
    assert (raw.n_frames, tra.n_frames) == (86, 86)
    resb = align_exact_time(raw, tra)
    assert resb.n_pairs == 86
    assert resb.coverage(raw, tra) == (1.0, 1.0)
    assert set(resb.residuals_ms) == {0}
    resa = align_by_framenb(raw, tra)
    assert resa.n_pairs == 86 and max(resa.residuals_ms) == 0


def test_real_flagship_session(real_data_dir):
    if real_data_dir is None:
        pytest.skip("repository data directory not available")
    raw = parse_target_csv(
        real_data_dir / "2026_05_20_16_24_32_Tatget_CTLRR-300-V2_1_RawTarget.csv",
        kind="raw")
    tra = parse_target_csv(
        real_data_dir / "2026_05_20_16_24_32_Tatget_CTLRR-300-V2_1_TraTarget.csv",
        kind="tra")
    assert (raw.n_frames, tra.n_frames) == (2456, 2909)
    resb = align_exact_time(raw, tra)
    assert resb.n_pairs == 2456  # every Raw frame has an exact Tra twin
    cov = resb.coverage(raw, tra)
    assert cov[0] == 1.0
    assert cov[1] == pytest.approx(2456 / 2909)
    assert align_tolerance(raw, tra, 0).n_pairs == 2456
    # FrameNb pairing produces the same *count* but seconds-scale residuals
    resa = align_by_framenb(raw, tra)
    assert resa.n_pairs == 2456
    assert max(abs(r) for r in resa.residuals_ms) > 1000
    # widening the tolerance adds no pairs: extra Tra frames are extra outputs
    rows = tolerance_sweep(raw, tra, [0, 10, 25, 50])
    assert rows[0]["n_pairs"] == rows[-1]["n_pairs"] == 2456
