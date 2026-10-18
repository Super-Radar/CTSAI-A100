"""Tests for the log-decimation fidelity study.

Covers: parser dialect variants, k=1 identity, weight conservation under the
merge rule, sampler determinism and phase-enumeration completeness, analytic
short-response omission, time- vs count-weighting on irregular gaps, and the
real repository sessions.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from decimation.fidelity import (
    CellGrid,
    analytic_missing_phases,
    count_statistics,
    frame_time_weights,
    merged_time_weights,
    short_response_metrics,
    synthetic_irregular_session,
    synthetic_pulse_session,
    tv_l1,
    l2_distance,
)
from decimation.parser import parse_target_log
from decimation.samplers import (
    byte_budget_k,
    enumerate_uniform_phases,
    stratified_keep_indices,
    uniform_keep_indices,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DATA1 = (
    REPO_ROOT / "RadarTools" / "RadarTools_Release" / "data"
    / "2026_05_20_16_24_32_Tatget_CTLRR-300-V2_1_RawTarget.csv"
)
DATA2 = (
    REPO_ROOT / "Python雷达可视化工具示例"
    / "负30度_Tatget_2026_03_10_15_34_39_CTLRR-300-V2_1_RawTarget.csv"
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _synthetic_csv(header_cols: int, trailing_comma: bool, n_frames=3) -> str:
    """Build a small RawTarget-style CSV with a chosen header width."""
    header = ",".join(
        ["ObjId", "range", "speed", "angle", "snr"]
        + [f"c{i}" for i in range(header_cols - 5)]
    )
    lines = [header]
    epoch = 1_700_000_000_000
    for f in range(n_frames):
        lines.append(f"START,FrameNb:{f},CameraNb:0,HeSaiNb:0,ARS410Nb:0")
        lines.append(f"2026/05/20 16:24:3{f}:815---->")
        lines.append(f"heightOffset:0,{epoch + f * 50},3917020,selfCaliBufCnt:0")
        for t in range(2):
            tail = "," if trailing_comma else ""
            lines.append(
                f"{t},{3.5 + t + 0.1 * f},0,{-3.2 + t},{36 + t},0,0,0{tail}"
            )
        lines.append(f"END,Total:2")
    lines.append("BB Hw:V3.1.9.1,app Fw:V6.4.13.8.0.0,Can CH:3.4.5.18")
    return "\n".join(lines) + "\n"


def _make_grid(r_max=30.0):
    return CellGrid(dr=0.5, dtheta=2.0, r_max=r_max)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("cols,trailing", [(57, True), (58, False), (59, True), (62, True)])
def test_parser_header_variants(tmp_path, cols, trailing):
    p = tmp_path / f"log_{cols}_{trailing}.csv"
    p.write_text(_synthetic_csv(cols, trailing), encoding="utf-8")
    log = parse_target_log(p)
    assert log.header_columns == cols
    assert log.n_frames == 3
    assert log.trailing_lines and log.trailing_lines[0].startswith("BB Hw:")
    # positional reads: frame 0 targets (3.5,-3.2,36) and (4.5,-2.2,37)
    np.testing.assert_allclose(log.frames[0].targets[0], [3.5, -3.2, 36.0])
    np.testing.assert_allclose(log.frames[0].targets[1], [4.5, -2.2, 37.0])
    assert log.frames[1].epoch_ms - log.frames[0].epoch_ms == 50
    assert log.check_times_monotonic()


def test_parser_end_total_mismatch_raises(tmp_path):
    text = _synthetic_csv(57, False).replace("END,Total:2", "END,Total:5")
    p = tmp_path / "bad_total.csv"
    p.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="END,Total"):
        parse_target_log(p)


def test_parser_empty_frame(tmp_path):
    """A frame with zero targets (START immediately followed by END)."""
    text = (
        "ObjId,range,speed,angle,snr\n"
        "START,FrameNb:0,CameraNb:0\n"
        "2026/05/20 16:24:32:815---->\n"
        "heightOffset:0,1700000000000,1\n"
        "END,Total:0\n"
        "START,FrameNb:1,CameraNb:0\n"
        "2026/05/20 16:24:32:900---->\n"
        "heightOffset:0,1700000000085,1\n"
        "0,5.5,0,10.0,40\n"
        "END,Total:1\n"
    )
    p = tmp_path / "empty_frame.csv"
    p.write_text(text, encoding="utf-8")
    log = parse_target_log(p)
    assert [f.n_targets for f in log.frames] == [0, 1]
    assert log.duration_s == pytest.approx(0.085)


@pytest.mark.skipif(not DATA1.exists(), reason="primary session not in checkout")
def test_parser_real_primary():
    log = parse_target_log(DATA1)
    assert log.n_frames == 2456
    assert log.header_columns == 62
    assert int(log.counts.sum()) == 78335
    assert log.duration_s == pytest.approx(145.294, abs=0.01)
    assert log.check_times_monotonic()


@pytest.mark.skipif(not DATA2.exists(), reason="secondary session not in checkout")
def test_parser_real_secondary():
    log = parse_target_log(DATA2)
    assert log.n_frames == 2463
    assert log.header_columns == 57
    assert int(log.counts.sum()) == 35803
    assert any(l.startswith("BB Hw:") for l in log.trailing_lines)


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------
def test_uniform_k1_identity_and_enumeration():
    n = 100
    assert np.array_equal(uniform_keep_indices(n, 1, 0), np.arange(n))
    for k in range(2, 17):
        phases = enumerate_uniform_phases(n, k)
        assert len(phases) == k  # complete phase enumeration
        union = np.sort(np.concatenate(list(phases.values())))
        assert np.array_equal(union, np.arange(n))  # covers every frame exactly once
        sizes = {len(v) for v in phases.values()}
        assert max(sizes) - min(sizes) <= 1  # balanced residue classes


def test_stratified_determinism_and_coverage():
    times = np.arange(1000, dtype=np.int64) * 53 + 1_700_000_000_000
    a = stratified_keep_indices(times, 200, seed=20261003)
    b = stratified_keep_indices(times, 200, seed=20261003)
    c = stratified_keep_indices(times, 200, seed=1)
    assert np.array_equal(a, b)  # same seed -> identical draw
    assert not np.array_equal(a, c)  # different seed -> different draw
    assert len(a) == 200 and len(np.unique(a)) == 200
    # one frame per duration stratum: at most 1 kept frame per 1/200 of duration
    span = times[-1] - times[0]
    edges = np.linspace(times[0], times[-1], 201)
    ids = np.digitize(times[a], edges[1:-1])
    assert len(np.unique(ids)) == 200


def test_byte_budget_k():
    assert byte_budget_k(1.0) == 1
    assert byte_budget_k(0.5) == 1
    assert byte_budget_k(2.0) == 2
    assert byte_budget_k(7.0000001) == 8
    assert byte_budget_k(15.2) == 16


# ---------------------------------------------------------------------------
# weights and conservation
# ---------------------------------------------------------------------------
def test_frame_time_weights_sum():
    rng = np.random.default_rng(7)
    times = np.cumsum(rng.integers(10, 200, size=500)) + 1_700_000_000_000
    w = frame_time_weights(times)
    assert w.sum() == pytest.approx(times[-1] - times[0], abs=1e-6)
    assert np.all(w >= 0)


def test_weight_conservation_under_sampling():
    times = np.cumsum(
        np.random.default_rng(11).integers(20, 200, size=1000)
    ) + 1_700_000_000_000
    duration = times[-1] - times[0]
    for keep in (
        uniform_keep_indices(1000, 3, 1),
        uniform_keep_indices(1000, 7, 5),
        stratified_keep_indices(times, 250, seed=20261003),
        np.array([0, 999]),  # extreme: only first and last frame kept
    ):
        wm = merged_time_weights(times, keep)
        assert wm.sum() == pytest.approx(duration, abs=1e-6)
    # hand-checked case: n=5, times [0,10,25,40,50], k=2, phase=0 ->
    # kept {0,2,4}. Full ownership weights [5,12.5,15,12.5,5]; nearest-kept
    # index assignment gives kept0<-{0,1}, kept1<-{2,3}, kept2<-{4}
    t5 = np.array([0, 10, 25, 40, 50], dtype=np.int64)
    wm5 = merged_time_weights(t5, np.array([0, 2, 4]))
    assert wm5.sum() == 50
    np.testing.assert_allclose(wm5, [17.5, 27.5, 5.0])


def test_k1_identity_metrics():
    frames, times = synthetic_pulse_session()
    grid = _make_grid()
    w = frame_time_weights(times)
    dens, _ = grid.histogram(frames, w)
    keep = uniform_keep_indices(len(frames), 1, 0)
    wm = merged_time_weights(times, keep)
    dens_k1, _ = grid.histogram([frames[j] for j in keep], wm)
    assert tv_l1(dens, dens_k1) == pytest.approx(0.0, abs=1e-12)
    exp_full = grid.exposure(frames, w / 1000.0)
    exp_k1 = grid.exposure([frames[j] for j in keep], wm / 1000.0)
    sr = short_response_metrics(exp_full, exp_k1, T=1.0)
    assert sr["retention_rate"] == pytest.approx(1.0)
    assert sr["n_missed"] == 0


# ---------------------------------------------------------------------------
# short responses: analytic omission
# ---------------------------------------------------------------------------
def test_analytic_missing_phases_matches_enumeration():
    for start in (0, 7, 23, 40):
        for L in (1, 2, 3, 5):
            for k in range(2, 10):
                predicted = set(analytic_missing_phases(start, L, k))
                simulated = set()
                for phi in range(k):
                    keep = set(int(x) for x in uniform_keep_indices(100, k, phi))
                    if not any(start <= j < start + L for j in keep):
                        simulated.add(phi)
                assert predicted == simulated, (start, L, k)
                if L >= k:
                    assert not predicted  # a response spanning >= k frames cannot be fully missed


def test_pulse_train_session_omission():
    """Pulse of L frames at start=37: phases predicted to miss actually miss."""
    start, L, k = 37, 3, 5
    frames, times = synthetic_pulse_session(pulses=((start, L, (6.25, 15.0)),))
    grid = _make_grid()
    # locate the pulse cell by its polar bin (only cell near (6.25, 15.0))
    ir = int((6.25 - grid.r_edges[0]) / grid.dr)
    it = int((15.0 - grid.theta_edges[0]) / grid.dtheta)
    for phi in range(k):
        keep = uniform_keep_indices(len(frames), k, phi)
        wm = merged_time_weights(times, keep)
        exp_s = grid.exposure([frames[j] for j in keep], wm / 1000.0)
        observed = exp_s[ir, it] > 0
        predicted_miss = phi in set(analytic_missing_phases(start, L, k))
        assert observed == (not predicted_miss), phi
    # and at least one phase misses while another retains
    assert set(analytic_missing_phases(start, L, k))
    assert len(set(analytic_missing_phases(start, L, k))) < k


# ---------------------------------------------------------------------------
# time weighting vs count weighting
# ---------------------------------------------------------------------------
def test_time_weighted_vs_count_weighted_irregular():
    frames, times = synthetic_irregular_session()
    grid = _make_grid()
    dens_t, _ = grid.histogram(frames, frame_time_weights(times))
    dens_c, _ = grid.histogram(frames, np.ones(len(frames)))
    ir_a = int((6.0 - grid.r_edges[0]) / grid.dr)
    it_a = int((10.0 - grid.theta_edges[0]) / grid.dtheta)
    ir_b = int((15.0 - grid.r_edges[0]) / grid.dr)
    it_b = int((-20.0 - grid.theta_edges[0]) / grid.dtheta)
    a_t, b_t = dens_t[ir_a, it_a], dens_t[ir_b, it_b]
    a_c, b_c = dens_c[ir_a, it_a], dens_c[ir_b, it_b]
    # equal-duration halves: time weighting equalises the two cells (~0.5 each)
    assert a_t == pytest.approx(0.5, abs=0.05)
    assert b_t == pytest.approx(0.5, abs=0.05)
    # count weighting over-weights the dense half by ~ the density ratio (10x)
    assert b_c / a_c == pytest.approx(10.0, abs=1.5)
    assert abs(a_c - 21 / 222) < 0.02 and abs(b_c - 201 / 222) < 0.02


def test_count_statistics_and_l1_l2_bounds():
    frames, times = synthetic_pulse_session(n_frames=50, pulses=())
    st = count_statistics(frames)
    assert st["mean"] == pytest.approx(4.0)  # 4 background cells, no pulses
    # identical histograms -> 0; disjoint support -> L1 = 1, L2 = sqrt(2)
    p = np.array([1.0, 0.0])
    q = np.array([0.0, 1.0])
    assert tv_l1(p, p) == 0.0 and l2_distance(p, p) == 0.0
    assert tv_l1(p, q) == pytest.approx(1.0)
    assert l2_distance(p, q) == pytest.approx(math.sqrt(2))
    # decimating a pulse-free session leaves the (uniform) counts intact
    full = count_statistics(frames)
    kept = [frames[j] for j in uniform_keep_indices(50, 5, 2)]
    part = count_statistics(kept)
    assert full["mean"] == pytest.approx(part["mean"])
    assert full["p90"] == pytest.approx(part["p90"])


@pytest.mark.skipif(not DATA1.exists(), reason="primary session not in checkout")
def test_real_data_weight_conservation_and_metrics():
    log = parse_target_log(DATA1)
    times = log.times_ms
    keep = uniform_keep_indices(log.n_frames, 8, 3)
    wm = merged_time_weights(times, keep)
    # conservation: merged exposure == full recording duration
    assert wm.sum() == pytest.approx(times[-1] - times[0], abs=1e-3)
    grid = _make_grid(r_max=30.0)
    dens_full, _ = grid.histogram(log.frames, frame_time_weights(times))
    dens_s, _ = grid.histogram([log.frames[j] for j in keep], wm)
    l1 = tv_l1(dens_full, dens_s)
    assert 0.0 <= l1 <= 1.0
    assert l1 > 0.0  # decimation must actually lose something
