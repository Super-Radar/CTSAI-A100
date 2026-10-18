"""Test suite for a100-filter-selection-bias.

Covers: the three header variants (58/59/62 columns) of the RawTarget frame
format, END/Total handling, metadata/timestamp lines, missing/NaN exclusion,
filter monotonicity (tightening => survivor subset), all-off identity,
analytic retention of a known synthetic mixture, stratified-count
conservation, Jaccard algebra, joint-grid consistency and quantile
monotonicity.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filter_bias.filters import FilterSpec, single_dim_spec
from filter_bias.parser import PointCloud, parse_text
from filter_bias.sweep import (
    deletion_mask,
    distribution_shift,
    joint_grid,
    pairwise_jaccard,
    retention,
    stratified_retention,
    sweep_dimension,
)

# ---------------------------------------------------------------------------
# synthetic helpers
# ---------------------------------------------------------------------------


def build_mixture() -> PointCloud:
    """Deterministic static/moving/background mixture with per-field grids.

    Within each component the fields are laid out as a full meshgrid, so the
    fraction of a component surviving any per-field threshold product is an
    exact rational number -- an analytic oracle independent of the code
    under test.
    """
    components = (
        # (ranges, speeds, angles, snrs)
        ([5.0, 10.0, 15.0, 20.0], [-0.2, -0.1, 0.0, 0.1, 0.2], [-10.0, 0.0, 10.0], [15.0, 20.0, 30.0, 40.0]),
        ([30.0, 46.0, 64.0, 80.0], [6.8, 7.2, 7.6, 8.0], [-40.0, 0.0, 40.0], [8.0, 19.0, 30.0]),
        ([1.0, 38.25, 75.5, 112.75, 150.0], [-20.0, -12.0, -4.0, 4.0, 12.0, 20.0], [-75.0, 0.0, 75.0], [0.5, 8.0, 16.5, 25.0]),
    )
    parts = {k: [] for k in ("range_m", "speed_ms", "angle_deg", "snr_db")}
    for ranges, speeds, angles, snrs in components:
        r, s, a, n = np.meshgrid(ranges, speeds, angles, snrs, indexing="ij")
        parts["range_m"].append(r.ravel())
        parts["speed_ms"].append(s.ravel())
        parts["angle_deg"].append(a.ravel())
        parts["snr_db"].append(n.ravel())
    flat = {k: np.concatenate(v) for k, v in parts.items()}
    m = flat["range_m"].shape[0]
    return PointCloud(
        file_id=np.zeros(m, dtype=np.int32),
        frame=np.zeros(m, dtype=np.int32),
        obj_id=np.arange(m, dtype=np.int32),
        file_names=("synthetic.csv",),
        **flat,
    )


# analytic retention of build_mixture() under specific thresholds:
#   static  : 240 pts, moving: 144 pts, background: 360 pts  (N = 744)
N_TOTAL = 744


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------

HEADER5 = "ObjId,range,speed,angle,snr"


def make_log(ncols: int, frames_targets: list) -> str:
    pad_header = ",".join(f"extra{i}" for i in range(5, ncols))
    pad_row = ",".join("0" for _ in range(5, ncols))
    lines = [HEADER5 + ("," + pad_header if ncols > 5 else "")]
    for frame_nb, targets in enumerate(frames_targets):
        lines.append(f"START,FrameNb:{frame_nb},CameraNb:2029,HeSaiNb:0,ARS410Nb:0")
        lines.append("2026/03/10 15:34:39:090---->")
        lines.append("heightOffset:0,1773128079090,0,selfCaliBufCnt:0")
        for t in targets:
            lines.append(",".join(str(x) for x in t) + ("," + pad_row if ncols > 5 else ""))
        lines.append(f"END,Total:{len(targets)}")
    return "\n".join(lines) + "\n"


TARGETS_FRAME0 = [(0, 2.31, 0.0, -22.45, 36.5), (1, 14.2, 1.5, 64.85, 24.5)]
TARGETS_FRAME1 = [(0, 9.9, -0.25, 11.0, 12.0)]


def test_parser_header_variants_identical():
    logs = {n: make_log(n, [TARGETS_FRAME0, TARGETS_FRAME1]) for n in (57, 58, 59, 62)}
    parsed = {n: parse_text(text) for n, text in logs.items()}
    for n in (58, 59, 62):
        assert parsed[n][0] == parsed[57][0]  # same target rows
        assert parsed[n][1] == parsed[57][1]  # same frame records
    assert parsed[57][2] == 57 and parsed[58][2] == 58
    assert parsed[59][2] == 59 and parsed[62][2] == 62


def test_parser_frames_and_totals():
    rows, frames, _ = parse_text(make_log(62, [TARGETS_FRAME0, TARGETS_FRAME1]))
    assert [f[0] for f in frames] == [0, 1]
    assert [f[1] for f in frames] == [len(TARGETS_FRAME0), len(TARGETS_FRAME1)]  # END,Total:N
    assert [r[0] for r in rows] == [0, 0, 1]  # frame attribution of each target
    assert len(rows) == 3


def test_parser_skips_timestamp_and_metadata():
    rows, frames, _ = parse_text(make_log(58, [TARGETS_FRAME0]))
    assert len(rows) == len(TARGETS_FRAME0)
    assert all(isinstance(r[2], float) for r in rows)
    # no line of the timestamp/metadata block may appear as a target row
    assert all(r[2] != 1773128079090 for r in rows)  # epoch-ms field of the meta line


def test_parser_excludes_missing_and_nan_rows():
    bad = [
        "7,3.0,,12.0,20.0" + ",0" * 53,   # empty speed
        "8,4.0,nan,12.0,20.0" + ",0" * 53,  # NaN speed
        "9,inf,1.0,12.0,20.0" + ",0" * 53,  # inf range
    ]
    text = make_log(58, [TARGETS_FRAME0])
    text = text.replace(
        "END,Total:2",
        "\n".join(bad) + "\nEND,Total:5",
    )
    rows, frames, _ = parse_text(text)
    assert len(rows) == 2  # malformed rows excluded, never zero-filled
    assert frames[0][1] == 5  # declared Total (parser reports it; exclusions are our policy)
    assert all(0.0 < r[2] < 100.0 for r in rows)  # no zero/NaN leakage into fields


# ---------------------------------------------------------------------------
# filters: identity + monotone inclusion
# ---------------------------------------------------------------------------


def test_filter_all_off_is_identity():
    c = build_mixture()
    keep = FilterSpec().kept_mask(c)
    assert keep.dtype == np.bool_
    assert np.all(keep)
    assert retention(keep) == 1.0


@pytest.mark.parametrize(
    "loose,tight",
    [
        (dict(range_max=80.0), dict(range_max=40.0)),
        (dict(angle_max=60.0), dict(angle_max=10.0)),
        (dict(snr_min=5.0), dict(snr_min=20.0)),
        (dict(speed_max=30.0), dict(speed_max=5.0)),
        (dict(speed_min=0.05), dict(speed_min=0.5)),
        (dict(range_min=1.0), dict(range_min=30.0)),
    ],
)
def test_tightening_single_threshold_yields_subset(loose, tight):
    c = build_mixture()
    k_loose = FilterSpec(**loose).kept_mask(c)
    k_tight = FilterSpec(**tight).kept_mask(c)
    assert np.all(~k_tight | k_loose)  # survivor(tight) subset of survivor(loose)
    assert np.count_nonzero(k_tight) < np.count_nonzero(k_loose)  # and strictly smaller


def test_single_dim_spec_unknown_dimension():
    with pytest.raises(KeyError):
        single_dim_spec("nope", 1.0)


# ---------------------------------------------------------------------------
# synthetic benchmark: analytic retention + stratification
# ---------------------------------------------------------------------------


def test_mixture_analytic_retention():
    c = build_mixture()
    assert len(c) == N_TOTAL
    spec = FilterSpec(speed_min=0.15, snr_min=10.0, range_max=60.0, angle_max=30.0)
    # static : (2/5 speed) * (1 range) * (1 angle) * (1 snr)
    # moving : (1 speed) * (2/4 range) * (1/3 angle: |±40|>30) * (2/3 snr)
    # bg     : (1 speed) * (2/5 range) * (1/3 angle: |±75|>30) * (2/4 snr)
    expected = (240 * (2 / 5) + 144 * (2 / 4 * 1 / 3 * 2 / 3) + 360 * (2 / 5 * 1 / 3 * 2 / 4)) / N_TOTAL
    assert retention(spec.kept_mask(c)) == pytest.approx(expected, abs=1e-12)
    assert np.count_nonzero(spec.kept_mask(c)) == 136  # 96 + 16 + 24


def test_stratified_counts_sum_to_overall():
    c = build_mixture()
    spec = FilterSpec(speed_min=0.15, snr_min=10.0)  # no range cut -> several bands kept
    keep = spec.kept_mask(c)
    for stratum in ("range_band", "angle_sector", "speed_bin"):
        bands = stratified_retention(c, keep, stratum)
        assert sum(b["n_total"] for b in bands) == len(c)
        assert sum(b["n_kept"] for b in bands) == np.count_nonzero(keep)


def test_stratified_band_counts_analytic():
    c = build_mixture()
    keep = FilterSpec(speed_min=0.15, snr_min=10.0).kept_mask(c)
    bands = {b["band"]: b for b in stratified_retention(c, keep, "range_band")}
    # hand-derived per-band kept counts (meshgrid oracle)
    assert bands["[0,50)"]["n_kept"] == 216  # 96 static + 48 moving + 72 bg
    assert bands["[50,100)"]["n_kept"] == 84  # 48 moving + 36 bg
    assert bands["[100,150)"]["n_kept"] == 36
    assert bands["[150,200)"]["n_kept"] == 36
    assert bands["[200,+inf)"]["n_kept"] == 0
    assert bands["[0,50)"]["n_total"] == 456


def test_nearzero_fraction_static_cluster():
    c = build_mixture()
    frac = float(np.count_nonzero(np.abs(c.speed_ms) < 0.15)) / len(c)
    assert frac == pytest.approx(144 / N_TOTAL, abs=1e-12)  # 3/5 of static cluster


# ---------------------------------------------------------------------------
# sweep / jaccard / joint grid / distribution shift
# ---------------------------------------------------------------------------


def test_sweep_dimension_rows_and_monotonicity():
    c = build_mixture()
    rows = sweep_dimension(c, "speed_min", [0.0, 0.05, 0.15, 0.5], strata=("range_band",))
    all_rows = [r for r in rows if r["stratum"] == "ALL"]
    assert [r["retention"] for r in all_rows] == sorted(
        [r["retention"] for r in all_rows], reverse=True
    )  # retention decreases as v_min tightens
    assert all_rows[0]["retention"] == 1.0  # v_min = 0 keeps everything
    assert len(rows) == 4 * (1 + 5)  # ALL + 5 range bands per threshold


def test_jaccard_algebra():
    c = build_mixture()
    d_far = c.range_m > 60
    d_near = c.range_m < 10
    same = { "far_a": d_far, "far_b": d_far }
    rec = pairwise_jaccard(same)[0]
    assert rec["jaccard"] == 1.0  # identical sets
    assert rec["containment_a"] == 1.0 and rec["containment_b"] == 1.0
    dis = {"far": d_far, "near": d_near}
    rec = pairwise_jaccard(dis)[0]
    assert rec["jaccard"] == 0.0  # disjoint sets
    assert rec["containment_a"] == 0.0 and rec["containment_b"] == 0.0
    ab = pairwise_jaccard({"a": deletion_mask(c, FilterSpec(range_max=60)),
                            "b": deletion_mask(c, FilterSpec(snr_min=15))})[0]
    ba = pairwise_jaccard({"b": deletion_mask(c, FilterSpec(snr_min=15)),
                            "a": deletion_mask(c, FilterSpec(range_max=60))})[0]
    assert ab["jaccard"] == pytest.approx(ba["jaccard"])  # symmetric
    assert ab["intersection"] <= min(ab["n_del_a"], ab["n_del_b"])
    assert ab["union"] >= max(ab["n_del_a"], ab["n_del_b"])
    # containment is the one-sided view: intersection over each deletion set
    assert ab["containment_a"] == pytest.approx(ab["intersection"] / ab["n_del_a"])
    assert ab["containment_b"] == pytest.approx(ab["intersection"] / ab["n_del_b"])
    # a small deletion set fully inside a large one: low Jaccard, full containment
    d_small = c.range_m > 149  # subset of the >60 deletion set
    rec = pairwise_jaccard({"big": d_far, "small": d_small})[0]
    assert rec["containment_b"] == 1.0
    assert rec["jaccard"] < 1.0


def test_joint_grid_matches_direct_and_analytic():
    c = build_mixture()
    rows = joint_grid(c, "speed_min", [0.1, 0.5], "snr_min", [10.0])
    direct = {
        (0.1, 10.0): FilterSpec(speed_min=0.1, snr_min=10.0).kept_mask(c),
        (0.5, 10.0): FilterSpec(speed_min=0.5, snr_min=10.0).kept_mask(c),
    }
    for r in rows:
        key = (r["value_a"], r["value_b"])
        assert r["retention"] == pytest.approx(retention(direct[key]), abs=1e-15)
        assert r["interaction"] == pytest.approx(r["retention"] - r["retention_indep"], abs=1e-15)
    # analytic: v=0.1,s=10 -> (240*4/5 + 144*2/3 + 360*1/2)/744 = 468/744
    assert rows[0]["retention"] == pytest.approx(468 / 744, abs=1e-12)
    # analytic: v=0.5,s=10 -> static fully removed -> (0 + 96 + 180)/744 = 276/744
    assert rows[1]["retention"] == pytest.approx(276 / 744, abs=1e-12)


def test_distribution_shift_quantiles_monotone():
    c = build_mixture()
    for spec in (FilterSpec(speed_min=0.5), FilterSpec(snr_min=15.0, range_max=100.0)):
        rows = distribution_shift(c, spec)
        for field in ("range", "angle", "snr", "speed"):
            qs = [r for r in rows if r["field"] == field and r["stat"] != "mean"]
            assert [r["stat"] for r in qs] == ["q05", "q10", "q25", "q50", "q75", "q90", "q95"]
            before = [r["before"] for r in qs]
            after = [r["after"] for r in qs]
            assert before == sorted(before)  # quantiles are non-decreasing
            assert after == sorted(after)
            assert len(rows) == 4 * (1 + 7)  # 4 fields x (mean + 7 quantiles)
