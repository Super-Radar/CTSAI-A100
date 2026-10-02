"""Census tests: per-column counts, junk-value exclusion, aggregation."""

import pytest

from coord_consistency.census import aggregate_rows, corpus_census, file_census
from coord_consistency.parser import parse_string

HEADER = ("ObjId,range,speed,angle,snr,Kstate,OboCarSpeed,heigh,heighAngle,"
          "X,Y,RCS,DbfMag")


def _log_with_x(x_values):
    rows = "\n".join(f"{i},5,0,10,20,0,0,0,0,{xv},1.0,-10,60" +
                     ("," if i % 2 else "")
                     for i, xv in enumerate(x_values))
    text = "\n".join([HEADER,
                      "START,FrameNb:0,CameraNb:0",
                      "2026/05/22 13:41:37:241---->",
                      "heightOffset:0,1779428497241,99940",
                      rows])
    return parse_string(text, "x_TraTarget.csv")


def test_census_counts_on_synthetic(sample_csv):
    log = parse_string(sample_csv)
    rows = {rec["column"]: rec for rec in file_census(log)}
    assert rows["range"]["rows"] == 3
    assert rows["range"]["nonmissing"] == 3
    assert rows["range"]["nonzero_effective"] == 3
    # Y: one NaN from the ragged row -> 2/3 parseable
    assert rows["Y"]["nonmissing"] == 2
    assert rows["Y"]["fill_rate"] == pytest.approx(2 / 3)
    assert rows["Y"]["effective_fill_rate"] == pytest.approx(2 / 3)
    assert rows["ObjId"]["rows"] == 3


def test_census_excludes_float_junk():
    log = _log_with_x(["0", "1.71472e-31", "-3.25", ""])
    rows = {rec["column"]: rec for rec in file_census(log)}
    x = rows["X"]
    assert x["rows"] == 4
    assert x["nonmissing"] == 3          # empty cell counts as missing
    assert x["nonzero_effective"] == 1   # 0 and 1.7e-31 are "no signal"
    assert x["fill_rate"] == pytest.approx(0.75)
    assert x["effective_fill_rate"] == pytest.approx(0.25)


def test_aggregate_rows_pool_counts():
    log_a = _log_with_x(["0", "-3.25"])
    log_b = _log_with_x(["7.5"])
    per_file = file_census(log_a) + file_census(log_b)
    agg = {(r["kind"], r["column"]): r for r in aggregate_rows(per_file)}
    x = agg[("Tra", "X")]
    assert x["file"] == "AGGREGATE"
    assert x["rows"] == 3
    assert x["nonmissing"] == 3
    assert x["nonzero_effective"] == 2
    assert x["effective_fill_rate"] == pytest.approx(2 / 3)
    # every focus column present in both logs pooled
    assert ("Tra", "DbfMag") in agg and ("Tra", "RCS") in agg


def test_corpus_census_contains_aggregate_rows():
    log = _log_with_x(["1.0", "2.0"])
    rows = corpus_census([log])
    files = {r["file"] for r in rows}
    assert log.filename in files and "AGGREGATE" in files
    agg_rows = [r for r in rows if r["file"] == "AGGREGATE"]
    assert len(agg_rows) >= 8  # one per focus column
