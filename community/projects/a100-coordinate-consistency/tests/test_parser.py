"""Parser tests on synthetic CSV strings (no corpus dependency)."""

import math

import pytest

from coord_consistency.parser import parse_string


def test_frames_and_frame_fields(sample_csv):
    log = parse_string(sample_csv, "2026_05_22_13_41_37_Tatget_CTLRR-300-V2_1_TraTarget.csv")
    assert [fr.frame_nb for fr in log.frames] == [0, 5]
    assert log.frames[0].timestamp == "2026/05/22 13:41:37:241"
    assert log.frames[1].timestamp == "2026/05/22 13:41:38:741"
    assert log.frames[0].epoch_ms == 1779428497241
    assert log.frames[1].epoch_ms == 1779428498741
    assert log.frames[0].start_fields["HeSaiNb"] == "0"
    assert log.frames[1].start_fields["HeSaiNb"] == "1"
    assert log.kind == "Tra"


def test_kind_raw_for_raw_filename(sample_csv):
    log = parse_string(sample_csv, "something_RawTarget.csv")
    assert log.kind == "Raw"


def test_row_counts_padding_and_skip(sample_csv):
    log = parse_string(sample_csv)
    # 2 rows in frame 0, 1 row in frame 5; the ",,,,,,,," junk is skipped.
    assert log.n_rows_total == 3
    assert log.n_rows_skipped == 1
    assert [fr.n_rows for fr in log.frames] == [2, 1]
    assert list(log.frame_index()) == [0, 0, 1]


def test_numeric_columns_with_spaces_and_ragged_rows(sample_csv):
    log = parse_string(sample_csv)
    # leading/trailing spaces inside cells and around lines are tolerated
    assert log.column("range").tolist() == pytest.approx([12.14, 15.58, 7.5])
    assert log.column("X").tolist() == pytest.approx([-6.212, 2.2755, 1.30])
    # ragged row is padded -> Y of row 1 is NaN, others parse
    y = log.column("Y")
    assert math.isnan(y[1])
    assert y[0] == pytest.approx(10.425)
    assert y[2] == pytest.approx(-7.38)  # " -7.38" with stray space


def test_trailing_comma_cell(sample_csv):
    log = parse_string(sample_csv)
    assert log.column("DbfMag").tolist()[0] == pytest.approx(85.0)
    assert math.isnan(log.column("DbfMag").tolist()[1])


def test_per_frame_end_does_not_truncate(sample_csv):
    """Corpus variant: every frame closes with its own END,Total:N line."""
    log = parse_string(sample_csv)
    assert [fr.frame_nb for fr in log.frames] == [0, 5]
    assert log.n_rows_total == 3


def test_end_and_footer_stop_collection(sample_csv):
    log = parse_string(sample_csv)
    assert log.n_rows_total == 3  # footer line adds no rows
    strs = log.column_str("ObjId")
    assert strs == ["0", "1", "3"]


def test_missing_column_raises(sample_csv):
    log = parse_string(sample_csv)
    with pytest.raises(KeyError):
        log.column("NotAColumn")


def test_real_corpus_smoke():
    from pathlib import Path

    from coord_consistency.cli import DEFAULT_DATA_DIR
    from coord_consistency.parser import parse_file

    if not DEFAULT_DATA_DIR.is_dir():
        pytest.skip("corpus not available")
    # the largest Tra file is a full multi-frame recording
    biggest = max(DEFAULT_DATA_DIR.glob("*_TraTarget.csv"), key=lambda p: p.stat().st_size)
    log = parse_file(str(biggest))
    assert len(log.frames) > 10
    assert log.n_rows_total > 100
    assert "X" in log.columns and "angle" in log.columns
