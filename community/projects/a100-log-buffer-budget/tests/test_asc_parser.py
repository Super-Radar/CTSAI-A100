"""ASC parser tests on synthetic content plus one real-data regression check."""

from pathlib import Path

import numpy as np
import pytest

from buffer_budget.asc_parser import parse_asc, parse_asc_dir

from conftest import DATA_DIR

SYNTHETIC = "\n".join([
    "date Wed May 20 04:24:32 pm 2026",
    "base hex timestamps absolute",
    "// version 7.2.0",
    "// an unrelated comment line",
    # DLC field (15) disagrees with the 4 data tokens actually present.
    "0.000000 CANFD 1 740 Rx 1 0 d 15 64 02 06 04",
    # same-timestamp cluster, one byte on the line
    "0.017000 CANFD 1 659 Rx 1 0 d 15 64",
    "0.017000 CANFD 2 65A Rx 1 0 d 15 64",
    # classic-CAN keyword form without the BRS/ESI flag pair
    "0.100000 CAN 1 123 Rx d 8 01 02 03 04 05 06 07 08",
    # DLC says 9 but zero data tokens follow -> zero payload bytes
    "0.200000 CANFD 1 740 Rx 1 0 d 9",
    "this line is not a message and must be skipped",
    "",
]) + "\n"


@pytest.fixture()
def synth_log(tmp_path: Path):
    p = tmp_path / "synthetic.asc"
    p.write_text(SYNTHETIC, encoding="utf-8")
    return parse_asc(p)


def test_headers_and_line_accounting(synth_log):
    assert len(synth_log) == 5
    assert synth_log.date_header.startswith("date ")
    assert synth_log.base_header.startswith("base ")
    assert "7.2.0" in synth_log.version_comment
    assert synth_log.n_skipped_lines == 1  # only the free-text line


def test_byte_counts_follow_tokens_not_dlc(synth_log):
    # 4, 1, 1, 8, 0 data-byte tokens regardless of the DLC field values.
    assert synth_log.n_bytes.tolist() == [4, 1, 1, 8, 0]
    assert synth_log.total_bytes == 14


def test_ids_channels_times(synth_log):
    assert synth_log.can_id.tolist() == [0x740, 0x659, 0x65A, 0x123, 0x740]
    assert synth_log.channel.tolist() == [1, 1, 2, 1, 1]
    assert synth_log.time_s.tolist() == [0.0, 0.017, 0.017, 0.1, 0.2]
    assert np.all(np.diff(synth_log.time_s) >= 0)


def test_empty_file_with_headers_only(tmp_path: Path):
    p = tmp_path / "empty.asc"
    p.write_text("date Wed May 20 04:24:32 pm 2026\n"
                 "base hex timestamps absolute\n"
                 "// version 7.2.0\n", encoding="utf-8")
    log = parse_asc(p)
    assert len(log) == 0
    assert log.duration_s == 0.0
    assert np.isnan(log.mean_rate_msgs_per_s)
    assert log.version_comment is not None


def test_single_message_file_roundtrip(tmp_path: Path):
    p = tmp_path / "one.asc"
    p.write_text("// version 7.2.0\n"
                 "1.500000 CANFD 3 18FF50E5 Rx 1 0 d 15 AA BB\n", encoding="utf-8")
    log = parse_asc(p)
    assert len(log) == 1
    assert log.can_id[0] == 0x18FF50E5
    assert log.n_bytes[0] == 2
    assert log.duration_s == 0.0  # single message: zero span


@pytest.mark.skipif(not DATA_DIR.is_dir(), reason="repository data directory not present")
def test_real_data_totals_match_manual_grep():
    logs = parse_asc_dir(DATA_DIR)
    with_msgs = [l for l in logs if len(l) > 0]
    assert len(with_msgs) == 10
    assert sum(len(l) for l in with_msgs) == 201_123
    rep = next(l for l in with_msgs
               if l.path.name == "2026_05_20_16_24_32_Tatget__CanData.asc")
    assert len(rep) == 100_131
    assert "7.2.0" in rep.version_comment
    # Measured quirk: every message line carries exactly one data token more
    # than the CAN-FD DLC code implies (code 15 -> 65 tokens, 13 -> 33, ...).
    # The parser must trust the token count, not the DLC field.
    assert 1 <= rep.n_bytes.max() <= 65
    assert float(rep.n_bytes.mean()) > 0
