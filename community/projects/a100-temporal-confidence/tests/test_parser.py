"""Parser tests: header-width variants 57/58/59/62, trailing comma, END blocks."""

import numpy as np
import pytest

from temporal_ci.parser import parse_raw_target_csv

HEADER5 = "ObjId,range,speed,angle,snr"


def make_csv(ncols_header: int, trailing_comma: bool, n_frames: int = 2) -> str:
    pad = ",".join(f"c{i}" for i in range(5, ncols_header))
    header = HEADER5 + ("," + pad if ncols_header > 5 else "")
    blocks = []
    for t in range(n_frames):
        tail = "," if trailing_comma else ""
        blocks.append(
            "\n".join(
                [
                    f"START,FrameNb:{t},CameraNb:2029,HeSaiNb:0,ARS410Nb:0",
                    f"2026/03/10 15:34:3{t}:090---->",
                    f"heightOffset:0,177312807909{t},0,selfCaliBufCnt:0",
                    f"{10 + t},2.5,3.0,-30.5,20.0" + tail,
                    f"{11 + t},14.2,0.0,64.85,24.5" + tail,
                    f"END,Total:2",
                ]
            )
        )
    return header + "\n" + "\n".join(blocks) + "\n"


@pytest.mark.parametrize("ncols", [57, 58, 59, 62])
@pytest.mark.parametrize("trailing", [False, True])
def test_column_variants_and_positional_values(tmp_path, ncols, trailing):
    p = tmp_path / "log.csv"
    p.write_text(make_csv(ncols, trailing), encoding="utf-8")
    res = parse_raw_target_csv(p)
    assert res.n_frames == 2
    assert res.n_total_mismatch == 0
    f0, f1 = res.frames
    assert f0.frame_nb == 0 and f1.frame_nb == 1
    assert f0.n_targets == 2 and f1.n_targets == 2
    assert f0.ts_text == "2026/03/10 15:34:30:090"
    assert f0.ts_epoch_ms == 1773128079090
    assert f1.ts_epoch_ms == 1773128079091
    assert np.allclose(f0.rng, [2.5, 14.2])
    assert np.allclose(f0.speed, [3.0, 0.0])
    assert np.allclose(f0.angle, [-30.5, 64.85])
    assert np.allclose(f0.snr, [20.0, 24.5])
    assert np.array_equal(f0.obj_id, [10, 11])


def test_skips_blank_banner_and_malformed_rows(tmp_path):
    body = (
        HEADER5 + ",c5,c6,c7\n"
        "START,FrameNb:0,CameraNb:0\n"
        "2026/03/10 15:34:30:090---->\n"
        "heightOffset:0,1773128079090,0\n"
        "not,a,target,row\n"
        "0,2.5,3.0,-30.5,20.0\n"
        "END,Total:1\n"
        "\n"
        "BB Hw:V3.1.9.1,app Fw:V6.4.13.8.0.0,boot Fw:V6.4.13.6.0.0\n"
    )
    p = tmp_path / "log.csv"
    p.write_text(body, encoding="utf-8")
    res = parse_raw_target_csv(p)
    assert res.n_frames == 1
    assert res.frames[0].n_targets == 1
    assert res.n_total_mismatch == 0
    # the malformed in-frame row and the trailing firmware banner line
    assert res.n_unknown_lines == 2


def test_end_total_mismatch_counted(tmp_path):
    body = (
        HEADER5 + "\n"
        "START,FrameNb:0,CameraNb:0\n"
        "0,2.5,3.0,-30.5,20.0\n"
        "END,Total:5\n"
    )
    p = tmp_path / "log.csv"
    p.write_text(body, encoding="utf-8")
    res = parse_raw_target_csv(p)
    assert res.n_frames == 1
    assert res.frames[0].n_targets == 1
    assert res.n_total_mismatch == 1
