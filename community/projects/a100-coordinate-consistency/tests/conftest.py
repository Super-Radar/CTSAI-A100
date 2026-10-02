"""Shared pytest fixtures: make src/ importable."""

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture
def sample_csv() -> str:
    """A miniature target log exercising every parsing rule.

    Frame 0 has two targets (one with stray spaces and a trailing comma,
    one ragged short row) and closes with its own ``END,Total:2`` -- the
    corpus carries one END per frame; frame 5 follows after a blank line
    with one target incl. a near-wrap angle.  Stray spaces, blank lines, a
    junk `,,,,,,,,` row and the firmware footer are all present.
    """
    return "\n".join([
        "ObjId,range,speed,angle,snr,Kstate,OboCarSpeed,heigh,heighAngle,X,Y,RCS,DbfMag",
        "",
        "   ",
        "START,FrameNb:0,CameraNb:0,HeSaiNb:0,ARS410Nb:0",
        "2026/05/22 13:41:37:241---->",
        "heightOffset:0,1779428497241,99940,selfCaliBufCnt:0,radarOffsetV:0.32",
        "0, 12.14 , 0, 30.78, 35.5, 0, 0, 0, -3.25, -6.212, 10.425, -14.5, 85,",
        "1,15.58,-0.12,-8.4,0,0,0,0,0,2.2755",
        ",,,,,,,,",
        "END,Total:2",
        "",
        "START,FrameNb:5,CameraNb:0,HeSaiNb:1,ARS410Nb:0",
        "2026/05/22 13:41:38:741---->",
        "heightOffset:0,1779428498741,100080,selfCaliBufCnt:0",
        "3,7.5,0.5,-170.0,20,1,0,0,0,1.30, -7.38, 3.0, 60",
        "END,Total:3",
        "BB Hw:V3.1.9.1,app Fw:V6.4.23.11.0.0",
        "",
    ])
