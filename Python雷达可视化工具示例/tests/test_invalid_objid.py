"""验证无效目标标识不会被伪装成目标 0。"""

import os
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _support import load_radar_viewer


class InvalidObjIdFallbackTest(unittest.TestCase):
    def test_invalid_objid_is_dropped_before_numeric_defaults(self):
        radar_viewer = load_radar_viewer("radar_viewer_invalid_objid_test")
        csv_text = "\n".join([
            "ObjId,range,speed,angle,RCS,snr,X,Y,FrameNb,Timestamp",
            "not-an-id,4.0,1.0,0,2,10,4,0,12,2026-08-11 10:00:00",
            ",5.0,2.0,0,3,11,5,0,12,2026-08-11 10:00:00",
            "7,6.0,invalid-speed,0,4,12,6,0,12,2026-08-11 10:00:00",
        ])

        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "targets.csv"
            fixture.write_text(csv_text, encoding="utf-8")
            frame = radar_viewer.RadarParser(fixture).get_frame(12)

        self.assertEqual(frame["ObjId"].tolist(), [7])
        self.assertEqual(frame["speed"].tolist(), [0.0])


if __name__ == "__main__":
    unittest.main()
