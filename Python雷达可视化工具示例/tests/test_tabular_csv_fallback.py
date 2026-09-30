"""验证普通矩形 CSV 能进入备用解析路径。"""

import os
import sys
from pathlib import Path
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _support import load_radar_viewer


class TabularCsvFallbackTest(unittest.TestCase):
    def test_frame_number_can_precede_object_id(self):
        radar_viewer = load_radar_viewer("radar_viewer_tabular_test")
        with tempfile.TemporaryDirectory() as temp_dir:
            fixture = Path(temp_dir) / "targets.csv"
            fixture.write_text(
                "FrameNb,ObjId,range,speed,angle,RCS,snr,X,Y\n"
                "12,7,2.5,0.1,-5,3,20,2.49,-0.22\n",
                encoding="utf-8",
            )
            parser = radar_viewer.RadarParser(fixture)
            frame = parser.get_frame(12)

        self.assertEqual(list(parser.frame_index), [12])
        self.assertEqual(frame["ObjId"].tolist(), [7])
        self.assertEqual(frame["range"].tolist(), [2.5])


if __name__ == "__main__":
    unittest.main()
