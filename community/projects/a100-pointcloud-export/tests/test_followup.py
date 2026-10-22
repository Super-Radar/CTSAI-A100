import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_clouds import export


class AmbiguousTimestampRegression(unittest.TestCase):
    def test_multiple_timestamps_fail_before_any_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "input.csv"
            source.write_text(
                "ObjId,range,angle,speed\nSTART,FrameNb:1\n2026/08/02 00:00:00:000---->\n0,1,0,0\n2026/08/02 00:00:00:001---->\nEND\n"
            )
            output = Path(tmp) / "output"
            with self.assertRaises(ValueError):
                export(source, output)
            self.assertFalse(output.exists())
