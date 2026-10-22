import sys
import tempfile
import unittest
from pathlib import Path

from pypcd4 import PointCloud

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_clouds import export


class SchemaTests(unittest.TestCase):
    def test_numeric_rows_before_late_header_are_not_silently_lost(self):
        text = (
            "START,1\n2026/08/02 00:00:00:000---->\n7,2,0,1\nEND\n"
            "ObjId,range,angle,speed\n"
        )
        self.assert_rejected_before_output(text, "before column header")

    def assert_rejected_before_output(self, text, message):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "input.csv", Path(tmp) / "out"
            source.write_text(text)
            with self.assertRaisesRegex(ValueError, message):
                export(source, out)
            self.assertFalse(out.exists())

    def test_duplicate_column_cannot_silently_choose_second_range(self):
        self.assert_rejected_before_output(
            "ObjId,range,angle,speed,range\nSTART,1\n"
            "2026/08/02 00:00:00:000---->\n7,2,0,1,20\nEND\n",
            "Duplicate target column",
        )

    def test_valid_reordered_and_repeated_headers_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, out = Path(tmp) / "input.csv", Path(tmp) / "out"
            source.write_text(
                "ObjId,speed,angle,range\nSTART,1\n"
                "2026/08/02 00:00:00:000---->\n7,-1,90,2\nEND\n"
                "ObjId,range,angle,speed\nSTART,2\n"
                "2026/08/02 00:00:00:100---->\n8,3,0,1\nEND\n"
            )
            report = export(source, out)
            self.assertEqual([f["points"] for f in report["frames"]], [1, 1])
            first = PointCloud.from_path(out / "pcd/000000.pcd").pc_data
            second = PointCloud.from_path(out / "pcd/000001.pcd").pc_data
            self.assertAlmostEqual(float(first["y"][0]), 2)
            self.assertAlmostEqual(float(second["x"][0]), 3)
            self.assertEqual(float(first["radial_speed"][0]), -1)
