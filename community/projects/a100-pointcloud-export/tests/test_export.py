import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from pypcd4 import PointCloud
from rosbags.highlevel import AnyReader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from export_clouds import DTYPE, cloud, export


class ExportTests(unittest.TestCase):
    def fixture(
        self, tmp, stamps=("2026/08/02 23:59:59:900", "2026/08/03 00:00:00:100")
    ):
        p = Path(tmp) / "capture.csv"
        p.write_text(
            "ObjId,range,angle,speed\nSTART,FrameNb:7\n" + stamps[0] + "---->\n"
            "4294967295,2,90,-1\nEND\nSTART,FrameNb:7\n" + stamps[1] + "---->\nEND\n"
        )
        return p

    def test_independent_pcd_and_ros_reader(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "out"
            report = export(self.fixture(tmp), out)
            self.assertEqual(len(report["frames"]), 2)
            pc = PointCloud.from_path(out / "pcd/000000.pcd")
            self.assertEqual(int(pc.pc_data["object_id"][0]), 2**32 - 1)
            self.assertAlmostEqual(float(pc.pc_data["y"][0]), 2)
            self.assertEqual(
                len(PointCloud.from_path(out / "pcd/000001.pcd").pc_data), 0
            )
            with AnyReader([out / "bag"]) as reader:
                items = []
                for conn, t, data in reader.messages(
                    connections=[
                        c for c in reader.connections if c.topic == "/a100/points"
                    ]
                ):
                    msg = reader.deserialize(data, conn.msgtype)
                    items.append((t, msg))
            self.assertEqual([x[0] for x in items], [1000000000, 1200000000])
            self.assertEqual(items[1][1].width, 0)
            msg = items[0][1]
            self.assertFalse(msg.is_bigendian)
            self.assertEqual(msg.point_step, 28)
            decoded = np.frombuffer(msg.data.tobytes(), dtype=DTYPE)
            np.testing.assert_array_equal(decoded, pc.pc_data)

    def test_coordinate_axes(self):
        points = cloud([(2, 0, 1, 0), (2, 90, -1, 1)])
        np.testing.assert_allclose(points["x"], [2, 0], atol=1e-6)
        np.testing.assert_allclose(points["y"], [0, 2], atol=1e-6)
        self.assertTrue((points["z"] == 0).all())

    def test_invalid_times_leave_no_output(self):
        for times in [
            ("2026/08/02 00:00:01:000",) * 2,
            ("2026/08/02 00:00:02:000", "2026/08/02 00:00:01:000"),
        ]:
            with tempfile.TemporaryDirectory() as tmp:
                dest = Path(tmp) / "out"
                with self.assertRaises(ValueError):
                    export(self.fixture(tmp, times), dest)
                self.assertFalse(dest.exists())

    def test_incomplete_and_invalid_rows_preserved_as_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_text(
                "ObjId,range,angle,speed\nSTART,FrameNb:1\n2026/08/02 00:00:01:000---->\n0,nan,0,1\n1,2,0,1\n"
            )
            report = export(p, Path(tmp) / "out")
            self.assertFalse(report["frames"][0]["complete"])
            self.assertEqual(report["frames"][0]["rejected_rows"], 1)
            self.assertEqual(report["frames"][0]["points"], 1)

    def test_missing_timestamp_and_existing_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "x.csv"
            p.write_text("ObjId,range,angle,speed\nSTART\nEND\n")
            with self.assertRaises(ValueError):
                export(p, Path(tmp) / "out")
            p = self.fixture(tmp)
            out = Path(tmp) / "existing"
            out.mkdir()
            with self.assertRaises(FileExistsError):
                export(p, out)

    def test_float32_overflow_rejected(self):
        with np.errstate(over="ignore"), self.assertRaises(ValueError):
            cloud([(1e100, 0, 0, 1)])


if __name__ == "__main__":
    unittest.main()
