import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import tempfile
import unittest
from pathlib import Path

from epochs import export, index_frames, reconstruct


def frame(stamp, complete=True):
    return (
        "START,FrameNb:7\r\n"
        + (stamp + "---->\r\n" if stamp else "")
        + "0,1,0,0\r\n"
        + ("END\r\n" if complete else "")
    ).encode()


class Tests(unittest.TestCase):
    def execute(self, data, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.csv"
            p.write_bytes(data)
            dest = Path(tmp) / "out"
            report = export(p, dest, **kwargs)
            self.assertEqual(reconstruct(dest), data)
            return report

    def test_monotonic_midnight_repeated_ids(self):
        data = (
            b"\xef\xbb\xbfObjId,range,angle,speed\r\n"
            + frame("2026/08/02 23:59:59:900")
            + frame("2026/08/03 00:00:00:100")
        )
        m = self.execute(data)
        self.assertEqual(m["epochs"], 1)

    def test_equal_reverse_and_long_gap(self):
        stamps = ["00:00:01:000", "00:00:01:000", "00:00:00:500", "00:00:10:000"]
        m = self.execute(
            b"header\n" + b"".join(frame("2026/08/02 " + s) for s in stamps)
        )
        self.assertEqual(m["epochs"], 4)
        self.assertEqual(
            [r["boundary_reason"] for r in m["frames"]][1:],
            ["non_increasing_time", "non_increasing_time", "long_gap"],
        )

    def test_missing_and_incomplete_quarantine(self):
        data = (
            frame("2026/08/02 00:00:00:000")
            + frame(None)
            + frame("2026/08/02 00:00:00:200")
            + frame("2026/08/02 00:00:00:300", False)
        )
        m = self.execute(data)
        self.assertEqual(m["epochs"], 2)
        self.assertEqual(sum(r["file"] == "quarantine.bin" for r in m["frames"]), 2)

    def test_multiple_timestamps_quarantine(self):
        data = frame("2026/08/02 00:00:00:000") + b"2026/08/02 00:00:00:001---->\n"
        m = self.execute(data)
        self.assertEqual(m["frames"][0]["file"], "quarantine.bin")

    def test_corruption_detected_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in"
            p.write_bytes(frame("2026/08/02 00:00:00:000"))
            out = Path(tmp) / "out"
            export(p, out)
            with self.assertRaises(FileExistsError):
                export(p, out)
            (out / "epoch_0000.csv").write_bytes(b"corrupt")
            with self.assertRaises(ValueError):
                reconstruct(out)

    def test_empty_and_bad_gap(self):
        with self.assertRaises(ValueError):
            index_frames(b"header\n")
        with self.assertRaises(ValueError):
            export("unused", "unused", max_gap=float("nan"))

    def test_trailing_bytes_and_empty_frame(self):
        data = b"header\nSTART,7\n2026/08/02 00:00:00:000---->\nEND\nstatus: done"
        m = self.execute(data)
        self.assertEqual(len(m["frames"]), 1)


if __name__ == "__main__":
    unittest.main()
