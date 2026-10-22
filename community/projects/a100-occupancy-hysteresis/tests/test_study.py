import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import study
from logio import load


class LogTests(unittest.TestCase):
    def test_frame_identity_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.csv"
            p.write_text(
                "ObjId,range,angle,speed\nSTART,FrameNb:1\n0,1,0,2\n1,nan,0,0\nEND\nSTART,FrameNb:1\n0,2,90,0\n"
            )
            frames, meta = load(p)
        self.assertEqual(len(frames), 2)
        self.assertEqual(meta["rejected_rows"], 1)
        self.assertEqual(meta["incomplete_frames"], 1)
        self.assertAlmostEqual(frames[1]["points"][0, 1], np.pi / 2)

    def test_empty_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.csv"
            p.write_text("")
            with self.assertRaises(ValueError):
                load(p)


class AlgorithmTests(unittest.TestCase):
    def test_published_roi_boundaries(self):
        from validate_presence import narrow_count

        points = np.array(
            [
                [2.5, np.deg2rad(-20), 0],
                [3.1, np.deg2rad(5), 0],
                [3.1001, 0, 0],
                [2.8, np.deg2rad(5.01), 0],
            ]
        )
        self.assertEqual(narrow_count(points), 2)

    def test_watchdog_without_new_measurements(self):
        o = study.Occupancy(enter_seconds=0)
        self.assertIs(o.update(0, 2), True)
        self.assertIs(o.value_at(0.2), True)
        self.assertIsNone(o.value_at(0.251))
        self.assertIsNone(o.value_at(0.4))

    def test_poll_does_not_confirm_or_extend_measurement(self):
        o = study.Occupancy(max_gap=0.5)
        o.update(0, 2)
        self.assertIsNone(o.value_at(0.4))
        self.assertEqual(o.last, 0)
        self.assertIs(o.update(0.4, 2), True)
        self.assertIs(o.value_at(0.8), True)
        self.assertIsNone(o.value_at(0.901))

    def test_watchdog_clock_reversal_and_invalid(self):
        o = study.Occupancy(enter_seconds=0)
        o.update(1, 2)
        self.assertIsNone(o.value_at(0.9))
        with self.assertRaises(ValueError):
            o.value_at(float("nan"))

    def test_irregular_time_not_frame_count(self):
        o = study.Occupancy(max_gap=0.5)
        self.assertIsNone(o.update(0, 2))
        self.assertIsNone(o.update(0.1, 2))
        self.assertIs(o.update(0.31, 2), True)
        self.assertIs(o.update(0.4, 0), True)
        self.assertIs(o.update(0.7, 0), True)
        self.assertIs(o.update(1.11, 0), False)

    def test_gap_and_reset(self):
        o = study.Occupancy(enter_seconds=0)
        self.assertIs(o.update(0, 2), True)
        self.assertIsNone(o.update(1, 0))
        self.assertIsNone(o.update(None, None))

    def test_border_breaks_confirmation(self):
        o = study.Occupancy()
        for t, c in ((0, 2), (0.2, 1), (0.3, 2), (0.5, 2)):
            self.assertIsNone(o.update(t, c))
        self.assertIs(o.update(0.6, 2), True)

    def test_clock_reversal_and_invalid(self):
        o = study.Occupancy(enter_seconds=0)
        o.update(3, 2)
        self.assertIsNone(o.update(2, 0))
        with self.assertRaises(ValueError):
            o.update(3, -1)
        with self.assertRaises(ValueError):
            study.Occupancy(enter_count=0)
