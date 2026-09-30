"""验证缺少 SciPy 时目标跟踪安全降级。"""

import os
import sys
from pathlib import Path
import unittest

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _support import load_radar_viewer, tkinter_root


class TrackingWithoutScipyTest(unittest.TestCase):
    def test_tracking_falls_back_to_object_ids(self):
        radar_viewer = load_radar_viewer("radar_viewer_no_scipy_test", without_scipy=True)
        with tkinter_root():
            tracker = radar_viewer.KalmanTracker()
            self.assertFalse(tracker.enable_tracking.get())
            tracker.enable_tracking.set(True)

        frame = pd.DataFrame({"ObjId": [7], "X": [1.0], "Y": [2.0]})
        first = tracker.track(frame.copy())
        second = tracker.track(frame.copy())

        self.assertEqual(first["track_id"].tolist(), [7])
        self.assertEqual(second["track_id"].tolist(), [7])


if __name__ == "__main__":
    unittest.main()
