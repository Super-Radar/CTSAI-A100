import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from application import CaptureClock, OccupancyApplication
from freshness_cases import replay

POINT = np.array([[2.8, 0.0, 0.0]])


class CaptureClockTests(unittest.TestCase):
    def test_affine_mapping_and_drift_bound(self):
        clock = CaptureClock(1000, 10, error_seconds=0.02, rate=1.001, drift_ppm=100)
        lower, upper = clock.bounds(1002)
        self.assertAlmostEqual(lower, 12.002 - 0.0202002)
        self.assertAlmostEqual(upper, 12.002 + 0.0202002)

    def test_continuously_increasing_old_packets_cannot_confirm(self):
        guarded = OccupancyApplication(
            capture_clock=CaptureClock(1000, 0), enter_count=1
        )
        legacy = OccupancyApplication(enter_count=1)
        for t in (0, 0.1, 0.2, 0.3, 0.4):
            result = guarded.observe(t + 1, t + 1000, POINT)
            legacy.observe(t + 1, t + 1000, POINT)
            self.assertFalse(result["accepted"])
            self.assertIsNone(result["state"])
            self.assertEqual(result["reason"], "old_or_uncertain_capture")
        self.assertIs(legacy.machine.state, True)

    def test_delayed_valid_packet_expires_by_capture_deadline(self):
        app = OccupancyApplication(
            capture_clock=CaptureClock(1000, 0), enter_count=1, enter_seconds=0
        )
        self.assertIs(app.observe(0.2, 1000, POINT)["state"], True)
        self.assertIs(app.poll(0.2499)["state"], True)
        result = app.poll(0.251)
        self.assertIsNone(result["state"])
        self.assertEqual(result["reason"], "capture_expired")

    def test_uncertainty_can_reject_a_physically_fresh_sample(self):
        app = OccupancyApplication(
            capture_clock=CaptureClock(1000, 0, error_seconds=0.1)
        )
        result = app.observe(0.2, 1000, POINT)
        self.assertEqual(result["reason"], "old_or_uncertain_capture")
        self.assertFalse(result["accepted"])

    def test_future_timestamp_does_not_poison_order_watermark(self):
        app = OccupancyApplication(
            capture_clock=CaptureClock(1000, 0), enter_count=1, enter_seconds=0
        )
        self.assertEqual(app.observe(0, 1010, POINT)["reason"], "future_capture")
        self.assertIsNone(app.device_time)
        self.assertTrue(app.observe(0.1, 1000.1, POINT)["accepted"])

    def test_restart_requires_new_mapping_without_silent_fallback(self):
        app = OccupancyApplication(
            capture_clock=CaptureClock(1000, 0), enter_count=1, enter_seconds=0
        )
        app.observe(0, 1000, POINT)
        app.start_epoch(1)
        self.assertEqual(
            app.observe(1, 0, POINT, epoch=1)["reason"], "missing_capture_clock"
        )
        self.assertIsNone(app.machine.state)
        epoch = app.start_epoch(2, capture_clock=CaptureClock(0, 2))["epoch"]
        self.assertTrue(app.observe(2, 0, POINT, epoch=epoch)["accepted"])

    def test_legacy_mode_still_uses_receipt_lease(self):
        app = OccupancyApplication(enter_count=1, enter_seconds=0)
        app.observe(10, 0, POINT)
        self.assertIs(app.poll(10.2)["state"], True)
        self.assertIsNone(app.poll(10.3)["state"])

    def test_invalid_clock_configurations(self):
        for kwargs in (
            {"rate": 0},
            {"error_seconds": -1},
            {"drift_ppm": 1e6},
            {"local_anchor": float("nan")},
        ):
            config = {"device_anchor": 0, "local_anchor": 0, **kwargs}
            with self.assertRaises(ValueError):
                CaptureClock(**config)
        with self.assertRaises(ValueError):
            CaptureClock(0, 0).bounds(float("inf"))

    def test_replay_uses_injected_truth_instead_of_reported_clock(self):
        frames = [
            {"time": t, "points": POINT, "complete": True}
            for t in (0, 0.1, 0.2, 0.3, 0.4)
        ]
        arrivals = [f["time"] + 1 for f in frames]
        valid, _, _ = replay(frames, arrivals, CaptureClock(10000, 0, 0.01))
        invalid, _, _ = replay(frames, arrivals, CaptureClock(10000, 0.95, 0.01))
        self.assertEqual(valid["methods"]["receipt_only"]["accepted_old_packets"], 5)
        self.assertEqual(valid["methods"]["capture_bound"]["accepted_old_packets"], 0)
        self.assertEqual(invalid["methods"]["capture_bound"]["accepted_old_packets"], 5)
        for report in (valid, invalid):
            for metrics in report["methods"].values():
                total = sum(metrics[k] for k in ("occupied_s", "clear_s", "unknown_s"))
                self.assertAlmostEqual(total, report["duration_s"])


if __name__ == "__main__":
    unittest.main()
