import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from application import OccupancyApplication, PolarROI
from application_cases import METHODS, run_case

POINT = np.array([[2.8, 0.0, 0.0]])
EMPTY = np.empty((0, 3))


class ApplicationTests(unittest.TestCase):
    def app(self, **kwargs):
        return OccupancyApplication(enter_count=1, enter_seconds=0, **kwargs)

    def test_separate_clocks_and_outage(self):
        app = self.app()
        self.assertIs(app.observe(0, 86400, POINT)["state"], True)
        self.assertIs(app.poll(0.25)["state"], True)
        self.assertIsNone(app.poll(0.251)["state"])
        self.assertEqual(app.events[-1]["reason"], "stale")

    def test_duplicates_cannot_renew_lease_even_after_expiry(self):
        app = self.app()
        app.observe(0, 10, POINT)
        for arrival in (0.1, 0.2, 0.3, 0.4):
            result = app.observe(arrival, 10, POINT)
            self.assertFalse(result["accepted"])
        self.assertIsNone(result["state"])
        self.assertEqual(app.device_time, 10)
        self.assertIs(app.observe(0.5, 11, POINT)["state"], True)

    def test_late_empty_frame_cannot_change_fresh_state(self):
        app = self.app(exit_seconds=0)
        app.observe(1, 10, POINT)
        result = app.observe(1.1, 9, EMPTY)
        self.assertIs(result["state"], True)
        self.assertEqual(result["reason"], "duplicate_or_out_of_order")
        self.assertEqual(app.machine.last, 1)

    def test_restart_requires_new_epoch_and_fresh_confirmation(self):
        app = OccupancyApplication(enter_count=1, enter_seconds=0.2)
        app.observe(0, 90, POINT)
        app.observe(0.2, 91, POINT)
        self.assertIs(app.machine.state, True)
        result = app.start_epoch(0.3)
        self.assertIsNone(result["state"])
        self.assertEqual(result["epoch"], 1)
        self.assertEqual(app.observe(0.31, 92, POINT, epoch=0)["reason"], "wrong_epoch")
        self.assertIsNone(app.observe(0.4, 0, POINT, epoch=1)["state"])
        self.assertIs(app.observe(0.6, 0.2, POINT, epoch=1)["state"], True)

    def test_clock_reset_is_not_inferred_from_old_packet(self):
        app = self.app()
        app.observe(0, 100, POINT)
        self.assertFalse(app.observe(0.1, 0, POINT)["accepted"])
        self.assertEqual(app.epoch, 0)
        self.assertIsNone(app.poll(0.3)["state"])

    def test_poll_clock_is_monotonic_even_between_measurements(self):
        app = self.app()
        app.observe(0, 0, POINT)
        app.poll(0.2)
        with self.assertRaises(ValueError):
            app.observe(0.1, 1, POINT)
        self.assertIsNone(app.machine.state)
        with self.assertRaises(ValueError):
            app.poll(float("nan"))

    def test_empty_is_clear_but_incomplete_is_unknown(self):
        app = self.app(exit_seconds=0)
        app.observe(0, 10, POINT)
        self.assertIs(app.observe(0.1, 11, EMPTY)["state"], False)
        self.assertIsNone(app.observe(0.2, 12, EMPTY, complete=False)["state"])
        self.assertEqual(app.events[-1]["reason"], "incomplete_frame")

    def test_invalid_points_and_time_reset_without_counting_as_clear(self):
        app = self.app()
        for index, points in enumerate(
            (np.array([[np.nan, 0, 0]]), np.array([[1, 4, 0]]), np.empty((0, 2)))
        ):
            app.observe(index, index * 2, POINT)
            result = app.observe(index + 0.1, index * 2 + 1, points)
            self.assertEqual(result["reason"], "invalid_points")
            self.assertIsNone(result["state"])
        self.assertEqual(app.observe(4, None, POINT)["reason"], "invalid_device_time")

    def test_roi_closed_boundaries_and_empty_shape(self):
        roi = PolarROI()
        points = np.array(
            [[2.5, np.deg2rad(-20), 0], [3.1, np.deg2rad(5), 0], [3.100001, 0, 0]]
        )
        self.assertEqual(roi.count(points), 2)
        self.assertEqual(roi.count(EMPTY), 0)
        with self.assertRaises(ValueError):
            PolarROI(near=4, far=3)

    def test_polls_do_not_confirm_and_events_only_change_state(self):
        app = OccupancyApplication(enter_count=1, enter_seconds=0.2, max_gap=0.5)
        app.observe(0, 10, POINT)
        for t in (0.1, 0.2, 0.3):
            self.assertIsNone(app.poll(t)["state"])
        self.assertEqual(app.events, [])
        app.observe(0.4, 11, POINT)
        app.poll(0.5)
        self.assertEqual(len(app.events), 1)


class ReplayTests(unittest.TestCase):
    @staticmethod
    def packet(arrival, device=None, points=POINT, epoch=0):
        return {
            "arrival": arrival,
            "device": arrival if device is None else device,
            "points": points,
            "epoch": epoch,
            "complete": True,
            "order": 0,
        }

    def test_duration_accounting_has_explicit_unknown_and_common_denominator(self):
        report, _, _ = run_case([self.packet(0)], end=1, poll_period=0.1, label=True)
        self.assertAlmostEqual(report["methods"]["instant"]["occupied_s"], 0.3)
        self.assertAlmostEqual(report["methods"]["ordered_application"]["unknown_s"], 1)
        self.assertEqual(report["common_known_s"], 0)
        for name in METHODS:
            self.assertAlmostEqual(sum(report["methods"][name].values()), 1)

    def test_late_delivery_is_detected_and_does_not_reanimate_state(self):
        packets = [self.packet(t) for t in (0, 0.1, 0.2, 0.3, 0.4)]
        packets.append(self.packet(0.9, device=0.1))
        report, traces, receipts = run_case(packets, end=1.2)
        self.assertEqual(receipts[-1]["reason"], "duplicate_or_out_of_order")
        self.assertIsNone(receipts[-1]["state"])
        at_late = next(r for r in traces if r["time_s"] == 0.9)
        self.assertIs(at_late["instant"], True)
        self.assertIsNone(at_late["ordered_application"])
        self.assertEqual(report["packet_reasons"]["duplicate_or_out_of_order"], 1)

    def test_restart_precedes_same_time_packet(self):
        packets = [self.packet(0), self.packet(0.1, device=0, epoch=1)]
        _, _, receipts = run_case(packets, restart_times=[0.1], end=0.5)
        self.assertTrue(receipts[-1]["accepted"])
        self.assertEqual(receipts[-1]["epoch"], 1)

    def test_empty_bad_period_and_end_are_rejected(self):
        for packets, kwargs in (
            ([], {}),
            ([self.packet(0)], {"poll_period": 0}),
            ([self.packet(1)], {"end": 0}),
            ([self.packet(0)], {"restart_times": [-1]}),
        ):
            with self.assertRaises(ValueError):
                run_case(packets, **kwargs)


if __name__ == "__main__":
    unittest.main()
