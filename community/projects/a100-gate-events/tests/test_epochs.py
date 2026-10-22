import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate import GateCounter


class EpochTests(unittest.TestCase):
    def crossing(self, counter):
        counter.update(0, {"a": 1})
        return counter.update(0.1, {"a": -1})[0]

    def test_repeated_clock_and_identity_are_distinct_events(self):
        counter = GateCounter(confirm_seconds=0)
        first = self.crossing(counter)
        second = self.crossing(counter)
        self.assertEqual(first["time_s"], second["time_s"])
        self.assertEqual(first["track_id"], second["track_id"])
        self.assertEqual([first["epoch"], second["epoch"]], [0, 1])

    def test_explicit_reset_without_clock_reset_starts_new_epoch(self):
        counter = GateCounter(confirm_seconds=0)
        self.crossing(counter)
        counter.reset()
        self.assertEqual(counter.epoch, 1)
        self.assertEqual(counter.update(1, {"a": -1}), [])
        event = counter.update(1.1, {"a": 1})[0]
        self.assertEqual(event["epoch"], 1)

    def test_invalid_frame_clears_history_and_advances_epoch(self):
        counter = GateCounter(confirm_seconds=0)
        self.crossing(counter)
        with self.assertRaises(ValueError):
            counter.update(float("nan"), {"a": 1})
        self.assertEqual(counter.epoch, 1)
        self.assertEqual(counter.update(0.2, {"a": 1}), [])

    def test_track_expiry_does_not_reset_other_identities_or_epoch(self):
        counter = GateCounter(confirm_seconds=0, max_gap=0.5)
        counter.update(0, {"a": 1, "b": 1})
        counter.update(0.4, {"b": 1})
        event = counter.update(0.8, {"a": -1, "b": -1})
        self.assertEqual([e["track_id"] for e in event], ["b"])
        self.assertEqual(event[0]["epoch"], 0)
