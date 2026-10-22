import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate import GateCounter


class MissingTrackRegression(unittest.TestCase):
    def test_missing_frame_breaks_confirmation_without_forgetting_side(self):
        g = GateCounter()
        events = []
        for t, d in [
            (0, {"a": 1}),
            (0.3, {"a": 1}),
            (0.4, {"a": -1}),
            (0.5, {}),
            (0.7, {"a": -1}),
        ]:
            events.extend(g.update(t, d))
        self.assertEqual(events, [])
        g.update(0.9, {"a": -1})
        self.assertEqual(len(g.update(1, {"a": -1})), 1)

    def test_missing_frame_interrupts_rearm(self):
        g = GateCounter(max_gap=1)
        events = []
        for t, d in [
            (0, {"a": 1}),
            (0.3, {"a": 1}),
            (0.4, {"a": -1}),
            (0.7, {"a": -1}),
            (1, {}),
            (1.3, {"a": -1}),
            (1.4, {"a": 1}),
            (1.7, {"a": 1}),
        ]:
            events.extend(g.update(t, d))
        self.assertEqual(len(events), 1)
