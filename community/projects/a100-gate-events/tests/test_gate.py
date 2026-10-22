import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gate import GateCounter


def feed(counter, series):
    events = []
    for t, distances in series:
        events.extend(counter.update(t, distances))
    return events


class GateTests(unittest.TestCase):
    def test_ambiguous_strip_interrupts_rearm_dwell(self):
        series = [
            (0, {"a": 1}),
            (0.3, {"a": 1}),
            (0.4, {"a": -1}),
            (0.7, {"a": -1}),
            (1.0, {"a": -1}),
            (1.2, {"a": 0}),
            (1.3, {"a": -1}),
            (1.5, {"a": -1}),
            (1.6, {"a": 1}),
            (1.9, {"a": 1}),
        ]
        self.assertEqual(len(feed(GateCounter(), series)), 1)

    def test_confirmed_crossing(self):
        events = feed(
            GateCounter(),
            [(0, {"a": 1}), (0.3, {"a": 1}), (0.4, {"a": -1}), (0.7, {"a": -1})],
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["direction"], "negative")

    def test_strip_jitter_and_short_excursion(self):
        series = [
            (0, {"a": 1}),
            (0.3, {"a": 1}),
            (0.4, {"a": -0.2}),
            (0.5, {"a": 0.05}),
            (0.6, {"a": -0.05}),
            (0.7, {"a": 1}),
        ]
        self.assertEqual(feed(GateCounter(), series), [])

    def test_reverse_after_rearming(self):
        series = [
            (0, {"a": 1}),
            (0.3, {"a": 1}),
            (0.4, {"a": -1}),
            (0.7, {"a": -1}),
            (1.0, {"a": -1}),
            (1.3, {"a": -1}),
            (1.4, {"a": 1}),
            (1.7, {"a": 1}),
        ]
        self.assertEqual(
            [x["direction"] for x in feed(GateCounter(), series)],
            ["negative", "positive"],
        )

    def test_quick_reverse_suppressed(self):
        series = [
            (0, {"a": 1}),
            (0.3, {"a": 1}),
            (0.4, {"a": -1}),
            (0.7, {"a": -1}),
            (0.8, {"a": 1}),
            (1.1, {"a": 1}),
        ]
        self.assertEqual(len(feed(GateCounter(), series)), 1)

    def test_gap_missing_identity_and_id_reuse(self):
        for middle in [(1, {}), (1, {"other": -1})]:
            series = [
                (0, {"a": 1}),
                (0.3, {"a": 1}),
                middle,
                (1.1, {"a": -1}),
                (1.4, {"a": -1}),
            ]
            self.assertEqual(feed(GateCounter(), series), [])

    def test_crossing_identities_are_independent(self):
        series = [
            (0, {"a": 1, "b": -1}),
            (0.3, {"a": 1, "b": -1}),
            (0.4, {"a": -1, "b": 1}),
            (0.7, {"a": -1, "b": 1}),
        ]
        events = feed(GateCounter(), series)
        self.assertEqual(
            {(e["track_id"], e["direction"]) for e in events},
            {("a", "negative"), ("b", "positive")},
        )

    def test_duplicate_and_reversed_time_reset(self):
        for stamp in (0.3, 0.2):
            events = feed(
                GateCounter(),
                [(0, {"a": 1}), (0.3, {"a": 1}), (stamp, {"a": -1}), (0.5, {"a": -1})],
            )
            self.assertEqual(events, [])

    def test_nonfinite_and_invalid_options(self):
        with self.assertRaises(ValueError):
            GateCounter(max_gap=0)
        with self.assertRaises(ValueError):
            GateCounter().update(float("nan"), {})
        with self.assertRaises(ValueError):
            GateCounter().update(0, {"a": float("inf")})


if __name__ == "__main__":
    unittest.main()
