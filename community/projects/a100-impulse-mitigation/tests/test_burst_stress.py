import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from benchmark import error
from burst_stress import inject, positions
from mitigate import mitigate


class BurstTests(unittest.TestCase):
    def test_injection_indices_clipping_and_source_immutability(self):
        original = np.zeros((3, 100))
        original[1, 99] = 32000
        damaged, truth = inject(
            original, np.array([1]), positions("right_edge", 3, 100), 20000
        )
        self.assertEqual(np.flatnonzero(truth[1]).tolist(), [97, 98, 99])
        self.assertEqual(damaged[1, 99], 32767)
        self.assertEqual(original[1, 99], 32000)
        self.assertFalse(truth[[0, 2]].any())

    def test_no_retained_rows_is_null_error_not_perfect_repair(self):
        x = np.zeros((2, 100))
        damaged, _ = inject(x, np.arange(2), positions("left_edge", 3, 100), 20000)
        repaired, _, usable = mitigate(damaged)
        self.assertFalse(usable.any())
        self.assertIsNone(error(x, repaired, usable))
        self.assertTrue(np.isnan(repaired).all())

    def test_half_chirp_contamination_defeats_mad_on_known_flat_signal(self):
        x = np.zeros((1, 100))
        damaged, truth = inject(
            x, np.array([0]), positions("half_chirp", 0, 100), 20000
        )
        repaired, mask, usable = mitigate(damaged)
        self.assertEqual(truth.sum(), 50)
        self.assertFalse(mask.any())
        self.assertTrue(usable[0])
        np.testing.assert_array_equal(repaired, damaged)

    def test_distributed_fraction_is_not_a_long_connected_run(self):
        indices = positions("distributed_6pct", 0, 256)
        self.assertGreater(len(indices) / 256, 0.05)
        self.assertTrue(np.all(np.diff(indices) > 1))
