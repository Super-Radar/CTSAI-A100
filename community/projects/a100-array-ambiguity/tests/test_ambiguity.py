import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import unittest

import numpy as np

from ambiguity import diagnose, peak_candidates, steering


class Tests(unittest.TestCase):
    def test_exact_spatial_alias(self):
        a = np.arange(-75, 75.1, 0.5)
        d = steering([0, 1, 2, 3], [-30, 30])
        self.assertAlmostEqual(abs(np.vdot(d[:, 0], d[:, 1])), 1)
        peaks = peak_candidates(d[:, 0], [0, 1, 2, 3], a)
        self.assertIn(-30, peaks)
        self.assertIn(30, peaks)

    def test_half_wave_disambiguates_pair(self):
        d = steering([0, 0.5, 1, 1.5], [-30, 30])
        self.assertLess(abs(np.vdot(d[:, 0], d[:, 1])), 1e-12)

    def test_translation_and_channel_permutation(self):
        a = np.arange(-70, 71)
        p = np.array([0, 0.9, 2.4, 3.1])
        one = diagnose(p, a)["coherence"]
        np.testing.assert_allclose(one, diagnose(p + 100, a)["coherence"], atol=1e-13)
        np.testing.assert_allclose(one, diagnose(p[::-1], a)["coherence"], atol=1e-13)

    def test_single_channel_unobservable(self):
        result = diagnose([0], np.arange(-75, 76))
        self.assertTrue(result["ambiguous"].all())

    def test_zero_and_amplitude_independence(self):
        a = np.arange(-75, 76)
        p = [0, 0.5, 1, 1.5]
        x = steering(p, [20])[:, 0]
        self.assertEqual(len(peak_candidates(x * 0, p, a)), 0)
        np.testing.assert_array_equal(
            peak_candidates(x, p, a), peak_candidates(x * 1e-100, p, a)
        )

    def test_invalid(self):
        with self.assertRaises(ValueError):
            diagnose([0, 1], [0, 0])
        with self.assertRaises(ValueError):
            diagnose([0, 1], [-1, 1], min_separation=10)
        with self.assertRaises(ValueError):
            steering([0, float("nan")], [0])
        with self.assertRaises(ValueError):
            peak_candidates([1], [0], [0], relative_power=0)


if __name__ == "__main__":
    unittest.main()
