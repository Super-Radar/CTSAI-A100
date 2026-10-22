import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ambiguity import peak_candidates


class SnapshotScaleRegression(unittest.TestCase):
    def test_finite_components_do_not_overflow_complex_magnitude(self):
        angles = np.array([-30, 0, 30])
        expected = peak_candidates([1 + 1j, 1 + 1j], [0, 0.5], angles)
        actual = peak_candidates([complex(1.7e308, 1.7e308)] * 2, [0, 0.5], angles)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, [0])
