import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spectrum import spectrum


class WeightScaleRegression(unittest.TestCase):
    def test_projection_is_invariant_to_finite_weight_scale(self):
        x = np.array([1 + 2j, 3 - 1j, -2 + 4j])
        t = np.array([0, 0.7, 2])
        f = np.array([0, 0.2])
        expected = spectrum(x, t, f, weights=[1, 1, 1])
        with np.errstate(over="raise", invalid="raise"):
            for scale in [1e308, 1e-310]:
                np.testing.assert_allclose(
                    spectrum(x, t, f, weights=[scale] * 3), expected, atol=1e-14
                )
