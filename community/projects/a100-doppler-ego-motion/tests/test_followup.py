import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from study import fit_velocity


class ConsensusRegression(unittest.TestCase):
    def test_returned_mask_matches_final_model_and_fit(self):
        theta = np.linspace(-1, 1, 30)
        a = -np.column_stack((np.cos(theta), np.sin(theta)))
        values = a @ [2, 0.4] + np.random.default_rng(1).normal(0, 0.08, 30)
        velocity, mask, _ = fit_velocity(theta, values, threshold=0.12)
        np.testing.assert_array_equal(mask, np.abs(a @ velocity - values) <= 0.12)
        np.testing.assert_allclose(
            velocity, np.linalg.lstsq(a[mask], values[mask], rcond=None)[0]
        )
