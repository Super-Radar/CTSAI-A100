import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from study import calibrate, rigid


class ConsensusRegression(unittest.TestCase):
    def test_returned_mask_matches_final_transform_and_fit(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=(30, 2))
        y = x + [1, 2] + rng.normal(0, 0.08, (30, 2))
        r, t, mask = calibrate(x, y, threshold=0.12)
        np.testing.assert_array_equal(
            mask, np.linalg.norm(x @ r.T + t - y, axis=1) <= 0.12
        )
        expected_r, expected_t = rigid(x[mask], y[mask])
        np.testing.assert_allclose(r, expected_r)
        np.testing.assert_allclose(t, expected_t)
