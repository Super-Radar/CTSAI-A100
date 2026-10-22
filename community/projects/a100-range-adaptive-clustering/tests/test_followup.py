import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from study import pair_scores


class NoiseLabelRegression(unittest.TestCase):
    def test_background_points_are_not_a_single_true_object(self):
        result = pair_scores(np.array([0, 0, 1, 1]), np.array([-1, -1, 0, 0]))
        self.assertEqual(result["precision"], 0.5)
        self.assertEqual(result["recall"], 1)

    def test_all_background_has_no_true_object_pairs(self):
        result = pair_scores(np.array([0, 0, 0]), np.array([-1, -1, -1]))
        self.assertEqual(result["precision"], 0)
        self.assertEqual(result["recall"], 0)
