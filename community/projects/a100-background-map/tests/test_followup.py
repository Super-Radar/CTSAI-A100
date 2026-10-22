import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from background import BackgroundMap


class FrozenGridRegression(unittest.TestCase):
    def test_inspecting_config_cannot_reinterpret_fitted_cells(self):
        points = np.array([[3.1, 0.0]])
        model = BackgroundMap(min_frames=1).fit([points] * 20)
        before = model.keep(points)
        config = model.config
        config["range_step"] = 1.0
        np.testing.assert_array_equal(model.keep(points), before)
        self.assertEqual(model.config["range_step"], 0.25)
