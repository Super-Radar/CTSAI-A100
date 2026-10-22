import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from background import BackgroundMap
from pose_stress import evaluate, frozen_keep, transform


class PoseTests(unittest.TestCase):
    def test_transform_against_cartesian_geometry(self):
        point = np.array([[5, np.rad2deg(np.arctan2(3, 4))]])
        # Original (4,3), translated sensor at (0,3), turned +90deg: (0,-4).
        np.testing.assert_allclose(transform(point, 3, 90), [[4, -90]], atol=1e-12)

    def test_outside_view_is_not_clipped_to_background(self):
        model = BackgroundMap(min_frames=1, threshold=0.1).fit(
            [np.array([[2, 90]])] * 20
        )
        keep, outside = frozen_keep(model, np.array([[2, 91], [2, 90]]))
        np.testing.assert_array_equal(keep, [True, False])
        self.assertEqual(outside, 1)

    def test_zero_perturbation_and_roi_denominator(self):
        model = BackgroundMap().fit([np.array([[2.8, 0]])] * 30)
        frames = [np.array([[2.8, 0], [4, 20]]), np.empty((0, 2))]
        report, _ = evaluate(model, frames, 0, 0)
        self.assertEqual(report["decision_flips"], 0)
        self.assertEqual(report["original_roi_frames"], 1)
        self.assertEqual(report["original_roi_frames_retained"], 0)
        self.assertEqual(report["points"], 2)
