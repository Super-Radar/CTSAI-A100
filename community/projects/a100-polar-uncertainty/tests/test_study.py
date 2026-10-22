import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from study import experiment, moments, read_targets


class MomentsTests(unittest.TestCase):
    def test_zero_angle_noise_is_radial(self):
        mean, cov, linear = moments(10, 0, 0.2, 0)
        np.testing.assert_allclose(mean, [10, 0])
        np.testing.assert_allclose(cov, [[0.04, 0], [0, 0]])
        np.testing.assert_allclose(cov, linear)

    def test_rotation_and_small_angle_limit(self):
        _, cov, _ = moments(10, 0, 0.1, 1e-7)
        _, rotated, _ = moments(10, math.pi / 2, 0.1, 1e-7)
        np.testing.assert_allclose(np.diag(rotated), np.diag(cov)[::-1], atol=1e-15)
        self.assertAlmostEqual(cov[1, 1], 100.01e-14, places=20)

    def test_noise_free_and_invalid(self):
        _, cov, _ = moments(2, 0.5, 0, 0)
        np.testing.assert_array_equal(cov, np.zeros((2, 2)))
        for args in ((-1, 0, 0.1, 0.1), (1, 0, -0.1, 0.1), (1, 0, 0.1, float("nan"))):
            with self.assertRaises(ValueError):
                moments(*args)

    def test_monte_carlo_matches_exact_and_exposes_linear_limit(self):
        rows = experiment(200000, 90210)
        self.assertLess(max(r["mc_relative_covariance_error"] for r in rows), 0.025)
        self.assertGreater(rows[-1]["linear_relative_covariance_error"], 0.1)

    def test_framed_records_and_invalid_values(self):
        content = "ObjId,range,angle\nSTART,FrameNb:8\n0,2,90\n1,nan,0\n2,-1,0\n3,2,inf\nEND\nSTART,FrameNb:8\n0,3,0\n"
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "targets.csv"
            path.write_text(content)
            targets, counts = read_targets(path)
        self.assertEqual(len(targets), 2)
        self.assertEqual(
            counts,
            {
                "frames": 2,
                "incomplete_frames": 1,
                "rejected_rows": 3,
                "valid_targets": 2,
            },
        )
        self.assertAlmostEqual(targets[0][1], math.pi / 2)


if __name__ == "__main__":
    unittest.main()
