import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import study
from logio import load


class LogTests(unittest.TestCase):
    def test_frame_identity_and_validation(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "log.csv"
            p.write_text(
                "ObjId,range,angle,speed\nSTART,FrameNb:1\n0,1,0,2\n1,nan,0,0\nEND\nSTART,FrameNb:1\n0,2,90,0\n"
            )
            frames, meta = load(p)
        self.assertEqual(len(frames), 2)
        self.assertEqual(meta["rejected_rows"], 1)
        self.assertEqual(meta["incomplete_frames"], 1)
        self.assertAlmostEqual(frames[1]["points"][0, 1], np.pi / 2)

    def test_empty_is_not_success(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "empty.csv"
            p.write_text("")
            with self.assertRaises(ValueError):
                load(p)


class AlgorithmTests(unittest.TestCase):
    def test_known_transform_and_inverse(self):
        x = np.array([[0, 0], [1, 0], [0, 2], [3, 4]], float)
        truth = np.array([[0, -1], [1, 0]], float)
        y = x @ truth.T + [2, 3]
        r, t = study.rigid(x, y)
        np.testing.assert_allclose(r, truth, atol=1e-12)
        np.testing.assert_allclose((y - t) @ r, x, atol=1e-12)
        self.assertAlmostEqual(np.linalg.det(r), 1)

    def test_outlier_recovery(self):
        x = np.random.default_rng(1).normal(size=(30, 2))
        y = x + [1, 2]
        y[::5] += 10
        r, t, mask = study.calibrate(x, y)
        np.testing.assert_allclose(r, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(t, [1, 2], atol=1e-12)
        self.assertEqual(mask.sum(), 24)

    def test_reflection_not_accepted_as_rotation(self):
        x = np.array([[0, 0], [1, 0], [0, 2], [2, 3]], float)
        y = x * [-1, 1]
        r, t = study.rigid(x, y)
        self.assertAlmostEqual(np.linalg.det(r), 1)
        self.assertGreater(np.linalg.norm(x @ r.T + t - y), 0.1)

    def test_degenerate(self):
        with self.assertRaises(ValueError):
            study.rigid(np.ones((4, 2)), np.ones((4, 2)))
