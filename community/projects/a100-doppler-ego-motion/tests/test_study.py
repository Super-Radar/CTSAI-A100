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
    def test_exact_recovery_and_permutation(self):
        theta = np.linspace(-1, 1, 30)
        truth = np.array([2.0, -0.3])
        v = -np.column_stack((np.cos(theta), np.sin(theta))) @ truth
        np.testing.assert_allclose(study.fit_velocity(theta, v)[0], truth, atol=1e-12)
        np.testing.assert_allclose(
            study.fit_velocity(theta[::-1], v[::-1])[0], truth, atol=1e-12
        )

    def test_outliers(self):
        theta = np.linspace(-1, 1, 40)
        truth = np.array([2.0, 0.4])
        v = -np.column_stack((np.cos(theta), np.sin(theta))) @ truth
        v[::4] += 3
        np.testing.assert_allclose(study.fit_velocity(theta, v)[0], truth, atol=1e-12)
        self.assertGreater(
            np.linalg.norm(study.fit_velocity(theta, v, robust=False)[0] - truth), 0.5
        )

    def test_unobservable_and_invalid(self):
        for theta, v in (
            (np.zeros(10), np.zeros(10)),
            ([0, 1], [0, 1]),
            ([0, 1, 2], [0, np.nan, 1]),
        ):
            with self.assertRaises(ValueError):
                study.fit_velocity(theta, v)
