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
    def test_empty_and_missed(self):
        self.assertEqual(study.gospa([], [])["distance"], 0)
        self.assertAlmostEqual(study.gospa([[0, 0]], [])["distance"], np.sqrt(2))
        self.assertEqual(study.gospa([[0, 0]], [[2, 0]])["matched"], 0)

    def test_symmetry_and_permutation(self):
        a = [[0, 0], [3, 0]]
        b = [[0.2, 0], [3.1, 0], [10, 0]]
        first = study.gospa(a, b)
        self.assertAlmostEqual(first["distance"], study.gospa(b, a)["distance"])
        self.assertAlmostEqual(
            first["distance"], study.gospa(a[::-1], b[::-1])["distance"]
        )
        self.assertAlmostEqual(first["localization_squared"], 0.05)
        self.assertEqual(first["false_squared"], 2)

    def test_exhaustive_oracle(self):
        import itertools

        rng = np.random.default_rng(4)
        for n in range(1, 4):
            for m in range(n, 5):
                a = rng.normal(size=(n, 2))
                b = rng.normal(size=(m, 2))
                d = np.minimum(np.linalg.norm(a[:, None] - b[None, :], axis=2), 2) ** 2
                best = (
                    min(
                        sum(d[i, j] for i, j in enumerate(p))
                        for p in itertools.permutations(range(m), n)
                    )
                    + (m - n) * 2
                )
                self.assertAlmostEqual(study.gospa(a, b)["distance"] ** 2, best)

    def test_invalid(self):
        with self.assertRaises(ValueError):
            study.gospa([[0, 0], [0, 0]], [])
        with self.assertRaises(ValueError):
            study.gospa([[np.nan, 0]], [])
