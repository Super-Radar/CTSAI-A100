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
    def test_two_groups_and_noise(self):
        p = np.array(
            [
                [0, 0],
                [0.1, 0],
                [0, 0.1],
                [0.1, 0.1],
                [3, 0],
                [3.1, 0],
                [3, 0.1],
                [3.1, 0.1],
                [10, 10],
            ]
        )
        labels = study.density_labels(study.neighbourhood(p))
        self.assertEqual(len(set(labels) - {-1}), 2)
        self.assertEqual(labels[-1], -1)

    def test_border_does_not_bridge(self):
        adj = np.eye(9, dtype=bool)
        adj[:4, :4] = True
        adj[5:, 5:] = True
        adj[3, 4] = adj[4, 3] = True
        adj[4, 5] = adj[5, 4] = True
        labels = study.density_labels(adj)
        self.assertNotEqual(labels[0], labels[8])

    def test_rotation_invariance(self):
        p = np.array([[10, 0], [10, 0.3], [20, 2], [21, 3]], float)
        r = np.array([[0, -1], [1, 0]])
        np.testing.assert_array_equal(
            study.neighbourhood(p, True), study.neighbourhood(p @ r.T, True)
        )

    def test_empty_and_invalid(self):
        self.assertEqual(len(study.density_labels(np.empty((0, 0), bool))), 0)
        self.assertEqual(study.neighbourhood(np.empty((0, 2)), True).shape, (0, 0))
        with self.assertRaises(ValueError):
            study.neighbourhood([[np.nan, 0]])
