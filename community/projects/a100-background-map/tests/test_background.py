import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from background import BackgroundMap


class Tests(unittest.TestCase):
    def test_per_frame_not_per_detection(self):
        m = BackgroundMap(min_frames=2).fit(
            [np.tile([2.0, 0.0], (100, 1)), np.empty((0, 2))]
        )
        self.assertEqual(m.hits.max(), 1)
        self.assertEqual(m.frames, 2)

    def test_frozen_map_retains_new_target(self):
        m = BackgroundMap().fit([[[2, 0]]] * 30)
        before = m.hits.copy()
        for _ in range(50):
            np.testing.assert_array_equal(m.keep([[2, 0], [3, 20]]), [False, True])
        np.testing.assert_array_equal(m.hits, before)

    def test_training_contamination(self):
        m = BackgroundMap().fit([[[2, 0], [3, 20]]] * 30)
        self.assertFalse(m.keep([[3, 20]])[0])

    def test_boundaries_and_outside(self):
        m = BackgroundMap().fit([[[0, -90], [29.9, 90]]] * 30)
        np.testing.assert_array_equal(
            m.keep([[0, -90], [29.9, 90], [30, 0]]), [False, False, True]
        )

    def test_wilson_against_analytic_all_success(self):
        m = BackgroundMap().fit([[[2, 0]]] * 30)
        self.assertAlmostEqual(m.lower_bound().max(), 30 / (30 + 1.959963984540054**2))

    def test_save_reload_and_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "m.json"
            m = BackgroundMap().fit([[[2, 0]]] * 30)
            m.save(p)
            np.testing.assert_array_equal(BackgroundMap.load(p).keep([[2, 0]]), [False])
            bad = json.loads(p.read_text())
            bad["hits"][0][0] = 31
            p.write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                BackgroundMap.load(p)

    def test_invalid_and_empty_frames(self):
        with self.assertRaises(ValueError):
            BackgroundMap().fit([])
        with self.assertRaises(ValueError):
            BackgroundMap(threshold=0)
        m = BackgroundMap(min_frames=1).fit([np.empty((0, 2))])
        with self.assertRaises(ValueError):
            m.keep([[1, float("nan")]])
        with self.assertRaises(ValueError):
            m.keep([[-1, 0]])
        self.assertTrue(m.keep([[1, 0]])[0])


if __name__ == "__main__":
    unittest.main()
