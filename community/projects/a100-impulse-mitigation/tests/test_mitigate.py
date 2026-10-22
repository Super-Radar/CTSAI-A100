import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import tempfile
import unittest
from pathlib import Path

import numpy as np

from adc_io import load
from mitigate import mitigate


class Tests(unittest.TestCase):
    def test_single_impulse_linear_oracle(self):
        x = np.arange(100, dtype=float)[None, :]
        bad = x.copy()
        bad[0, 50] = 10000
        y, mask, valid = mitigate(bad)
        np.testing.assert_array_equal(y, x)
        self.assertEqual(mask.sum(), 1)
        self.assertTrue(valid[0])

    def test_clean_identity_and_independent_rows(self):
        x = np.stack([np.sin(np.arange(100)), 20 * np.sin(np.arange(100))])
        y, m, v = mitigate(x)
        np.testing.assert_array_equal(y, x)
        self.assertFalse(m.any())
        self.assertTrue(v.all())

    def test_edges_are_not_fabricated(self):
        x = np.zeros((1, 100))
        x[0, 0] = 100
        y, _m, v = mitigate(x)
        self.assertFalse(v[0])
        self.assertTrue(np.isnan(y).all())

    def test_long_runs_and_fraction_reject(self):
        for length, kwargs in [(9, {"max_fraction": 0.5}), (6, {"max_run": 20})]:
            x = np.zeros((1, 100))
            x[0, 30 : 30 + length] = 100
            _, _, v = mitigate(x, **kwargs)
            self.assertFalse(v[0])

    def test_zero_mode_and_recovery_mask(self):
        x = np.zeros((1, 100))
        x[0, 50] = 100
        y, m, _v = mitigate(x, method="zero")
        self.assertEqual(y.sum(), 0)
        self.assertTrue(m[0, 50])

    def test_invalid(self):
        for x in [[], [[float("nan")]], [[1j]]]:
            with self.assertRaises(ValueError):
                mitigate(x)
        with self.assertRaises(ValueError):
            mitigate([[1, 2]], max_run=0)

    def test_adc_loader_signed_and_corrupt(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "adc.txt"
            p.write_text("0,2,1,4294934528,0")
            x, _ = load(p)
            np.testing.assert_array_equal(x, [[-1, -32768]])
            for data in ["0,2,1,1,,0", "0,2,1,4294967296", "0,2,2,1"]:
                p.write_text(data)
                with self.assertRaises(ValueError):
                    load(p)


if __name__ == "__main__":
    unittest.main()
