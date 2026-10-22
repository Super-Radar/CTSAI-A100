import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
import unittest

import numpy as np

from spectrum import sampling_window, spectrum


class Tests(unittest.TestCase):
    def test_zero_fill_oracle_with_missing_first_sample(self):
        t = np.array([3, 4, 6, 8, 9])
        x = np.array([1, 2j, 3, 4j, 5])
        f = np.fft.fftfreq(16)
        full = np.zeros(16, dtype=complex)
        full[t] = x
        expected = np.fft.fft(full) / len(x) * np.exp(2j * np.pi * f * t[0])
        np.testing.assert_allclose(spectrum(x, t, f), expected, atol=1e-13)

    def test_complete_uniform_equals_fft(self):
        rng = np.random.default_rng(1)
        x = rng.normal(size=37) + 1j * rng.normal(size=37)
        f = np.fft.fftfreq(len(x))
        np.testing.assert_allclose(
            spectrum(x, np.arange(len(x)), f), np.fft.fft(x) / len(x), atol=1e-13
        )

    def test_weighted_fft_oracle(self):
        x = np.exp(2j * np.pi * 0.125 * np.arange(32))
        w = np.hanning(32)
        np.testing.assert_allclose(
            spectrum(x, np.arange(32), np.fft.fftfreq(32), w),
            np.fft.fft(x * w) / w.sum(),
            atol=1e-13,
        )

    def test_irregular_known_tone(self):
        t = np.array([0, 0.7, 1.3, 2.2, 4.8, 6.1, 8.0])
        f = np.linspace(-0.5, 0.5, 1001)
        truth = 0.173
        x = np.exp(2j * np.pi * truth * t)
        self.assertAlmostEqual(f[np.argmax(abs(spectrum(x, t, f)))], truth, places=12)

    def test_missing_indices_preserved(self):
        t = np.arange(64)[::2]
        np.arange(-32, 32) / 64
        x = np.exp(2j * np.pi * 0.125 * t)
        self.assertAlmostEqual(abs(spectrum(x, t, [0.125])[0]), 1)
        # Regular decimation has exact aliases; timestamp awareness cannot fix them.
        self.assertAlmostEqual(abs(spectrum(x, t, [-0.375])[0]), 1)

    def test_origin_shift_and_blocking(self):
        t = np.array([10, 10.5, 12.0, 15.0])
        x = np.array([1, 2j, 3, 4j])
        f = np.linspace(-1, 1, 301)
        np.testing.assert_allclose(
            spectrum(x, t, f, block=1), spectrum(x, t - 10, f, block=71), atol=1e-14
        )

    def test_window_and_invalid(self):
        self.assertAlmostEqual(sampling_window([0, 1, 2], [0])[0], 1)
        for t in [[0, 0], [1, 0], [0, float("nan")]]:
            with self.assertRaises(ValueError):
                spectrum([1, 1], t, [0])
        with self.assertRaises(ValueError):
            spectrum([1, 1], [0, 1], [0], weights=[0, 0])
        with self.assertRaises(ValueError):
            spectrum([1, 1], [0, 1], [0], block=0)


if __name__ == "__main__":
    unittest.main()
