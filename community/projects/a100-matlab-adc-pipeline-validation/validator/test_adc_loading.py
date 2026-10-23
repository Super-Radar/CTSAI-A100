"""Sample-order regressions, independent of FFT/CFAR agreement."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from run_reference_validation import load_adc


class ADCLoadingTests(unittest.TestCase):
    cfg = {"nfft": 4, "chirps": 2}
    # High half precedes low half; two words make one chirp.
    words = [2147549183, 32767, 655380, 1966120]
    samples = np.array([[-32768, 10], [-1, 20], [0, 30], [32767, 40]], float)

    def load(self, tokens):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adc.txt"
            path.write_text(",".join(map(str, tokens)) + "\n", encoding="utf-8")
            return load_adc([path], self.cfg)[:, :, 0]

    def test_padded_and_unpadded_capture_preserve_every_sample(self):
        expected = self.samples - self.samples.mean(axis=0)
        for suffix in ([], [0]):
            with self.subTest(suffix=suffix):
                np.testing.assert_array_equal(self.load(self.words + suffix), expected)

    def test_only_extra_zero_is_removed(self):
        # The final payload word is zero too: it must remain in the last chirp.
        samples = self.samples.copy()
        samples[2:, 1] = 0
        expected = samples - samples.mean(axis=0)
        for suffix in ([], [0]):
            np.testing.assert_array_equal(self.load(self.words[:-1] + [0] + suffix), expected)

    def test_all_zero_capture(self):
        np.testing.assert_array_equal(self.load([0] * 5), np.zeros((4, 2)))

    def test_reject_nonzero_or_invalid_trailer(self):
        for trailer in (1, -1, "NaN", "Inf", "bad", ""):
            with self.subTest(trailer=trailer), self.assertRaises(ValueError):
                self.load(self.words + [trailer])

    def test_reject_invalid_payload_without_deleting_positions(self):
        for bad in ("NaN", "Inf", "-Inf", -1, 2**32, 1.5, "", "bad", "1+2i"):
            for suffix in ([], [0]):
                with self.subTest(bad=bad, suffix=suffix), self.assertRaises(ValueError):
                    self.load([self.words[0], bad] + self.words[2:] + suffix)

    def test_reject_wrong_lengths_and_three_field_header(self):
        for tokens in ([], self.words[:-1], self.words + [0, 0], [0, 4, 2] + self.words):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                self.load(tokens)

    def test_channel_order_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = []
            expected = []
            for rx in range(4):
                samples = self.samples + rx * np.arange(1, 5)[:, None]
                samples[0, 0] += 100  # Keep all channels within int16.
                samples[3, 0] -= 100
                flat = samples.astype(int).flatten(order="F")
                words = [((int(a) & 65535) << 16) | (int(b) & 65535)
                         for a, b in zip(flat[::2], flat[1::2])]
                path = Path(directory) / f"rx{rx}.txt"
                path.write_text(",".join(map(str, words + [0])), encoding="utf-8")
                paths.append(path)
                expected.append(samples - samples.mean(axis=0))
            np.testing.assert_array_equal(load_adc(paths, self.cfg), np.stack(expected, axis=2))

    def test_public_captures_match_scalar_unpacking(self):
        data_dir = Path(__file__).resolve().parents[4] / "ADC数据采集" / "示例adc数据和结果"
        for marker, nfft, chirps in (("f0", 1024, 256), ("f1", 2048, 128)):
            files = sorted(data_dir.glob(f"*_{marker}_*.txt"))
            self.assertEqual(len(files), 4, "Run within a repository checkout containing public ADC data")
            channels = []
            for path in files:
                words = [int(token) for token in path.read_text().strip().split(",")]
                count = nfft // 2 * chirps
                self.assertEqual(len(words), count + 1)
                self.assertEqual(words[-1], 0)
                samples = []
                # Independently implement the documented high/low int16 order
                # with scalar integer division, rather than NumPy bit views.
                for word in words[:count]:
                    for half in divmod(word, 65536):
                        samples.append(half if half < 32768 else half - 65536)
                x = np.array(samples, float).reshape(chirps, nfft).T
                channels.append(x - x.mean(axis=0))
            with self.subTest(profile=marker):
                actual = load_adc(files, {"nfft": nfft, "chirps": chirps})
                np.testing.assert_array_equal(actual, np.stack(channels, axis=2))


if __name__ == "__main__":
    unittest.main()
