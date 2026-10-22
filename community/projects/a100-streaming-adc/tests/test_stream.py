import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stream_adc import range_profile, unpack, words


class StreamingTests(unittest.TestCase):
    def capture(self, text, block=8):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "capture.txt"
            path.write_text(text, encoding="ascii")
            return range_profile(path, block)

    def test_signed_order(self):
        np.testing.assert_array_equal(
            unpack([0x7FFF8000, 0xFFFF0001]), [32767, -32768, -1, 1]
        )

    def test_chunk_boundary_and_trailing_comma(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "x"
            path.write_text("0,1024,256,4294967295,0,\n")
            for chunk in (1, 2, 7, 65536):
                self.assertEqual(
                    list(words(path, chunk)), [0, 1024, 256, 4294967295, 0]
                )

    def test_oracle_and_partial_block(self):
        samples = np.array([[1, -2, 3, 4], [5, 6, -7, 8], [9, 10, 11, -12]])
        unsigned = samples.astype(np.int64) & 65535
        packed = (unsigned[:, ::2] << 16) | unsigned[:, 1::2]
        text = "0,4,3," + ",".join(map(str, packed.ravel())) + ",0"
        expected = np.mean(
            np.abs(np.fft.rfft(samples * np.hanning(4), axis=1)) ** 2, axis=0
        )
        for block in (1, 2, 8):
            actual, meta = self.capture(text, block)
            np.testing.assert_allclose(actual, expected, rtol=1e-14)
            self.assertEqual(meta["chirps"], 3)

    def test_invalid_tokens(self):
        for value in ("nan", "inf", "1.2", "-1", "1e3", "bad", "", "4294967296"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.capture(f"0,4,1,1,{value},0")

    def test_bad_header_and_truncation(self):
        for text in ("", "0,4", "0,3,1,0,0", "4,4,1,0,0", "0,4,0,0,0", "0,4,2,1,2,3"):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.capture(text)

    def test_trailer(self):
        for ending in ("1", "0,0", "0,1"):
            with self.assertRaises(ValueError):
                self.capture("0,4,1,1,2," + ending)
        self.assertEqual(self.capture("0,4,1,1,2")[1]["padding_words"], 0)

    def test_block_contract(self):
        for block in (0, -1, 257, 1.5):
            with self.assertRaises(ValueError):
                self.capture("0,4,1,0,0", block)


if __name__ == "__main__":
    unittest.main()
