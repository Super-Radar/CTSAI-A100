import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from stream_adc import range_profile


class DegenerateWindowRegression(unittest.TestCase):
    def test_two_samples_cannot_silently_produce_zero_hann_spectrum(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "input.txt"
            p.write_text("0,2,1,65537")
            with self.assertRaises(ValueError):
                range_profile(p)
