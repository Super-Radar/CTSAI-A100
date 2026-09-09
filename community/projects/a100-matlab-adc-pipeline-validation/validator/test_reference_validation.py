"""Regression tests for the independent CTSAI-A100 reference validator."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from run_reference_validation import load_adc


class LoadAdcValidationTest(unittest.TestCase):
    def test_nonfinite_payload_is_not_mistaken_for_an_optional_header(self) -> None:
        cfg = {"nfft": 4, "chirps": 2}
        with TemporaryDirectory() as directory:
            path = Path(directory) / "rx0.txt"
            # Four packed words are expected. The leading value is an optional
            # metadata word, while NaN corrupts the payload itself.
            np.savetxt(path, [99, 1, 2, np.nan, 3], delimiter=",")

            with self.assertRaisesRegex(ValueError, "non-finite payload"):
                load_adc([path], cfg)


if __name__ == "__main__":
    unittest.main()
