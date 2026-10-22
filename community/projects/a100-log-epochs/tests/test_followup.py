import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from epochs import export, restore


class StreamingRestoreTests(unittest.TestCase):
    def test_exact_restore_small_blocks_and_no_overwrite(self):
        data = (
            b"\xef\xbb\xbfheader\r\nSTART,7\r\n2026/08/02 00:00:00:000---->\r\n"
            + b"0,1,0,0\r\n" * 10000
            + b"END\r\nSTART,7\r\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.csv"
            p.write_bytes(data)
            directory = Path(tmp) / "split"
            export(p, directory)
            output = Path(tmp) / "restored.csv"
            restore(directory, output, chunk_bytes=17)
            self.assertEqual(output.read_bytes(), data)
            with self.assertRaises(FileExistsError):
                restore(directory, output)
            self.assertEqual(output.read_bytes(), data)

    def test_corrupt_export_does_not_publish_partial_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "in.csv"
            p.write_bytes(b"START,7\n2026/08/02 00:00:00:000---->\nEND\n")
            directory = Path(tmp) / "split"
            export(p, directory)
            (directory / "epoch_0000.csv").write_bytes(b"corrupt")
            output = Path(tmp) / "restored.csv"
            with self.assertRaises(ValueError):
                restore(directory, output)
            self.assertFalse(output.exists())
            self.assertEqual(
                sorted(p.name for p in Path(tmp).iterdir()), ["in.csv", "split"]
            )
