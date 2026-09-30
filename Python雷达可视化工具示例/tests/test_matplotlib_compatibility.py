"""验证当前 Matplotlib 配色 API 的兼容性。"""

import os
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _support import load_radar_viewer, tkinter_root


class MatplotlibCompatibilityTest(unittest.TestCase):
    def test_cluster_colormap_can_be_created(self):
        radar_viewer = load_radar_viewer("radar_viewer_matplotlib_test")

        with tkinter_root():
            cluster = radar_viewer.RadarCluster()

        self.assertEqual(cluster.cluster_colors.N, 20)


if __name__ == "__main__":
    unittest.main()
