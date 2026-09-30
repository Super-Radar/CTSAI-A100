"""验证 DBSCAN 噪点中心保持数值标量。"""

import os
import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _support import load_radar_viewer, Value


class RadarClusterNoiseCenterTest(unittest.TestCase):
    def test_noise_center_uses_its_own_coordinates(self):
        radar_viewer = load_radar_viewer("radar_viewer_cluster_test")
        cluster = radar_viewer.RadarCluster.__new__(radar_viewer.RadarCluster)
        cluster.enable_clustering = Value(True)
        cluster.eps = Value(0.5)
        cluster.min_samples = Value(2)
        points = pd.DataFrame({"X": [0.0, 0.2, 10.0], "Y": [0.0, 0.2, 20.0]})

        result = cluster.cluster(points)

        grouped = result[result["cluster_id"] == 0]
        self.assertEqual(grouped["cluster_size"].tolist(), [2, 2])
        self.assertTrue(all(np.isclose(grouped["cluster_center_x"], 0.1)))
        self.assertTrue(all(np.isclose(grouped["cluster_center_y"], 0.1)))
        noise = result[result["cluster_id"] == -1].iloc[0]
        self.assertEqual((noise["cluster_center_x"], noise["cluster_center_y"]), (10.0, 20.0))
        self.assertTrue(all(np.isscalar(value) for value in result["cluster_center_x"]))
        self.assertTrue(all(np.isscalar(value) for value in result["cluster_center_y"]))


if __name__ == "__main__":
    unittest.main()
