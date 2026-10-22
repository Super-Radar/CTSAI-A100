"""Frozen per-frame polar-cell occupancy map; never learn during filtering."""

import json
from pathlib import Path

import numpy as np


class BackgroundMap:
    def __init__(
        self,
        range_step=0.25,
        angle_step=5.0,
        max_range=30.0,
        threshold=0.2,
        min_frames=20,
    ):
        values = [range_step, angle_step, max_range, threshold, min_frames]
        if (
            not np.isfinite(values).all()
            or min(range_step, angle_step, max_range) <= 0
            or not 0 < threshold < 1
            or min_frames < 1
            or int(min_frames) != min_frames
        ):
            raise ValueError("Invalid grid/probability/training length")
        self._config = {
            "range_step": range_step,
            "angle_step": angle_step,
            "max_range": max_range,
            "threshold": threshold,
            "min_frames": int(min_frames),
        }
        self.shape = (
            int(np.ceil(max_range / range_step)),
            int(np.ceil(180 / angle_step)),
        )
        self.hits = np.zeros(self.shape, dtype=np.int64)
        self.frames = 0

    @property
    def config(self):
        """A detached snapshot; changing it cannot reinterpret fitted cells."""
        return self._config.copy()

    def cells(self, points):
        p = np.asarray(points, dtype=float)
        if p.ndim != 2 or p.shape[1] != 2 or not np.isfinite(p).all():
            raise ValueError("Expected finite Nx2 range metres/angle degrees")
        if np.any(p[:, 0] < 0) or np.any(np.abs(p[:, 1]) > 90):
            raise ValueError("Range >=0 and angle in [-90,90] required")
        valid = p[:, 0] < self.config["max_range"]
        ids = np.zeros((len(p), 2), dtype=int)
        ids[valid, 0] = np.floor(p[valid, 0] / self.config["range_step"]).astype(int)
        # Include +90 in the last cell, while other edges are left inclusive.
        ids[valid, 1] = np.minimum(
            np.floor((p[valid, 1] + 90) / self.config["angle_step"]).astype(int),
            self.shape[1] - 1,
        )
        return ids, valid

    def fit(self, frames):
        hits = np.zeros(self.shape, dtype=np.int64)
        count = 0
        for points in frames:
            ids, valid = self.cells(points)
            for r, a in np.unique(ids[valid], axis=0):
                hits[r, a] += 1
            count += 1
        if count < self.config["min_frames"]:
            raise ValueError("Too few training frames")
        self.hits = hits
        self.frames = count
        return self

    def lower_bound(self):
        if self.frames < self.config["min_frames"]:
            raise ValueError("Fit the map first")
        n = self.frames
        p = self.hits / n
        z = 1.959963984540054
        return (
            p + z * z / (2 * n) - z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        ) / (1 + z * z / n)

    def keep(self, points):
        ids, valid = self.cells(points)
        keep = np.ones(len(ids), dtype=bool)
        background = self.lower_bound() >= self.config["threshold"]
        keep[valid] = ~background[ids[valid, 0], ids[valid, 1]]
        return keep

    def save(self, path):
        self.lower_bound()
        Path(path).write_text(
            json.dumps(
                {
                    "version": 1,
                    "config": self.config,
                    "frames": self.frames,
                    "hits": self.hits.tolist(),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path):
        value = json.loads(Path(path).read_text(encoding="utf-8"))
        if value.get("version") != 1:
            raise ValueError("Unsupported model version")
        model = cls(**value["config"])
        hits = np.asarray(value["hits"])
        frames = value["frames"]
        if (
            not isinstance(frames, int)
            or frames < model.config["min_frames"]
            or hits.shape != model.shape
            or not np.isfinite(hits).all()
            or np.any(hits < 0)
            or np.any(hits > frames)
            or np.any(hits != np.floor(hits))
        ):
            raise ValueError("Invalid saved counts")
        model.frames = frames
        model.hits = hits.astype(np.int64)
        return model
