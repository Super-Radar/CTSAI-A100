"""Minimal framed A100 log reader for these offline experiments."""

import csv
import hashlib
from datetime import datetime
from pathlib import Path

import numpy as np


def load(path):
    frames, frame, header = [], None, None
    rejected = 0
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            if row[0] == "ObjId":
                header = {k: i for i, k in enumerate(row)}
                if not {"ObjId", "range", "angle", "speed"} <= header.keys():
                    raise ValueError("Missing measurement columns")
            elif row[0] == "START":
                if frame is not None:
                    frames.append(frame)
                frame = {"time": None, "points": [], "complete": False}
            elif row[0] == "END":
                if frame is not None:
                    frame["complete"] = True
                    frames.append(frame)
                    frame = None
            elif frame is not None and "---->" in row[0]:
                frame["time"] = datetime.strptime(
                    row[0].replace("---->", "").strip(), "%Y/%m/%d %H:%M:%S:%f"
                )
            elif frame is not None and header is not None:
                try:
                    float(row[0])
                except ValueError:
                    if len(row) >= len(header):
                        rejected += 1
                    continue
                try:
                    obj, r, deg, v = [
                        float(row[header[k]])
                        for k in ("ObjId", "range", "angle", "speed")
                    ]
                    if (
                        not np.isfinite([obj, r, deg, v]).all()
                        or obj < 0
                        or obj != int(obj)
                        or r < 0
                    ):
                        raise ValueError("Invalid measurement")
                except (ValueError, IndexError):
                    rejected += 1
                    continue
                frame["points"].append([r, np.deg2rad(deg), v])
    if frame is not None:
        frames.append(frame)
    if not frames or not any(f["points"] for f in frames):
        raise ValueError("No framed measurements")
    origin = next((f["time"] for f in frames if f["time"] is not None), None)
    for f in frames:
        f["time"] = (
            (f["time"] - origin).total_seconds() if f["time"] is not None else None
        )
        f["points"] = np.array(f["points"], dtype=float).reshape(-1, 3)
    metadata = {
        "source": Path(path).name,
        "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "frames": len(frames),
        "records": sum(len(f["points"]) for f in frames),
        "incomplete_frames": sum(not f["complete"] for f in frames),
        "rejected_rows": rejected,
    }
    return frames, metadata


def default_source():
    return (
        Path(__file__).resolve().parents[2]
        / "a100-indoor-cart-mti/data/targets/2026_08_02_16_19_35_Tatget_CTLRR-300-V2_1_RawTarget.csv"
    )


def xy(polar):
    return polar[:, 0, None] * np.column_stack(
        (np.cos(polar[:, 1]), np.sin(polar[:, 1]))
    )
