"""Export framed target logs to binary PCD and ROS2 PointCloud2 bags."""

import argparse
import csv
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
from rosbags.rosbag2 import Writer
from rosbags.typesys import Stores, get_typestore

FIELDS = ("x", "y", "z", "radial_speed", "range", "angle", "object_id")
DTYPE = np.dtype(
    [(field, "<u4" if field == "object_id" else "<f4") for field in FIELDS]
)
MSGTYPE = "sensor_msgs/msg/PointCloud2"


def read_frames(path):
    frames, current, columns = [], None, None
    with Path(path).open(encoding="utf-8-sig", newline="") as source:
        for row in csv.reader(source):
            if not row:
                continue
            if row[0] == "ObjId":
                if len(set(row)) != len(row):
                    raise ValueError("Duplicate target column names")
                columns = {name: i for i, name in enumerate(row)}
                if not {"ObjId", "range", "angle", "speed"} <= columns.keys():
                    raise ValueError("Missing target columns")
            elif row[0] == "START":
                if current is not None:
                    frames.append(current)
                current = {
                    "device_frame": row[1] if len(row) > 1 else "",
                    "stamp": None,
                    "complete": False,
                    "rejected_rows": 0,
                    "points": [],
                }
            elif row[0] == "END" and current is not None:
                current["complete"] = True
                frames.append(current)
                current = None
            elif current is not None and "---->" in row[0]:
                if current["stamp"] is not None:
                    raise ValueError("Multiple capture timestamps in one frame")
                current["stamp"] = datetime.strptime(
                    row[0].replace("---->", "").strip(), "%Y/%m/%d %H:%M:%S:%f"
                )
            elif current is not None:
                try:
                    float(row[0])
                except ValueError:
                    continue  # diagnostic/status rows
                if columns is None:
                    raise ValueError("Numeric target before column header")
                try:
                    obj, r, angle, speed = [
                        float(row[columns[k]])
                        for k in ("ObjId", "range", "angle", "speed")
                    ]
                    if (
                        not np.isfinite([obj, r, angle, speed]).all()
                        or r < 0
                        or obj != int(obj)
                        or not 0 <= obj <= 2**32 - 1
                    ):
                        raise ValueError("Invalid target")
                    current["points"].append((r, angle, speed, int(obj)))
                except (ValueError, IndexError):
                    current["rejected_rows"] += 1
    if current is not None:
        frames.append(current)
    if not frames or columns is None:
        raise ValueError("No framed target log")
    return frames


def cloud(points):
    result = np.zeros(len(points), dtype=DTYPE)
    if points:
        r, angle, speed, obj = np.asarray(points, dtype=float).T
        radians = np.deg2rad(angle)
        result["x"] = r * np.cos(radians)
        result["y"] = r * np.sin(radians)
        result["radial_speed"] = speed
        result["range"] = r
        result["angle"] = angle
        result["object_id"] = obj.astype(np.uint32)
    if not all(np.isfinite(result[field]).all() for field in FIELDS):
        raise ValueError("Measurement cannot be represented in float32")
    return result


def stamps(frames):
    if any(f["stamp"] is None for f in frames):
        raise ValueError("Every frame needs a capture timestamp; no invented timing")
    first = frames[0]["stamp"]
    result = []
    for frame in frames:
        delta = frame["stamp"] - first
        ns = (
            delta.days * 86400 * 1000000 + delta.seconds * 1000000 + delta.microseconds
        ) * 1000 + 10**9
        if result and ns <= result[-1]:
            raise ValueError(
                "Non-increasing capture timestamps; split the log into monotonic sessions"
            )
        if ns // 10**9 > 2**31 - 1:
            raise ValueError("Duration exceeds ROS Time seconds range")
        result.append(ns)
    return result


def write_pcd(path, points):
    header = (
        f"VERSION 0.7\nFIELDS {' '.join(FIELDS)}\nSIZE 4 4 4 4 4 4 4\n"
        f"TYPE F F F F F F U\nCOUNT 1 1 1 1 1 1 1\nWIDTH {len(points)}\nHEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(points)}\nDATA binary\n"
    )
    with Path(path).open("xb") as stream:
        stream.write(header.encode("ascii"))
        stream.write(points.tobytes())


def export(path, destination):
    frames = read_frames(path)
    times = stamps(frames)  # Validate before creating any output.
    # Verify representability before writing a potentially partial dataset.
    for frame in frames:
        cloud(frame["points"])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    pcd_dir = destination / "pcd"
    pcd_dir.mkdir()
    store = get_typestore(Stores.ROS2_HUMBLE)
    types = store.types
    PointField = types["sensor_msgs/msg/PointField"]
    fields = [
        PointField(
            name=name, offset=i * 4, datatype=6 if name == "object_id" else 7, count=1
        )
        for i, name in enumerate(FIELDS)
    ]
    String = types["std_msgs/msg/String"]
    metadata = []
    with Writer(destination / "bag", version=9) as writer:
        connection = writer.add_connection("/a100/points", MSGTYPE, typestore=store)
        context = writer.add_connection(
            "/a100/frame_context", "std_msgs/msg/String", typestore=store
        )
        for index, (frame, timestamp) in enumerate(zip(frames, times)):
            points = cloud(frame["points"])
            write_pcd(pcd_dir / f"{index:06d}.pcd", points)
            header = types["std_msgs/msg/Header"](
                stamp=types["builtin_interfaces/msg/Time"](
                    sec=timestamp // 10**9, nanosec=timestamp % 10**9
                ),
                frame_id="a100_radar_planar",
            )
            message = types[MSGTYPE](
                header=header,
                height=1,
                width=len(points),
                fields=fields,
                is_bigendian=False,
                point_step=DTYPE.itemsize,
                row_step=len(points) * DTYPE.itemsize,
                data=np.frombuffer(points.tobytes(), dtype=np.uint8),
                is_dense=True,
            )
            writer.write(connection, timestamp, store.serialize_cdr(message, MSGTYPE))
            meta = {
                "order": index,
                "device_frame": frame["device_frame"],
                "complete": frame["complete"],
                "rejected_rows": frame["rejected_rows"],
                "points": len(points),
                "bag_time_ns": timestamp,
                "source_wall_time": frame["stamp"].isoformat(),
            }
            metadata.append(meta)
            writer.write(
                context,
                timestamp,
                store.serialize_cdr(String(json.dumps(meta)), String.__msgtype__),
            )
    with Path(path).open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    report = {
        "source": Path(path).name,
        "sha256": digest,
        "frames": metadata,
        "time_policy": "Recorded wall-time deltas from first frame + 1s. No timezone/UTC or hardware clock claim.",
        "coordinate_policy": "x=range*cos(angle), y=range*sin(angle), z=0. x forward, y left relative to published polar convention; z is unmeasured.",
        "fields": "angle in degrees; radial_speed in m/s is not Cartesian velocity; object_id is original per-frame identifier, not a proven persistent identity.",
    }
    (destination / "manifest.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("capture", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    result = export(args.capture, args.destination)
    print(f"Exported {len(result['frames'])} frames into {args.destination}")
