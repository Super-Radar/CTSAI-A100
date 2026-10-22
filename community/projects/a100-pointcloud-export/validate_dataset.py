"""Read exported artifacts through pypcd4 and rosbags, then render bag frames."""

import csv
import hashlib
import json
import platform
import tempfile
from importlib.metadata import version
from pathlib import Path

import numpy as np
from pypcd4 import PointCloud
from rosbags.highlevel import AnyReader

from export_clouds import DTYPE, FIELDS, export

ROOT = Path(__file__).resolve().parent


def main():
    data = ROOT.parent / "a100-indoor-cart-mti"
    rows = list(csv.DictReader((data / "data/manifest.csv").open()))
    results = []
    animation = []
    with tempfile.TemporaryDirectory() as tmp:
        for row in rows:
            if row["data_type"] != "raw_target_csv":
                continue
            source = data / row["relative_path"]
            assert hashlib.sha256(source.read_bytes()).hexdigest() == row["sha256"]
            destination = Path(tmp) / row["scenario"]
            report = export(source, destination)
            count = 0
            points = 0
            context_count = 0
            with AnyReader([destination / "bag"]) as reader:
                for conn, t, raw in reader.messages():
                    msg = reader.deserialize(raw, conn.msgtype)
                    if conn.topic == "/a100/frame_context":
                        context = json.loads(msg.data)
                        assert context == report["frames"][context_count]
                        context_count += 1
                        continue
                    expected = report["frames"][count]
                    assert t == expected["bag_time_ns"]
                    assert msg.header.stamp.sec * 10**9 + msg.header.stamp.nanosec == t
                    assert msg.header.frame_id == "a100_radar_planar"
                    assert (
                        msg.row_step == msg.width * 28 and len(msg.data) == msg.row_step
                    )
                    assert [
                        (f.name, f.offset, f.datatype, f.count) for f in msg.fields
                    ] == [
                        (name, i * 4, 6 if name == "object_id" else 7, 1)
                        for i, name in enumerate(FIELDS)
                    ]
                    pc = PointCloud.from_path(destination / "pcd" / f"{count:06d}.pcd")
                    decoded = np.frombuffer(msg.data.tobytes(), dtype=DTYPE)
                    np.testing.assert_array_equal(pc.pc_data, decoded)
                    assert msg.width == expected["points"]
                    if row["scenario"] == "approaching_cart" and 650 <= count < 730:
                        animation.append((t / 1e9, decoded.copy()))
                    count += 1
                    points += msg.width
            assert count == context_count == len(report["frames"])
            results.append(
                {
                    "scenario": row["scenario"],
                    "source": row["relative_path"],
                    "sha256": row["sha256"],
                    "frames_verified": count,
                    "points_verified": points,
                    "pcd_and_bag_payload_equal": True,
                }
            )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    fig, ax = plt.subplots(figsize=(6, 5), layout="constrained")

    def draw(index):
        ax.clear()
        t, p = animation[index]
        ax.scatter(
            p["y"], p["x"], c=p["radial_speed"], vmin=-1, vmax=1, cmap="coolwarm", s=20
        )
        ax.set(
            xlim=(-10, 10),
            ylim=(0, 18),
            xlabel="Left (m)",
            ylabel="Forward (m)",
            title=f"Bag readback: capture delta {t - 1:.3f} s",
        )
        ax.grid(alpha=0.2)

    movie = FuncAnimation(fig, draw, frames=len(animation), interval=100)
    movie.save(out / "bag-replay.gif", writer=PillowWriter(fps=10))
    draw(len(animation) // 2)
    fig.savefig(out / "bag-readback.png", dpi=140)
    plt.close(fig)
    summary = {
        "python": platform.python_version(),
        "rosbags": version("rosbags"),
        "pypcd4": version("pypcd4"),
        "sessions": results,
        "replay_frames": len(animation),
        "validation": "Independent PCD and bag readers on every frame; GIF renders deserialized bag data at fixed 10fps, not hardware/ROS2/RViz execution.",
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
