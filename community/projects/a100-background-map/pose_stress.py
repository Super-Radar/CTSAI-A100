"""Frozen-map sensitivity to declared pose offsets, not new moved-radar data."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from background import BackgroundMap
from benchmark import polar
from logio import load

ROOT = Path(__file__).resolve().parent
CASES = (
    [("unchanged", 0, 0)]
    + [(f"yaw_{yaw:+g}deg", 0, yaw) for yaw in (-5, -2.5, -1, 1, 2.5, 5)]
    + [(f"lateral_{offset:+g}m", offset, 0) for offset in (-0.1, -0.05, 0.05, 0.1)]
)


def transform(points, lateral_m=0, yaw_degrees=0):
    """Sensor translation along original +y, followed by sensor yaw about +z."""
    points = np.asarray(points, dtype=float)
    if lateral_m == 0 and yaw_degrees == 0:
        return points.copy()
    angle = np.deg2rad(points[:, 1])
    x, y = points[:, 0] * np.cos(angle), points[:, 0] * np.sin(angle) - lateral_m
    yaw = np.deg2rad(yaw_degrees)
    new_x = np.cos(yaw) * x + np.sin(yaw) * y
    new_y = -np.sin(yaw) * x + np.cos(yaw) * y
    return np.column_stack(
        (np.hypot(new_x, new_y), np.rad2deg(np.arctan2(new_y, new_x)))
    )


def frozen_keep(model, points):
    # Do not clip an out-of-view angle into a learned edge cell. Such transformed
    # samples are retained here and counted separately, not claimed detectable.
    in_view = np.abs(points[:, 1]) <= 90
    keep = np.ones(len(points), dtype=bool)
    keep[in_view] = model.keep(points[in_view])
    return keep, int((~in_view).sum())


def evaluate(model, frames, lateral, yaw):
    metrics = {
        "frames": len(frames),
        "points": 0,
        "retained": 0,
        "decision_flips": 0,
        "original_roi_frames": 0,
        "original_roi_frames_retained": 0,
        "outside_view": 0,
    }
    rows = []
    for index, points in enumerate(frames):
        baseline = model.keep(points)
        keep, outside = frozen_keep(model, transform(points, lateral, yaw))
        # Membership is fixed in original coordinates to compare the same points,
        # not a different set selected by a shifted sensor-frame ROI.
        roi = (
            (points[:, 0] >= 2.5)
            & (points[:, 0] <= 3.1)
            & (points[:, 1] >= -20)
            & (points[:, 1] <= 5)
        )
        row = {
            "order": index,
            "points": len(points),
            "retained": int(keep.sum()),
            "decision_flips": int(np.count_nonzero(keep != baseline)),
            "original_roi_frames": int(roi.any()),
            "original_roi_frames_retained": int((roi & keep).any()),
            "outside_view": outside,
        }
        rows.append(row)
        for key, value in row.items():
            if key != "order":
                metrics[key] += value
    return metrics, rows


def main():
    source = ROOT.parent / "a100-indoor-cart-mti"
    with (source / "data/manifest.csv").open(encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    sessions, sources = {}, {}
    for scene in ("empty_room_person_session", "stationary_person_3m"):
        entry = next(
            r
            for r in manifest
            if r["scenario"] == scene and r["data_type"] == "raw_target_csv"
        )
        path = source / entry["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError("Source hash mismatch")
        frames, meta = load(path)
        if meta["rejected_rows"] or any(not f["complete"] for f in frames):
            raise ValueError("Complete source frames required")
        sessions[scene], sources[scene] = [polar(f) for f in frames], meta
    empty = sessions["empty_room_person_session"]
    split = len(empty) // 2
    model = BackgroundMap().fit(empty[:split])
    original_hits, original_frames = model.hits.copy(), model.frames
    results, details = [], []
    for name, lateral, yaw in CASES:
        report = {
            "case": name,
            "lateral_m": lateral,
            "yaw_degrees": yaw,
            "sessions": {},
        }
        for scene, frames in (
            ("held_out_empty", empty[split:]),
            ("person", sessions["stationary_person_3m"]),
        ):
            report["sessions"][scene], rows = evaluate(model, frames, lateral, yaw)
            details.extend(dict(case=name, session=scene, **r) for r in rows)
        results.append(report)
    if model.frames != original_frames or not np.array_equal(model.hits, original_hits):
        raise RuntimeError("Sensitivity queries modified the frozen model")
    out = ROOT / "results/pose-stress"
    out.mkdir(parents=True, exist_ok=True)
    summary = {
        "sources": sources,
        "training_frames": split,
        "config": model.config,
        "cases": results,
        "limits": "Declared geometric perturbations of recorded points; no new acquisition, changed visibility, multipath, or hardware pose measurement. Original ROI membership is held fixed. Out-of-view transformed points are retained and counted separately. No labels for false alarms or person detection; no threshold retuning or automatic drift detector.",
    }
    (out / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(details[0]))
        writer.writeheader()
        writer.writerows(details)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), layout="constrained")
    x = np.arange(len(results))
    axes[0].bar(x, [r["sessions"]["held_out_empty"]["retained"] for r in results])
    axes[0].set(
        ylabel="Empty holdout points retained (of 641)",
        title="Same frozen grid; synthetic sensor-pose perturbations",
    )
    axes[1].bar(
        x, [r["sessions"]["person"]["original_roi_frames_retained"] for r in results]
    )
    axes[1].set(ylabel="Original ROI frames retained (of 163)", ylim=(0, 170))
    for ax in axes:
        ax.set_xticks(x, [r["case"].replace("_", "\n") for r in results], fontsize=8)
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "sensitivity.png", dpi=140)
    plt.close(fig)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
