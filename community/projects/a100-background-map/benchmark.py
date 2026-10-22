"""Chronological empty-room holdout and frozen-map person-session diagnostics."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from background import BackgroundMap
from logio import load

ROOT = Path(__file__).resolve().parent


def polar(frame):
    p = frame["points"]
    return np.column_stack((p[:, 0], np.rad2deg(p[:, 1])))


def main():
    source = ROOT.parent / "a100-indoor-cart-mti"
    manifest = list(
        csv.DictReader((source / "data/manifest.csv").open(encoding="utf-8"))
    )
    sessions = {}
    for scene in ("empty_room_person_session", "stationary_person_3m"):
        row = next(
            r
            for r in manifest
            if r["scenario"] == scene and r["data_type"] == "raw_target_csv"
        )
        path = source / row["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
        frames, meta = load(path)
        assert all(f["complete"] for f in frames)
        sessions[scene] = ([polar(f) for f in frames], meta)
    empty = sessions["empty_room_person_session"][0]
    person = sessions["stationary_person_3m"][0]
    split = len(empty) // 2
    model = BackgroundMap().fit(empty[:split])
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    model.save(out / "background-model.json")
    restored = BackgroundMap.load(out / "background-model.json")
    traces = []
    reports = []
    for scene, frames in [
        ("held_out_empty", empty[split:]),
        ("person_session", person),
    ]:
        total = retained = roi_before = roi_after = 0
        for index, points in enumerate(frames):
            keep = restored.keep(points)
            roi = (
                (points[:, 0] >= 2.5)
                & (points[:, 0] <= 3.1)
                & (points[:, 1] >= -20)
                & (points[:, 1] <= 5)
            )
            total += len(points)
            retained += int(keep.sum())
            roi_before += int(roi.any())
            roi_after += int((roi & keep).any())
            traces.append(
                [
                    scene,
                    index,
                    len(points),
                    int(keep.sum()),
                    bool(roi.any()),
                    bool((roi & keep).any()),
                ]
            )
        reports.append(
            {
                "scene": scene,
                "frames": len(frames),
                "points": total,
                "retained_points": retained,
                "removed_fraction": 1 - retained / total,
                "roi_frames_before": roi_before,
                "roi_frames_after": roi_after,
            }
        )
    # Deliberately contaminated training is a separate negative control.
    contaminated = BackgroundMap().fit(empty[:split] + person)
    bad_roi = sum(
        bool(
            np.any(
                contaminated.keep(p)
                & (p[:, 0] >= 2.5)
                & (p[:, 0] <= 3.1)
                & (p[:, 1] >= -20)
                & (p[:, 1] <= 5)
            )
        )
        for p in person
    )
    report = {
        "training": {
            "scene": "empty_room_person_session",
            "indices": [0, split - 1],
            "frames": split,
        },
        "config": model.config,
        "sources": {s: m for s, (_, m) in sessions.items()},
        "evaluation": reports,
        "contaminated_negative_control": {
            "training_includes_person": True,
            "person_roi_frames_retained": bad_roi,
        },
        "caveat": "Same-session chronological holdout, not independent-site validation. Frame correlation means Wilson lower score is not a calibrated 95% confidence guarantee. Point removal is not object detection accuracy.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "decisions.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scene", "order", "points", "retained", "roi_before", "roi_after"])
        w.writerows(traces)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    im = axes[0].imshow(
        model.lower_bound().T,
        origin="lower",
        aspect="auto",
        extent=[0, 30, -90, 90],
        vmin=0,
        vmax=1,
    )
    axes[0].set(
        xlabel="Range (m)",
        ylabel="Angle (deg)",
        title="Frozen empty-room occupancy score",
    )
    fig.colorbar(im, ax=axes[0])
    for label, style, col in [("Input", "-", 2), ("Retained", "--", 3)]:
        rows = [r for r in traces if r[0] == "person_session"]
        axes[1].plot(
            [r[1] for r in rows], [r[col] for r in rows], style, label=label, alpha=0.7
        )
    axes[1].set(
        xlabel="Person session frame",
        ylabel="Points",
        title="Filtering output, no object labels",
    )
    axes[1].legend()
    fig.savefig(out / "background.png", dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
