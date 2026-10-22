"""Frozen ROI comparison with explicit unknowns and a no-input polling test."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from logio import load, xy
from study import Occupancy

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT.parent / "a100-indoor-cart-mti"


def narrow_count(points):
    """Published stationary-person ROI, fixed before this comparison."""
    return int(
        np.sum(
            (points[:, 0] >= 2.5)
            & (points[:, 0] <= 3.1)
            & (points[:, 1] >= np.deg2rad(-20))
            & (points[:, 1] <= np.deg2rad(5))
        )
    )


def distribution(states, reference):
    known = [value for value in states if value is not None]
    return {
        "total_frames": len(states),
        "known_frames": len(known),
        "occupied": sum(value is True for value in states),
        "clear": sum(value is False for value in states),
        "unknown": sum(value is None for value in states),
        "disagreements_with_session_label": sum(value != reference for value in known),
        "label_caveat": "Session description only; no frame-level person/ROI annotation.",
    }


def main():
    results = ROOT / "results"
    results.mkdir(exist_ok=True)
    manifest = list(
        csv.DictReader((SOURCE / "data/manifest.csv").open(encoding="utf-8"))
    )
    reports, traces = [], []
    for scenario, reference in [
        ("empty_room_person_session", False),
        ("stationary_person_3m", True),
    ]:
        entry = next(
            row
            for row in manifest
            if row["scenario"] == scenario and row["data_type"] == "raw_target_csv"
        )
        path = SOURCE / entry["relative_path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"]
        frames, metadata = load(path)
        broad = Occupancy()
        narrow = Occupancy(enter_count=1, enter_seconds=0)
        old_states, instant_states, narrow_states = [], [], []
        for index, frame in enumerate(frames):
            point_xy = xy(frame["points"])
            wide_count = int(
                np.sum(
                    (point_xy[:, 0] >= 1)
                    & (point_xy[:, 0] <= 5)
                    & (np.abs(point_xy[:, 1]) <= 2)
                )
            )
            count = narrow_count(frame["points"])
            valid = frame["complete"] and frame["time"] is not None
            old = (
                broad.update(frame["time"], wide_count)
                if valid
                else broad.update(None, None)
            )
            state = (
                narrow.update(frame["time"], count)
                if valid
                else narrow.update(None, None)
            )
            instant = count >= 1 if valid else None
            old_states.append(old)
            instant_states.append(instant)
            narrow_states.append(state)
            traces.append([scenario, index, frame["time"], count, instant, state])
        common = [
            i
            for i in range(len(frames))
            if instant_states[i] is not None and narrow_states[i] is not None
        ]
        reports.append(
            {
                "scenario": scenario,
                "input": metadata,
                "session_label": reference,
                "broad_roi_original": distribution(old_states, reference),
                "narrow_roi_instant": distribution(instant_states, reference),
                "narrow_roi_hysteresis": distribution(narrow_states, reference),
                "common_known_frames": len(common),
                "instant_disagreements_on_common_frames": sum(
                    instant_states[i] != reference for i in common
                ),
                "hysteresis_disagreements_on_common_frames": sum(
                    narrow_states[i] != reference for i in common
                ),
            }
        )

    # A simulated application polls during a complete input outage. No update
    # call is made after 1.0 s until 2.0 s; the watchdog must expire by itself.
    clock = np.arange(0, 3.01, 0.05)
    machine = Occupancy(enter_count=1, enter_seconds=0)
    decisions = []
    for tick, time in enumerate(clock):
        measured = tick % 2 == 0 and not (1.0 < time < 2.0)
        if measured:
            machine.update(float(time), 1)
        decisions.append(machine.value_at(float(time)))
    unknown = [float(time) for time, state in zip(clock, decisions) if state is None]
    assert unknown and 1.25 <= unknown[0] <= 1.30 + 1e-12
    report = {
        "parameters": {
            "range_m": [2.5, 3.1],
            "angle_deg": [-20, 5],
            "enter_count": 1,
            "enter_s": 0,
            "exit_s": 0.7,
            "max_gap_s": 0.25,
        },
        "roi_origin": "a100-indoor-cart-mti published stationary-person ROI; no parameter sweep or fitting here",
        "real_sessions": reports,
        "no_input_watchdog": {
            "last_pre_outage_measurement_s": 1.0,
            "first_unknown_poll_s": unknown[0],
            "resume_measurement_s": 2.0,
            "poll_period_s": 0.05,
            "unknown_polls": len(unknown),
        },
        "limits": "Narrow ROI results characterize detection persistence, not a validated person classifier. Session descriptions are weak proxy labels, not per-frame truth.",
    }
    (results / "presence-validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    with (results / "presence-decisions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "scenario",
                "capture_order",
                "time_s",
                "narrow_roi_count",
                "instant",
                "hysteresis",
            ]
        )
        writer.writerows(
            [["unknown" if value is None else value for value in row] for row in traces]
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), layout="constrained")
    for ax, report_row in zip(axes[:2], reports):
        rows = [row for row in traces if row[0] == report_row["scenario"]]
        ax.step(
            [row[2] for row in rows],
            [row[4] for row in rows],
            where="post",
            alpha=0.6,
            label="Instant narrow ROI",
        )
        ax.step(
            [row[2] for row in rows],
            [np.nan if row[5] is None else int(row[5]) for row in rows],
            where="post",
            label="Hysteresis",
        )
        ax.set(
            title=report_row["scenario"] + " (session label only)",
            ylabel="Detection state",
            ylim=(-0.2, 1.2),
        )
        ax.legend(loc="lower right")
    axes[2].step(
        clock,
        [np.nan if state is None else int(state) for state in decisions],
        where="post",
        label="Polled state",
    )
    axes[2].scatter(unknown, np.full(len(unknown), -0.1), marker="x", label="Unknown")
    axes[2].axvspan(1, 2, alpha=0.15, color="gray", label="No measurement updates")
    axes[2].set(
        xlabel="Time (s)",
        ylabel="Detection state",
        title="Watchdog expires while input is silent",
        ylim=(-0.2, 1.2),
    )
    axes[2].legend(loc="lower right")
    fig.savefig(results / "presence-validation.png", dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
