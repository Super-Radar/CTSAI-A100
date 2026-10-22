"""Eight replay cases: five measured sessions and three explicit transport faults."""

import argparse
import csv
import hashlib
import json
import platform
from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from application import OccupancyApplication, PolarROI
from logio import load
from study import Occupancy

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = {
    "enter_count": 1,
    "exit_count": 0,
    "enter_seconds": 0.3,
    "exit_seconds": 0.7,
    "max_gap": 0.25,
}
METHODS = ("instant", "receipt_hysteresis", "ordered_application")


def packets_from_frames(frames):
    packets = []
    previous = -float("inf")
    for order, frame in enumerate(frames):
        time = frame["time"]
        if time is None or not np.isfinite(time) or time <= previous:
            raise ValueError(
                "Source replay requires strictly increasing valid timestamps"
            )
        previous = time
        packets.append(
            {
                "arrival": float(time),
                "device": float(time),
                "epoch": 0,
                "order": order,
                "points": frame["points"],
                "complete": frame["complete"],
            }
        )
    return packets


def run_case(packets, *, restart_times=(), poll_period=0.05, end=None, label=None):
    """Deterministic simulated arrival clock; no wall-clock sleeps or networking.

    Simultaneous events are ordered restart, packet (stable input order), poll.
    All three methods receive exactly the same packets and timer ticks.
    """
    if not packets or not np.isfinite(poll_period) or poll_period <= 0:
        raise ValueError("Packets and positive finite poll period required")
    arrivals = [p["arrival"] for p in packets]
    if not np.isfinite(arrivals).all() or min(arrivals) < 0:
        raise ValueError("Finite nonnegative arrival times required")
    end = max(arrivals) + 0.5 if end is None else end
    if not np.isfinite(end) or end < max(arrivals):
        raise ValueError("Replay end must include every packet")
    if any(not np.isfinite(t) or t < 0 or t > end for t in restart_times):
        raise ValueError("Restart outside replay interval")
    queue = [(float(t), 0, None) for t in restart_times]
    queue += [(p["arrival"], 1, p) for p in packets]
    queue += [(float(t), 2, None) for t in np.arange(0, end, poll_period)]
    queue.append((float(end), 2, None))
    queue.sort(key=lambda item: (item[0], item[1]))
    app = OccupancyApplication(**PARAMETERS)
    plain = Occupancy(**PARAMETERS)
    roi = PolarROI()
    instant = None
    last_instant = None
    traces, receipts = [], []
    for time, kind, packet in queue:
        if last_instant is not None and time - last_instant > PARAMETERS["max_gap"]:
            instant = None
        plain.value_at(time)
        app.poll(time)
        if kind == 0:
            app.start_epoch(time)
        elif kind == 1:
            result = app.observe(
                time,
                packet["device"],
                packet["points"],
                epoch=packet["epoch"],
                complete=packet["complete"],
            )
            count = roi.count(packet["points"])
            if packet["complete"]:
                plain.update(time, count)
                instant = count >= PARAMETERS["enter_count"]
                last_instant = time
            else:
                plain.update(None, None)
                instant, last_instant = None, None
            receipts.append(
                {
                    "arrival_s": time,
                    "source_order": packet["order"],
                    "device_time_s": packet["device"],
                    "packet_epoch": packet["epoch"],
                    **result,
                }
            )
        traces.append(
            {
                "time_s": time,
                "instant": instant,
                "receipt_hysteresis": plain.state,
                "ordered_application": app.machine.state,
            }
        )
    totals = {
        name: {"occupied_s": 0.0, "clear_s": 0.0, "unknown_s": 0.0} for name in METHODS
    }
    common_seconds = 0.0
    disagreements = dict.fromkeys(METHODS, 0.0)
    for a, b in pairwise(traces):
        duration = b["time_s"] - a["time_s"]
        for name in METHODS:
            state = a[name]
            key = "unknown_s" if state is None else "occupied_s" if state else "clear_s"
            totals[name][key] += duration
        if all(a[name] is not None for name in METHODS):
            common_seconds += duration
            if label is not None:
                for name in METHODS:
                    disagreements[name] += duration * (a[name] != label)
    report = {
        "received_packets": len(packets),
        "duration_s": end,
        "poll_period_s": poll_period,
        "methods": totals,
        "common_known_s": common_seconds,
        "session_label": label,
        "disagreement_s_on_common_known": disagreements if label is not None else None,
        "packet_reasons": dict(Counter(r["reason"] for r in receipts)),
        "state_events": app.events,
    }
    return report, traces, receipts


def build_cases(sessions):
    scenes = [
        "empty_room_person_session",
        "stationary_person_3m",
        "approaching_cart",
        "receding_cart",
        "receding_person_short",
    ]
    cases = []
    for scene in scenes:
        packets = packets_from_frames(sessions[scene][0])
        label = False if scene == scenes[0] else True if scene == scenes[1] else None
        cases.append(
            {
                "name": scene,
                "source_scene": scene,
                "packets": packets,
                "label": label,
                "injection": None,
                "restart_times": [],
            }
        )
    original = packets_from_frames(sessions[scenes[1]][0])
    # Fixed scenario definitions, not selected by successful application output.
    cases.append(
        {
            "name": "injected_silence",
            "source_scene": scenes[1],
            "packets": [p.copy() for p in original if not 4 <= p["arrival"] < 6],
            "label": None,
            "restart_times": [],
            "injection": {"type": "remove_packets", "arrival_interval_s": [4, 6]},
        }
    )
    delayed = [p.copy() for p in original]
    delayed[30]["arrival"] = original[40]["arrival"] + 0.001
    duplicate_arrivals = []
    for repeat in range(6):
        duplicate = original[20].copy()
        duplicate["arrival"] = original[-1]["arrival"] + 0.4 + repeat * 0.1
        duplicate_arrivals.append(duplicate["arrival"])
        delayed.append(duplicate)
    cases.append(
        {
            "name": "injected_late_and_duplicate",
            "source_scene": scenes[1],
            "packets": delayed,
            "label": None,
            "restart_times": [],
            "injection": {
                "type": "delay_and_duplicate",
                "delayed_source_order": 30,
                "delayed_arrival_s": delayed[30]["arrival"],
                "duplicate_source_order": 20,
                "duplicate_arrivals_s": duplicate_arrivals,
            },
        }
    )
    reset = original[100]["arrival"]
    restarted = [
        {
            **p,
            "device": p["device"] - reset if p["order"] >= 100 else p["device"],
            "epoch": 1 if p["order"] >= 100 else 0,
        }
        for p in original
    ]
    old_epoch = original[90].copy()
    old_epoch["arrival"] = reset + 0.01
    restarted.append(old_epoch)
    cases.append(
        {
            "name": "injected_source_restart",
            "source_scene": scenes[1],
            "packets": restarted,
            "label": None,
            "restart_times": [reset],
            "injection": {
                "type": "explicit_restart",
                "source_order": 100,
                "restart_s": reset,
                "old_epoch_packet_order": 90,
                "old_epoch_arrival_s": old_epoch["arrival"],
            },
        }
    )
    return cases


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(
            {key: "unknown" if value is None else value for key, value in row.items()}
            for row in rows
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT.parent / "a100-indoor-cart-mti"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results/application")
    args = parser.parse_args()
    with (args.source / "data/manifest.csv").open(encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    scenes = (
        "empty_room_person_session",
        "stationary_person_3m",
        "approaching_cart",
        "receding_cart",
        "receding_person_short",
    )
    sessions = {}
    for scene in scenes:
        row = next(
            r
            for r in manifest
            if r["scenario"] == scene and r["data_type"] == "raw_target_csv"
        )
        path = args.source / row["relative_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Source SHA mismatch: {path.name}")
        sessions[scene] = load(path)
    cases = build_cases(sessions)
    reports, all_traces, all_receipts = [], [], []
    for case in cases:
        report, traces, receipts = run_case(
            case["packets"], restart_times=case["restart_times"], label=case["label"]
        )
        report.update(
            {
                "name": case["name"],
                "source_scene": case["source_scene"],
                "injection": case["injection"],
            }
        )
        reports.append(report)
        all_traces.extend({"case": case["name"], **row} for row in traces)
        all_receipts.extend({"case": case["name"], **row} for row in receipts)
    report = {
        "sources": {scene: metadata for scene, (_, metadata) in sessions.items()},
        "parameters": PARAMETERS,
        "roi": asdict(PolarROI()),
        "cases": reports,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "system": platform.system(),
        },
        "interpretation": "Original measurements are reused. Receipt clocks are simulated from log times. Faults are explicit injections, not observed sensor failures. Duration totals integrate displayed states between event/timer ticks. Only two session-level weak labels are used; movement cases have no occupancy truth. No person classification, entry/exit accuracy or transport-latency guarantee.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(args.output / "timeline.csv", all_traces)
    write_csv(args.output / "receipts.csv", all_receipts)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), layout="constrained")
    for ax, case in zip(axes.flat, cases):
        rows = [r for r in all_traces if r["case"] == case["name"]]
        for offset, name in enumerate(METHODS):
            # Unknown has a visible lane; it is never plotted as clear.
            values = [
                (-1 if r[name] is None else int(r[name])) + offset * 0.06 for r in rows
            ]
            ax.step(
                [r["time_s"] for r in rows],
                values,
                where="post",
                alpha=0.75,
                label=name,
            )
        ax.set(
            title=case["name"],
            xlabel="Simulated receipt time (s)",
            yticks=[-1, 0, 1],
            yticklabels=["unknown", "clear", "occupied"],
            ylim=(-1.15, 1.3),
        )
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(args.output / "eight-cases.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps(
            [
                {
                    "name": r["name"],
                    "duration_s": r["duration_s"],
                    "methods": r["methods"],
                    "packet_reasons": r["packet_reasons"],
                }
                for r in reports
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
