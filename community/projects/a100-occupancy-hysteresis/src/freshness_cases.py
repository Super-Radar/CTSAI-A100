"""Clock-bound freshness under declared transport injections; no hardware timing claim."""

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path

import numpy as np

from application import CaptureClock, OccupancyApplication
from application_cases import PARAMETERS, write_csv
from logio import load

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261022
POLL = 0.05


def scenarios(times):
    rng = np.random.default_rng(SEED)
    nominal = times + 0.05
    queue = np.where((times >= 4) & (times < 6), 6 + (times - 4) * 0.02, nominal)
    return [
        ("normal_50ms", nominal, CaptureClock(10000, 0, 0.01), True),
        ("constant_1s_delay", times + 1, CaptureClock(10000, 0, 0.01), True),
        ("queue_release", queue, CaptureClock(10000, 0, 0.01), True),
        (
            "jitter_0_to_400ms",
            times + rng.uniform(0, 0.4, len(times)),
            CaptureClock(10000, 0, 0.01),
            True,
        ),
        ("conservative_300ms_uncertainty", nominal, CaptureClock(10000, 0, 0.3), True),
        # Deliberately incorrect synchronization: demonstrate an undetectable
        # assumption violation, never count this as a successful guard case.
        (
            "incorrect_anchor_negative_control",
            times + 1,
            CaptureClock(10000, 0.95, 0.01),
            False,
        ),
    ]


def replay(frames, arrivals, capture_clock, poll_period=POLL):
    end = float(max(arrivals) + 0.5)
    events = [(float(t), 0, i) for i, t in enumerate(arrivals)]
    events += [(float(t), 1, None) for t in np.arange(0, end, poll_period)]
    events.append((end, 1, None))
    events.sort(key=lambda e: (e[0], e[1]))
    apps = {
        "receipt_only": OccupancyApplication(**PARAMETERS),
        "capture_bound": OccupancyApplication(
            capture_clock=capture_clock, **PARAMETERS
        ),
    }
    accepted_capture = dict.fromkeys(apps)
    traces, receipts = [], []
    counts = {name: Counter() for name in apps}
    accepted_old = dict.fromkeys(apps, 0)
    fresh_rejected = dict.fromkeys(apps, 0)
    for now, kind, index in events:
        for app in apps.values():
            app.poll(now)
        if kind == 0:
            frame = frames[index]
            actual_capture = frame["time"]
            actual_age = now - actual_capture
            for name, app in apps.items():
                result = app.observe(
                    now,
                    actual_capture + 10000,
                    frame["points"],
                    complete=frame["complete"],
                )
                counts[name][result["reason"]] += 1
                if result["accepted"]:
                    accepted_capture[name] = actual_capture
                    accepted_old[name] += int(actual_age > PARAMETERS["max_gap"])
                elif actual_age <= PARAMETERS["max_gap"]:
                    fresh_rejected[name] += 1
                receipts.append(
                    {
                        "method": name,
                        "arrival_s": now,
                        "source_order": index,
                        "true_capture_s": actual_capture,
                        "true_age_s": actual_age,
                        **result,
                    }
                )
        traces.append(
            {
                "time_s": now,
                **{name: app.machine.state for name, app in apps.items()},
                **{name + "_source_time": accepted_capture[name] for name in apps},
            }
        )
    stats = {
        name: {
            "occupied_s": 0.0,
            "clear_s": 0.0,
            "unknown_s": 0.0,
            "known_beyond_true_capture_deadline_s": 0.0,
            "stale_known_at_observed_ticks": 0,
        }
        for name in apps
    }
    for row in traces:
        for name in apps:
            source_time = row[name + "_source_time"]
            if (
                row[name] is not None
                and source_time is not None
                and row["time_s"] - source_time > PARAMETERS["max_gap"] + 1e-9
            ):
                stats[name]["stale_known_at_observed_ticks"] += 1
    for row, following in pairwise(traces):
        start, stop = row["time_s"], following["time_s"]
        for name in apps:
            state = row[name]
            key = "unknown_s" if state is None else "occupied_s" if state else "clear_s"
            stats[name][key] += stop - start
            if state is not None:
                deadline = row[name + "_source_time"] + PARAMETERS["max_gap"]
                stats[name]["known_beyond_true_capture_deadline_s"] += max(
                    0, stop - max(start, deadline)
                )
    for name in apps:
        stats[name].update(
            {
                "packet_reasons": dict(counts[name]),
                "accepted_old_packets": accepted_old[name],
                "rejected_physically_fresh_packets": fresh_rejected[name],
            }
        )
    return (
        {"duration_s": end, "packets": len(frames), "methods": stats},
        traces,
        receipts,
    )


def main():
    source = ROOT.parent / "a100-indoor-cart-mti"
    with (source / "data/manifest.csv").open(encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    entry = next(
        r
        for r in manifest
        if r["scenario"] == "stationary_person_3m"
        and r["data_type"] == "raw_target_csv"
    )
    path = source / entry["relative_path"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Source SHA mismatch")
    frames, metadata = load(path)
    if metadata["rejected_rows"] or any(
        not f["complete"] or f["time"] is None for f in frames
    ):
        raise ValueError("Benchmark requires complete valid source frames")
    times = np.array([f["time"] for f in frames])
    if np.any(np.diff(times) <= 0):
        raise ValueError("Benchmark source times must increase")
    reports, all_traces, all_receipts = [], [], []
    for name, arrivals, clock, valid in scenarios(times):
        report, traces, receipts = replay(frames, arrivals, clock)
        report.update(
            {"name": name, "clock": asdict(clock), "clock_bound_contains_truth": valid}
        )
        # A direct check against the known injected capture timeline, not
        # against the mapping function used by the application.
        if valid and report["methods"]["capture_bound"]["accepted_old_packets"]:
            raise RuntimeError(
                "Accepted stale packet despite a valid supplied clock bound"
            )
        if (
            valid
            and report["methods"]["capture_bound"]["stale_known_at_observed_ticks"]
        ):
            raise RuntimeError("Stale visible state at a polled event")
        reports.append(report)
        all_traces.extend({"case": name, **row} for row in traces)
        all_receipts.extend({"case": name, **row} for row in receipts)
    result = {
        "input": metadata,
        "parameters": PARAMETERS,
        "poll_period_s": POLL,
        "seed": SEED,
        "cases": reports,
        "limits": "Physical capture times and transport delays are simulation truth over public measured point frames. Clock mappings are supplied externally, not inferred or measured on hardware. Incorrect-bound negative control is expected to fail. Known beyond deadline integrates sample-and-hold display state, including polling latency. Receipt-time dwell is retained and is not a proof of continuous physical occupancy.",
    }
    out = ROOT / "results/freshness"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    write_csv(out / "timeline.csv", all_traces)
    write_csv(out / "receipts.csv", all_receipts)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), layout="constrained")
    x = np.arange(len(reports))
    for offset, method in enumerate(("receipt_only", "capture_bound")):
        axes[0].bar(
            x + (offset - 0.5) * 0.35,
            [r["methods"][method]["accepted_old_packets"] for r in reports],
            width=0.35,
            label=method,
        )
        axes[1].bar(
            x + (offset - 0.5) * 0.35,
            [r["methods"][method]["unknown_s"] / r["duration_s"] for r in reports],
            width=0.35,
            label=method,
        )
    labels = [r["name"].replace("_", "\n") for r in reports]
    axes[0].set(
        ylabel="Accepted packets older than 0.25 s",
        title="Declared clock bounds; last case intentionally violates the assumption",
    )
    axes[1].set(ylabel="Unknown fraction", ylim=(0, 1.05))
    for ax in axes:
        ax.set_xticks(x, labels, fontsize=8)
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
    fig.savefig(out / "freshness-tradeoff.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps(
            [{"name": r["name"], "methods": r["methods"]} for r in reports], indent=2
        )
    )


if __name__ == "__main__":
    main()
