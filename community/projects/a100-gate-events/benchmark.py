"""Synthetic event truth and explicitly unlabelled published envelope replay."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from gate import GateCounter

ROOT = Path(__file__).resolve().parent


def run(series, **kwargs):
    counter = GateCounter(**kwargs)
    events = []
    for t, d in series:
        events.extend(counter.update(float(t), d))
    return events


def epoch_case():
    # Two distinct injected passes repeat the same ID and local clock after a
    # restart. Truth is two passes; an aggregation key must keep that boundary.
    cycle = [(0.0, 1), (0.3, 1), (0.4, -1), (0.7, -1)]
    counter = GateCounter()
    events = []
    for _ in range(2):
        for time, distance in cycle:
            events.extend(counter.update(time, {"a": distance}))
    legacy_keys = {(e["track_id"], e["time_s"], e["direction"]) for e in events}
    epoch_keys = {
        (e["epoch"], e["track_id"], e["time_s"], e["direction"]) for e in events
    }
    if len(events) != 2 or len(epoch_keys) != 2:
        raise RuntimeError("Repeated-time passes lost across epochs")
    return {
        "expected_passes": 2,
        "events": events,
        "deduplicated_without_epoch": len(legacy_keys),
        "deduplicated_with_epoch": len(epoch_keys),
        "limits": "Synthetic reset scenario; epoch is local to this counter instance. Persistent sinks must additionally supply a unique run/source identifier.",
    }


def main():
    times = np.arange(0, 6, 0.1)
    # Independently specified event truth: one pass at 2s, return at 4s.
    distance = np.where(times < 2, 1.0, np.where(times < 4, -1.0, 1.0))
    clean = [(float(t), {"a": float(d)}) for t, d in zip(times, distance)]
    noisy = distance.copy()
    noisy[[8, 14, 26, 32]] *= -1
    jitter = [(float(t), {"a": float(d)}) for t, d in zip(times, noisy)]
    synthetic = []
    for name, series in [("clean", clean), ("short_excursions", jitter)]:
        events = run(series)
        assert [e["direction"] for e in events] == ["negative", "positive"]
        naive = int(
            np.sum(
                np.sign([d["a"] for _, d in series])[1:]
                != np.sign([d["a"] for _, d in series])[:-1]
            )
        )
        synthetic.append(
            {
                "case": name,
                "expected_events": 2,
                "naive_sign_changes": naive,
                "confirmed_events": events,
                "latency_s": [events[0]["time_s"] - 2, events[1]["time_s"] - 4],
            }
        )
    source = ROOT.parent / "a100-indoor-cart-mti/results/motion_trajectory.csv"
    rows = list(csv.DictReader(source.open()))
    measured = []
    for scene in sorted({r["scenario"] for r in rows}):
        selected = [r for r in rows if r["scenario"] == scene]
        gate = 3.5 if scene == "receding_person_short" else 2.0
        series = [
            (float(r["elapsed_s"]), {"envelope": float(r["range_m"]) - gate})
            for r in selected
        ]
        events = run(series, max_gap=2.0)
        last = None
        simple = 0
        for _, d in series:
            value = d["envelope"]
            side = 1 if value > 0.1 else -1 if value < -0.1 else 0
            if side:
                if last is not None and side != last:
                    simple += 1
                last = side
        measured.append(
            {
                "scene": scene,
                "gate_m": gate,
                "samples": len(series),
                "simple_band_changes": simple,
                "confirmed_events": events,
                "ground_truth_available": False,
            }
        )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    report = {
        "parameters": {
            "band_m": 0.1,
            "confirm_s": 0.3,
            "rearm_s": 0.6,
            "max_gap_s": 0.5,
            "real_replay_max_gap_s": 2.0,
        },
        "synthetic": synthetic,
        "clock_reset_aggregation": epoch_case(),
        "published_envelope": measured,
        "source": source.relative_to(ROOT.parent).as_posix(),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "limits": "Real input is a published single detection envelope, not identity tracking or event truth; count changes are diagnostics, not accuracy. Fast real returns inside rearm dwell are intentionally suppressed.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4), layout="constrained")
    ax.step(times, noisy, where="post", label="Injected short excursions")
    ax.axhspan(-0.1, 0.1, color="gray", alpha=0.2, label="Ambiguous strip")
    for e in synthetic[1]["confirmed_events"]:
        ax.axvline(e["time_s"], color="green", linestyle="--")
    ax.set(
        xlabel="Synthetic time (s)",
        ylabel="Signed gate distance (m)",
        title="Two confirmed events; green lines include confirmation delay",
    )
    ax.legend()
    fig.savefig(out / "event-replay.png", dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
