"""Timestamp-based ROI occupancy debounce with explicit unknown state."""

from itertools import pairwise

import numpy as np

from logio import xy


class Occupancy:
    def __init__(
        self,
        enter_count=2,
        exit_count=0,
        enter_seconds=0.3,
        exit_seconds=0.7,
        max_gap=0.25,
    ):
        if not all(
            np.isfinite(v)
            for v in (enter_count, exit_count, enter_seconds, exit_seconds, max_gap)
        ):
            raise ValueError("Finite parameters required")
        if (
            enter_count != int(enter_count)
            or exit_count != int(exit_count)
            or exit_count < 0
            or enter_count <= exit_count
        ):
            raise ValueError("Integer count thresholds must satisfy enter > exit >= 0")
        if min(enter_seconds, exit_seconds) < 0 or max_gap <= 0:
            raise ValueError("Invalid durations")
        self.enter, self.exit = enter_count, exit_count
        self.on, self.off, self.gap = enter_seconds, exit_seconds, max_gap
        self.reset()

    def reset(self):
        self.state = None
        self.last = None
        self.pending = None
        self.since = None

    def value_at(self, time):
        """Poll freshness even when no new measurement arrives.

        Use the same monotonic time basis as update; polling does not renew
        the last measurement time or advance occupancy confirmation timers.
        """
        if not np.isfinite(time):
            self.reset()
            raise ValueError("Finite polling time required")
        if self.last is not None and (time < self.last or time - self.last > self.gap):
            self.reset()
        return self.state

    def update(self, time, count):
        if time is None or count is None:
            self.reset()
            return None
        if (
            not np.isfinite(time)
            or not np.isfinite(count)
            or count < 0
            or int(count) != count
        ):
            self.reset()
            raise ValueError("Finite timestamp and nonnegative integer count required")
        if self.last is not None and (time <= self.last or time - self.last > self.gap):
            self.reset()
        self.last = time
        desired = True if count >= self.enter else False if count <= self.exit else None
        if desired is None or desired == self.state:
            self.pending = None
            self.since = None
        else:
            if desired != self.pending:
                self.pending = desired
                self.since = time
            duration = self.on if desired else self.off
            if time - self.since >= duration - 1e-12:
                self.state = desired
                self.pending = None
                self.since = None
        return self.state


def experiment(frames, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = np.arange(0, 20, 0.1)
    truth = (times >= 5) & (times < 12)
    counts = np.where(truth, 3, 0)
    counts[[10, 20, 150]] = 3
    counts[[72, 90, 110]] = 0
    counts = counts.astype(object)
    counts[(times >= 8) & (times < 8.5)] = None
    machine = Occupancy()
    states = [machine.update(float(t), c) for t, c in zip(times, counts)]
    known = np.array([s is not None for s in states])
    predictions = np.array([bool(s) for s in states])
    valid = np.array([c is not None for c in counts])
    baseline = np.array([c is not None and c >= 2 for c in counts])
    real = Occupancy()
    real_states = []
    real_counts = []
    for f in frames:
        p = xy(f["points"])
        count = int(np.sum((p[:, 0] >= 1) & (p[:, 0] <= 5) & (np.abs(p[:, 1]) <= 2)))
        real_counts.append(count)
        real_states.append(
            real.update(f["time"], count) if f["complete"] else real.update(None, None)
        )
    import csv

    with (out / "synthetic-decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time_s", "truth", "count", "debounced", "instant"])
        for t, truth_value, count, state, instant in zip(
            times, truth, counts, states, baseline
        ):
            writer.writerow(
                [
                    float(t),
                    int(truth_value),
                    count,
                    "unknown" if state is None else int(state),
                    "unknown" if count is None else int(instant),
                ]
            )
    with (out / "logged-decisions.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["capture_order", "time_s", "roi_count", "state"])
        for index, (frame, count, state) in enumerate(
            zip(frames, real_counts, real_states)
        ):
            writer.writerow(
                [
                    index,
                    frame["time"],
                    count,
                    "unknown" if state is None else int(state),
                ]
            )
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), layout="constrained")
    axes[0].step(times, truth, label="synthetic truth", where="post")
    axes[0].step(
        times,
        [np.nan if s is None else int(s) for s in states],
        label="debounced",
        where="post",
    )
    axes[0].step(
        times,
        [np.nan if c is None else int(c >= 2) for c in counts],
        label="instant threshold",
        alpha=0.5,
        where="post",
    )
    axes[0].set(
        xlabel="Synthetic time (s)",
        ylabel="Occupancy",
        title="Gaps produce unknown, not a remembered decision",
    )
    axes[0].scatter(
        times[~known], np.full(int((~known).sum()), -0.1), marker="x", label="unknown"
    )
    axes[0].legend()
    axes[1].plot(real_counts, label="logged ROI point count")
    axes[1].set(
        xlabel="Capture order",
        ylabel="Points in ROI",
        title="Public log diagnostic; no occupancy ground truth",
    )
    axes[1].legend()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    return {
        "parameters": {
            "enter_count": 2,
            "exit_count": 0,
            "enter_s": 0.3,
            "exit_s": 0.7,
            "max_gap_s": 0.25,
        },
        "synthetic": {
            "observations": len(times),
            "common_known_available_samples": int(np.sum(known & valid)),
            "debounced_wrong_common_samples": int(
                np.sum(predictions[known & valid] != truth[known & valid])
            ),
            "instant_wrong_common_samples": int(
                np.sum(baseline[known & valid] != truth[known & valid])
            ),
            "debounced_unknown_samples": int(np.sum(~known)),
            "instant_unavailable_samples": int(np.sum(~valid)),
            "known_fraction": float(known.mean()),
            "debounced_wrong_known_samples": int(
                np.sum(predictions[known] != truth[known])
            ),
            "instant_wrong_available_samples": int(
                np.sum(baseline[valid] != truth[valid])
            ),
            "debounced_transitions": sum(
                a is not None and b is not None and a != b for a, b in pairwise(states)
            ),
            "instant_transitions": int(np.sum(baseline[1:] != baseline[:-1])),
        },
        "real_log": {
            "roi_m": {"x": [1, 5], "y": [-2, 2]},
            "occupied_samples": sum(s is True for s in real_states),
            "clear_samples": sum(s is False for s in real_states),
            "unknown_samples": sum(s is None for s in real_states),
        },
        "caveat": "ROI detections are not people/vehicle identities. Synthetic delays trade response latency for chatter suppression. No safety/control validation.",
    }


if __name__ == "__main__":
    import argparse
    import json
    import platform
    from pathlib import Path

    from logio import default_source, load

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=default_source())
    parser.add_argument(
        "--output", type=Path, default=Path(__file__).resolve().parents[1] / "results"
    )
    args = parser.parse_args()
    frames, metadata = load(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    result = experiment(frames, args.output)
    result["input"] = metadata
    result["environment"] = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "os": platform.system(),
    }
    result["seed"] = 20261022
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, allow_nan=False))
