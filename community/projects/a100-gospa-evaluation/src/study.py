"""GOSPA p=2, alpha=2 point-set evaluation with explicit error decomposition."""

import numpy as np
from scipy.optimize import linear_sum_assignment

from logio import xy


def points(value):
    value = np.asarray(value, float)
    if value.size == 0:
        value = np.empty((0, 2))
    if value.ndim != 2 or value.shape[1] != 2 or not np.isfinite(value).all():
        raise ValueError("Finite N by 2 point set required")
    if len(np.unique(value, axis=0)) != len(value):
        raise ValueError("Duplicate point locations: provide a set")
    return value


def gospa(reference, estimate, cutoff=2.0):
    reference, estimate = points(reference), points(estimate)
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("Positive finite cutoff required")
    n, m = len(reference), len(estimate)
    pairs = []
    localization = 0.0
    if n and m:
        distance = np.linalg.norm(reference[:, None] - estimate[None, :], axis=2)
        rows, cols = linear_sum_assignment(np.minimum(distance, cutoff) ** 2)
        # At exactly c, pairing and a miss+false have equal cost. Report the
        # latter consistently; the total metric is invariant to this tie.
        pairs = [
            (int(i), int(j)) for i, j in zip(rows, cols) if distance[i, j] < cutoff
        ]
        localization = sum(float(distance[i, j] ** 2) for i, j in pairs)
    missed = (n - len(pairs)) * cutoff**2 / 2
    false = (m - len(pairs)) * cutoff**2 / 2
    return {
        "distance": float(np.sqrt(localization + missed + false)),
        "localization_squared": localization,
        "missed_squared": missed,
        "false_squared": false,
        "matched": len(pairs),
        "pairs": pairs,
    }


def experiment(frames, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20261022)
    scores = {k: [] for k in ("noise_only", "misses", "false_points", "mixed")}
    nn_rmse = []
    removed_duplicates = 0
    for f in frames:
        original = xy(f["points"])
        reference = np.unique(original, axis=0)
        removed_duplicates += len(original) - len(reference)
        if not len(reference):
            continue
        noisy = reference + rng.normal(0, 0.05, reference.shape)
        keep = np.arange(len(reference)) % 3 != 0
        extras = rng.uniform([50, 50], [60, 60], (3, 2))
        estimates = {
            "noise_only": noisy,
            "misses": reference[keep],
            "false_points": np.vstack((reference, extras)),
            "mixed": np.vstack((noisy[keep], extras)),
        }
        for name, estimate in estimates.items():
            scores[name].append(gospa(reference, estimate))
        if keep.any():
            # Shows why one-way nearest-neighbour RMSE ignores missing targets.
            distances = np.linalg.norm(
                reference[keep, None] - reference[None, :], axis=2
            )
            nn_rmse.append(float(np.sqrt(np.mean(np.min(distances, axis=1) ** 2))))
    means = {
        k: {
            field: float(np.mean([s[field] for s in v]))
            for field in (
                "distance",
                "localization_squared",
                "missed_squared",
                "false_squared",
            )
        }
        for k, v in scores.items()
    }
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    bottom = np.zeros(4)
    for field in ("localization_squared", "missed_squared", "false_squared"):
        values = [means[k][field] for k in scores]
        ax.bar(list(scores), values, bottom=bottom, label=field)
        bottom += values
    ax.set(
        ylabel="Mean squared GOSPA component (m²)",
        title="Known synthetic corruption of A100 point snapshots",
    )
    ax.legend()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    return {
        "p": 2,
        "alpha": 2,
        "cutoff_m": 2,
        "evaluated_frames": len(scores["misses"]),
        "duplicate_locations_removed": removed_duplicates,
        "mean_scores": means,
        "misses_only_one_way_nn_rmse_m": float(np.mean(nn_rmse)) if nn_rmse else None,
        "caveat": "Reference is the recorded point set, not physical object truth. This measures injected corruption, not detector accuracy or track identity.",
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
