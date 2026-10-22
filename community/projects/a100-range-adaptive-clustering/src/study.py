"""Density clustering under fixed and polar-noise-scaled neighbourhoods."""

import numpy as np

from logio import xy


def neighbourhood(
    points,
    adaptive=False,
    eps=0.3,
    sigma_range=0.12,
    sigma_angle=np.pi / 180,
    gate=2.0,
):
    p = np.asarray(points, float)
    if p.ndim != 2 or p.shape[1:] != (2,) or not np.isfinite(p).all():
        raise ValueError("Finite N by 2 coordinates required")
    if not all(np.isfinite(v) and v > 0 for v in (eps, sigma_range, sigma_angle, gate)):
        raise ValueError("Positive finite scales required")
    delta = p[:, None] - p[None, :]
    if not adaptive:
        return np.sum(delta**2, axis=2) <= eps**2
    radius = np.linalg.norm(p, axis=1)
    angle = np.arctan2(p[:, 1], p[:, 0])
    radial = np.column_stack((np.cos(angle), np.sin(angle)))
    tangent = np.column_stack((-np.sin(angle), np.cos(angle)))
    cov = (
        sigma_range**2 * radial[:, :, None] * radial[:, None, :]
        + np.maximum(radius * sigma_angle, sigma_range)[:, None, None] ** 2
        * tangent[:, :, None]
        * tangent[:, None, :]
    )
    pair = cov[:, None] + cov[None, :]
    solved = np.linalg.solve(pair, delta[..., None])[..., 0]
    return np.sum(delta * solved, axis=2) <= gate**2


def density_labels(adj, min_samples=4):
    adj = np.asarray(adj)
    if (
        adj.ndim != 2
        or adj.shape[0] != adj.shape[1]
        or adj.dtype != bool
        or not np.array_equal(adj, adj.T)
        or not np.diag(adj).all()
    ):
        raise ValueError("Symmetric boolean adjacency including self required")
    if min_samples < 1 or int(min_samples) != min_samples:
        raise ValueError("Positive integer min_samples required")
    core = adj.sum(1) >= min_samples
    labels = np.full(len(adj), -1, dtype=int)
    label = 0
    # Expand only through core points; border points cannot bridge two cores.
    for i in range(len(adj)):
        if not core[i] or labels[i] >= 0:
            continue
        queue = [i]
        labels[i] = label
        while queue:
            current = queue.pop()
            for j in np.flatnonzero(adj[current]):
                if labels[j] >= 0:
                    continue
                labels[j] = label
                if core[j]:
                    queue.append(int(j))
        label += 1
    return labels


def pair_scores(labels, truth):
    labels, truth = np.asarray(labels), np.asarray(truth)
    if (
        labels.ndim != 1
        or truth.shape != labels.shape
        or not len(labels)
        or not np.issubdtype(labels.dtype, np.integer)
        or not np.issubdtype(truth.dtype, np.integer)
        or np.any(labels < -1)
        or np.any(truth < -1)
    ):
        raise ValueError("Equal nonempty integer labels required; -1 denotes noise")
    upper = np.triu(np.ones((len(labels), len(labels)), dtype=bool), 1)
    actual = (truth[:, None] == truth[None, :]) & (truth[:, None] >= 0) & upper
    predicted = (labels[:, None] == labels[None, :]) & (labels[:, None] >= 0) & upper
    tp = int((actual & predicted).sum())
    return {
        "precision": tp / int(predicted.sum()) if predicted.any() else 0.0,
        "recall": tp / int(actual.sum()) if actual.any() else 0.0,
        "noise_fraction": float(np.mean(labels < 0)),
    }


def experiment(frames, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(20261022)
    rows = []
    for separation in (1.5, 4.0):
        for radius in (5.0, 15.0, 30.0, 50.0):
            samples = {"fixed": [], "adaptive": []}
            for _ in range(30):
                centers = np.array(
                    [[radius, -separation / 2], [radius, separation / 2]]
                )
                truth = np.repeat([0, 1], 12)
                ideal = np.repeat(centers, 12, axis=0)
                ranges = np.linalg.norm(ideal, axis=1) + rng.normal(0, 0.12, 24)
                angles = np.arctan2(ideal[:, 1], ideal[:, 0]) + rng.normal(
                    0, np.deg2rad(1), 24
                )
                measured = ranges[:, None] * np.column_stack(
                    (np.cos(angles), np.sin(angles))
                )
                for name, adaptive in (("fixed", False), ("adaptive", True)):
                    labels = density_labels(neighbourhood(measured, adaptive))
                    samples[name].append(pair_scores(labels, truth))
            for name, values in samples.items():
                rows.append(
                    {
                        "range_m": radius,
                        "separation_m": separation,
                        "method": name,
                        **{
                            k: float(np.mean([v[k] for v in values])) for k in values[0]
                        },
                    }
                )
    logged = {"fixed": [], "adaptive": []}
    for frame in frames:
        p = xy(frame["points"])
        for name, adaptive in (("fixed", False), ("adaptive", True)):
            labels = density_labels(neighbourhood(p, adaptive))
            logged[name].append(len(set(labels) - {-1}))
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, separation in zip(axes, (1.5, 4.0)):
        for name in ("fixed", "adaptive"):
            selected = [
                r
                for r in rows
                if r["separation_m"] == separation and r["method"] == name
            ]
            ax.plot(
                [r["range_m"] for r in selected],
                [r["recall"] for r in selected],
                "o-",
                label=name + " recall",
            )
            ax.plot(
                [r["range_m"] for r in selected],
                [r["precision"] for r in selected],
                "x--",
                label=name + " precision",
            )
        ax.set(
            xlabel="Synthetic range (m)",
            ylabel="Pairwise score",
            ylim=(0, 1.05),
            title=f"Object separation {separation:g} m",
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    return {
        "synthetic_trials_per_case": 30,
        "points_per_object": 12,
        "sigma_range_m": 0.12,
        "sigma_angle_deg": 1,
        "fixed_eps_m": 0.3,
        "adaptive_gate": 2,
        "min_samples": 4,
        "synthetic_scores": rows,
        "logged_cluster_count_median": {
            k: float(np.median(v)) for k, v in logged.items()
        },
        "logged_cluster_counts": logged,
        "caveat": "Adaptive neighbourhoods can merge nearby objects at long range. Noise scales are assumed. Real logs have no object labels, so only counts are reported.",
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
