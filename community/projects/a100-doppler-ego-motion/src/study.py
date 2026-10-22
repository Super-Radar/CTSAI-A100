"""Static-scatterer Doppler ego-motion model and its observability limits."""

import numpy as np


def fit_velocity(angles, doppler, threshold=0.15, robust=True, seed=0):
    angles, doppler = np.asarray(angles, dtype=float), np.asarray(doppler, dtype=float)
    if angles.ndim != 1 or doppler.shape != angles.shape or len(angles) < 3:
        raise ValueError("At least three aligned bearings and Doppler samples required")
    if (
        not np.isfinite(angles).all()
        or not np.isfinite(doppler).all()
        or not np.isfinite(threshold)
        or threshold <= 0
    ):
        raise ValueError("Finite inputs and positive residual threshold required")
    a = -np.column_stack((np.cos(angles), np.sin(angles)))
    if np.linalg.cond(a) > 100:
        raise ValueError("Bearing geometry does not constrain both velocity axes")
    inliers = np.ones(len(a), dtype=bool)
    if robust:
        rng = np.random.default_rng(seed)
        best = (-1, -np.inf)
        for _ in range(200):
            ids = rng.choice(len(a), 2, replace=False)
            if np.linalg.cond(a[ids]) > 100:
                continue
            v = np.linalg.solve(a[ids], doppler[ids])
            residual = np.abs(a @ v - doppler)
            mask = residual <= threshold
            score = (int(mask.sum()), -float(np.minimum(residual, threshold).sum()))
            if score > best:
                best, inliers = score, mask
        if inliers.sum() < 3 or np.linalg.cond(a[inliers]) > 100:
            raise ValueError("No observable static-scatterer consensus")
    # A least-squares refit can move points across the original RANSAC cutoff.
    # Return a fixed point of refitting and classification, or fail explicitly.
    for _ in range(50):
        if inliers.sum() < 3 or np.linalg.cond(a[inliers]) > 100:
            raise ValueError("No observable consensus after refinement")
        velocity = np.linalg.lstsq(a[inliers], doppler[inliers], rcond=None)[0]
        selected = np.abs(a @ velocity - doppler) <= threshold
        if not robust or np.array_equal(selected, inliers):
            return velocity, inliers, float(np.linalg.cond(a[inliers]))
        inliers = selected
    raise ValueError("Consensus refinement did not converge")


def experiment(frames, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    bearings = next(f["points"][:, 1] for f in frames if len(f["points"]) >= 8)
    truth = np.array([2.0, 0.4])
    rng = np.random.default_rng(20261022)
    errors = {"least_squares": [], "consensus": []}
    for trial in range(40):
        v = -np.column_stack((np.cos(bearings), np.sin(bearings))) @ truth + rng.normal(
            0, 0.03, len(bearings)
        )
        ids = rng.choice(len(v), max(1, len(v) // 4), replace=False)
        v[ids] += rng.choice([-1.0, 1.0], len(ids)) * 2
        for name, robust in (("least_squares", False), ("consensus", True)):
            estimate, _, _ = fit_velocity(bearings, v, robust=robust, seed=trial)
            errors[name].append(float(np.linalg.norm(estimate - truth)))
    rejected = 0
    apparent = []
    for f in frames:
        try:
            vel, mask, cond = fit_velocity(f["points"][:, 1], f["points"][:, 2])
            apparent.append(
                {
                    "velocity": vel.tolist(),
                    "inliers": int(mask.sum()),
                    "condition": cond,
                }
            )
        except ValueError:
            rejected += 1
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for name, values in errors.items():
        ax.plot(sorted(values), label=name)
    ax.set(
        xlabel="Sorted synthetic trial",
        ylabel="Velocity error norm (m/s)",
        title="Injected Doppler on A100 bearings; approximately 25% outliers",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    return {
        "synthetic_truth_mps": truth.tolist(),
        "synthetic_trials": 40,
        "noise_sigma_mps": 0.03,
        "outlier_offset_mps": 2,
        "bearings": len(bearings),
        "outliers_per_trial": max(1, len(bearings) // 4),
        "median_error_mps": {k: float(np.median(v)) for k, v in errors.items()},
        "trial_errors_mps": errors,
        "logged_speed_model_only": {
            "accepted_frames": len(apparent),
            "rejected_frames": rejected,
            "median_apparent_velocity": np.median(
                [x["velocity"] for x in apparent], axis=0
            ).tolist()
            if apparent
            else None,
        },
        "caveat": "Logged speed sign and static-scatterer assumption uncalibrated; apparent fit is not measured ego velocity",
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
