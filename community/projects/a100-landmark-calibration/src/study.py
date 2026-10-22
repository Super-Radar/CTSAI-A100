"""Robust planar rigid alignment of paired landmarks; synthetic calibration study."""

import numpy as np

from logio import xy


def rigid(source, target):
    source, target = np.asarray(source, float), np.asarray(target, float)
    if (
        source.ndim != 2
        or source.shape[1:] != (2,)
        or target.shape != source.shape
        or len(source) < 3
    ):
        raise ValueError("At least three paired 2D landmarks required")
    if not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError("Nonfinite coordinates")
    x, y = source - source.mean(0), target - target.mean(0)
    # Conservatively reject collinear layouts, even though some 2D rotations
    # can be determined from two distinct points; this study needs spatial support.
    if np.linalg.matrix_rank(x) < 2 or np.linalg.matrix_rank(y) < 2:
        raise ValueError("Degenerate landmark geometry")
    u, _, vt = np.linalg.svd(x.T @ y)
    correction = np.diag([1.0, np.linalg.det(vt.T @ u.T)])
    rotation = vt.T @ correction @ u.T
    translation = target.mean(0) - rotation @ source.mean(0)
    return rotation, translation


def calibrate(source, target, threshold=0.12, seed=0):
    source, target = np.asarray(source, float), np.asarray(target, float)
    rigid(source, target)
    if not np.isfinite(threshold) or threshold <= 0:
        raise ValueError("Positive finite threshold required")
    rng = np.random.default_rng(seed)
    best = (-1, -np.inf)
    mask = None
    for _ in range(200):
        ids = rng.choice(len(source), 3, replace=False)
        try:
            r, t = rigid(source[ids], target[ids])
        except ValueError:
            continue
        error = np.linalg.norm(source @ r.T + t - target, axis=1)
        selected = error <= threshold
        score = (int(selected.sum()), -float(np.minimum(error, threshold).sum()))
        if score > best:
            best, mask = score, selected
    if mask is None or mask.sum() < 3:
        raise ValueError("No consensus")
    # The returned mask must describe the returned transform, not its seed.
    for _ in range(50):
        if mask.sum() < 3:
            raise ValueError("Consensus collapsed during refinement")
        r, t = rigid(source[mask], target[mask])
        selected = np.linalg.norm(source @ r.T + t - target, axis=1) <= threshold
        if np.array_equal(selected, mask):
            return r, t, mask
        mask = selected
    raise ValueError("Consensus refinement did not converge")


def experiment(frames, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    source = xy(next(f["points"] for f in frames if len(f["points"]) >= 8))
    angle = np.deg2rad(12)
    truth_r = np.array(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
    )
    truth_t = np.array([0.4, -0.2])
    clean = source @ truth_r.T + truth_t
    rng = np.random.default_rng(20261022)
    # Evaluation coordinates are independent of fitting coordinates.
    heldout = rng.uniform([-2, -4], [15, 4], (100, 2))
    expected = heldout @ truth_r.T + truth_t
    errors = {"all_pairs": [], "consensus": []}
    for trial in range(40):
        measured = clean + rng.normal(0, 0.02, clean.shape)
        ids = rng.choice(len(source), max(1, len(source) // 4), replace=False)
        measured[ids] += rng.normal(0, 2, (len(ids), 2))
        fits = [rigid(source, measured), calibrate(source, measured, seed=trial)[:2]]
        for name, (r, t) in zip(errors, fits):
            errors[name].append(
                float(
                    np.sqrt(
                        np.mean(np.sum((heldout @ r.T + t - expected) ** 2, axis=1))
                    )
                )
            )
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for name, values in errors.items():
        ax.plot(sorted(values), label=name)
    ax.set(
        xlabel="Sorted synthetic trial",
        ylabel="Held-out position RMSE (m)",
        title="Paired-landmark calibration: approximately 25% corrupted pairs",
    )
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(out / "comparison.png", dpi=150)
    plt.close(fig)
    return {
        "landmarks": len(source),
        "outliers_per_trial": max(1, len(source) // 4),
        "rotation_deg": 12,
        "translation_m": truth_t.tolist(),
        "noise_sigma_m": 0.02,
        "median_heldout_rmse_m": {k: float(np.median(v)) for k, v in errors.items()},
        "trial_rmse_m": errors,
        "caveat": "Only A100 point geometry is real; paired landmarks, transform, correspondence errors and validation truth are synthetic",
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
