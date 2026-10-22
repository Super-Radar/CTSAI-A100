"""Conditional polar measurement uncertainty; not a sensor calibration."""

import argparse
import csv
import hashlib
import json
import math
import platform
from pathlib import Path

import numpy as np


def moments(r, theta, sigma_r, sigma_theta):
    """Exact and first-order Cartesian moments for independent Gaussian R, T.

    Angles are radians. X = R cos(T), Y = R sin(T). Gaussian R is unbounded;
    near-zero range results are mathematical sensitivity cases, not physical fits.
    """
    if not all(math.isfinite(v) for v in (r, theta, sigma_r, sigma_theta)):
        raise ValueError("Inputs must be finite")
    if r < 0 or sigma_r < 0 or sigma_theta < 0:
        raise ValueError("Range and standard deviations must be nonnegative")
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    q = sigma_theta**2
    # expm1 avoids cancellation for small angular errors.
    one_minus_e = -math.expm1(-q)
    e2 = math.exp(-2 * q)
    radial = r * r * one_minus_e**2 / 2 + sigma_r**2 * (1 + e2) / 2
    tangential = (r * r + sigma_r**2) * (-math.expm1(-2 * q)) / 2
    mean = r * math.exp(-q / 2) * np.array([c, s])
    exact = rotation @ np.diag([radial, tangential]) @ rotation.T
    linear = rotation @ np.diag([sigma_r**2, (r * sigma_theta) ** 2]) @ rotation.T
    return mean, exact, linear


def read_targets(path):
    """Read target records inside START/END frames; preserve file occurrence order.

    This is a measurement sample reader, not a tracking/frame reconstruction API.
    Repeated device frame numbers do not overwrite records. Incomplete final
    frames are counted, and their available records remain available.
    """
    targets = []
    counts = {"frames": 0, "incomplete_frames": 0, "rejected_rows": 0}
    active = False
    header = None
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        for row in csv.reader(stream):
            if not row:
                continue
            if row[0] == "ObjId":
                header = {key: i for i, key in enumerate(row)}
                if not {"ObjId", "range", "angle"} <= header.keys():
                    raise ValueError("Required target columns missing")
                continue
            if row[0] == "START":
                counts["incomplete_frames"] += int(active)
                counts["frames"] += 1
                active = True
                continue
            if row[0] == "END":
                active = False
                continue
            # Timestamps and capture metadata have no numeric object ID.
            if not active or header is None:
                continue
            try:
                float(row[0])
            except ValueError:
                if len(row) >= len(header):
                    counts["rejected_rows"] += 1
                continue
            try:
                obj, r, deg = (
                    float(row[header[k]]) for k in ("ObjId", "range", "angle")
                )
                if not all(math.isfinite(v) for v in (obj, r, deg)):
                    raise ValueError("Nonfinite measurement")
                if obj < 0 or not obj.is_integer() or r < 0:
                    raise ValueError("Invalid ID or range")
            except (ValueError, IndexError):
                counts["rejected_rows"] += 1
                continue
            targets.append((r, math.radians(deg)))
    counts["incomplete_frames"] += int(active)
    if not targets:
        raise ValueError("No valid framed target records")
    counts["valid_targets"] = len(targets)
    return targets, counts


def experiment(samples=100000, seed=20261022):
    if samples < 1000:
        raise ValueError("Use at least 1000 Monte Carlo samples")
    rng = np.random.default_rng(seed)
    results = []
    for r in (1.0, 10.0, 50.0):
        for deg in (1.0, 5.0, 20.0):
            t, sr, st = math.radians(30), 0.1, math.radians(deg)
            mean, exact, linear = moments(r, t, sr, st)
            rr = rng.normal(r, sr, samples)
            tt = rng.normal(t, st, samples)
            xy = np.column_stack((rr * np.cos(tt), rr * np.sin(tt)))
            covariance = np.cov(xy, rowvar=False)
            results.append(
                {
                    "range_m": r,
                    "sigma_angle_deg": deg,
                    "mean_bias_m": float(
                        np.linalg.norm(mean - r * np.array([math.cos(t), math.sin(t)]))
                    ),
                    "exact_major_sigma_m": float(
                        np.sqrt(np.linalg.eigvalsh(exact)[-1])
                    ),
                    "mc_relative_covariance_error": float(
                        np.linalg.norm(covariance - exact) / np.linalg.norm(exact)
                    ),
                    "linear_relative_covariance_error": float(
                        np.linalg.norm(linear - exact) / np.linalg.norm(exact)
                    ),
                }
            )
    return results


def run(
    source, output, sigma_r=0.1, sigma_angle_deg=1.0, samples=100000, seed=20261022
):
    moments(1.0, 0.0, sigma_r, math.radians(sigma_angle_deg))
    targets, counts = read_targets(source)
    major = [
        float(
            np.sqrt(
                np.linalg.eigvalsh(
                    moments(r, t, sigma_r, math.radians(sigma_angle_deg))[1]
                )[-1]
            )
        )
        for r, t in targets
    ]
    data = {
        "source_name": source.name,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "os": platform.system(),
        },
        "assumptions": {
            "sigma_range_m": sigma_r,
            "sigma_angle_deg": sigma_angle_deg,
            "independent_gaussian_errors": True,
            "sensor_calibration": False,
            "coordinate_convention": "X=R*cos(T), Y=R*sin(T); T in radians",
        },
        "counts": counts,
        "range_m_quantiles": np.quantile(
            [r for r, t in targets], [0, 0.5, 0.95, 1]
        ).tolist(),
        "major_sigma_m_quantiles": np.quantile(major, [0, 0.5, 0.95, 1]).tolist(),
        "near_origin_records": sum(r < 5 * sigma_r for r, t in targets),
        "seed": seed,
        "samples_per_case": samples,
        "synthetic_cases": experiment(samples, seed),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(
        json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (output / "synthetic.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(data["synthetic_cases"][0]))
        writer.writeheader()
        writer.writerows(data["synthetic_cases"])
    return data


def plot_results(data, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for angle in (1.0, 5.0, 20.0):
        cases = [c for c in data["synthetic_cases"] if c["sigma_angle_deg"] == angle]
        axes[0].plot(
            [c["range_m"] for c in cases],
            [c["exact_major_sigma_m"] for c in cases],
            "o-",
            label=f"{angle:g} deg",
        )
        axes[1].plot(
            [c["range_m"] for c in cases],
            [100 * c["linear_relative_covariance_error"] for c in cases],
            "o-",
        )
    axes[0].set(ylabel="Exact major-axis standard deviation (m)", yscale="log")
    axes[0].legend(title="Assumed angle std. dev.")
    axes[1].set(ylabel="First-order covariance relative error (%)")
    for ax in axes:
        ax.set(xlabel="Nominal range (m)")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Synthetic sensitivity study: range std. dev. = 0.1 m; NOT sensor accuracy"
    )
    fig.savefig(output / "sensitivity.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    project = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=project.parent
        / "a100-indoor-cart-mti/data/targets/2026_08_02_16_19_35_Tatget_CTLRR-300-V2_1_RawTarget.csv",
    )
    parser.add_argument("--output", type=Path, default=project / "results")
    parser.add_argument("--sigma-range", type=float, default=0.1)
    parser.add_argument("--sigma-angle-deg", type=float, default=1.0)
    parser.add_argument("--samples", type=int, default=100000)
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save synthetic sensitivity plot (requires matplotlib)",
    )
    args = parser.parse_args()
    result = run(
        args.input, args.output, args.sigma_range, args.sigma_angle_deg, args.samples
    )
    if args.plot:
        plot_results(result, args.output)
    print(
        json.dumps(
            {
                "counts": result["counts"],
                "major_sigma_m_quantiles": result["major_sigma_m_quantiles"],
            },
            indent=2,
        )
    )
