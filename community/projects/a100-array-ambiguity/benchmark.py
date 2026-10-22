"""Published profile geometry plus analytic ULA ambiguity controls."""

import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from ambiguity import diagnose, peak_candidates, steering

ROOT = Path(__file__).resolve().parent


def profile(path):
    text = path.read_text(encoding="utf-8")
    match = re.search(r"\.ant_pos\s*=\s*\{(.*?)\}\s*,\s*\.ant_comps", text, re.DOTALL)
    groups = re.search(r"\.tx_groups\s*=\s*\{([^}]+)\}", text)
    if not match or not groups:
        raise ValueError("Missing explicit profile geometry/TX groups")
    pairs = re.findall(r"\{\s*([-+0-9.eE]+)\s*,\s*([-+0-9.eE]+)\s*\}", match.group(1))
    positions = np.array(pairs, dtype=float)
    enabled = [
        i for i, value in enumerate(groups.group(1).split(",")) if int(value.strip(), 0)
    ]
    if len(enabled) != 1 or positions.shape != (16, 2):
        raise ValueError("This benchmark requires explicit one-TX four-RX profile")
    tx = enabled[0]
    return positions[4 * tx : 4 * tx + 4, 0], tx + 1


def main():
    directory = ROOT.parent / "a100-matlab-adc-signal-processing/config"
    angles = np.arange(-75, 75.01, 0.5)
    reports = []
    maps = []
    source_rows = []
    for index in range(4):
        path = directory / f"sensor_config_init{index}.hxx"
        p, tx = profile(path)
        source_rows.append(
            {
                "file": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "physical_tx": tx,
                "positions_lambda": p.tolist(),
            }
        )
        result = diagnose(p, angles)
        maps.append(result["coherence"])
        finite = result["max_coherence"][np.isfinite(result["max_coherence"])]
        reports.append(
            {
                "profile": index,
                "grid_angles": len(angles),
                "ambiguous_angles": int(result["ambiguous"].sum()),
                "maximum_separated_coherence": float(finite.max()),
                "criterion": "absolute normalized inner product >= 0.95 at separation >=15 degrees",
            }
        )
    unit = np.arange(4)
    half = np.arange(4) * 0.5
    exact = peak_candidates(steering(unit, [-30])[:, 0], unit, angles)
    disambiguated = peak_candidates(steering(half, [-30])[:, 0], half, angles)
    assert -30 in exact and 30 in exact
    assert len(disambiguated) == 1
    rng = np.random.default_rng(20261022)
    trials = []
    for label, p in [
        ("half_lambda_ULA", half),
        ("one_lambda_ULA", unit),
        ("published_Pf0_azimuth", np.array(source_rows[0]["positions_lambda"])),
    ]:
        dictionary = steering(p, angles)
        wrong = ambiguous = 0
        for trial in range(400):
            truth = -30 if trial % 2 else 30
            x = steering(p, [truth])[:, 0] + 0.01 * (
                rng.normal(size=4) + 1j * rng.normal(size=4)
            ) / np.sqrt(2)
            estimate = angles[np.argmax(abs(dictionary.conj().T @ x))]
            wrong += abs(estimate - truth) > 15
            candidates = peak_candidates(x, p, angles)
            ambiguous += bool(np.any(np.abs(candidates - truth) > 15))
        trials.append(
            {
                "array": label,
                "runs": 400,
                "balanced_truth_angles_deg": [-30, 30],
                "errors_over_15deg": int(wrong),
                "secondary_near_peak_cases": int(ambiguous),
            }
        )
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    report = {
        "profiles": reports,
        "sources": source_rows,
        "analytic_control": {
            "source_angle_deg": -30,
            "one_lambda_candidates_deg": exact.tolist(),
            "half_lambda_candidates_deg": disambiguated.tolist(),
        },
        "synthetic_noise_trials": trials,
        "seed": 20261022,
        "caveat": "Azimuth-only model using published x/lambda coordinates, not measured phase calibration or capture synchronization. Spatial ambiguity is different from a flat no-information spectrum; 0.95 coherence and 15-degree separation are diagnostic choices, not hardware resolution specifications.",
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (out / "pf0-angle-diagnostics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["angle_deg", "competitor_deg", "coherence", "ambiguous"])
        r = diagnose(source_rows[0]["positions_lambda"], angles)
        writer.writerows(
            zip(angles, r["competitor_deg"], r["max_coherence"], r["ambiguous"])
        )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, p, title in [
        (axes[0], unit, "Analytic one-wavelength ULA"),
        (axes[1], source_rows[0]["positions_lambda"], "Published Pf0 azimuth geometry"),
    ]:
        values = diagnose(p, angles)["coherence"]
        im = ax.imshow(
            values,
            origin="lower",
            extent=[-75, 75, -75, 75],
            vmin=0,
            vmax=1,
            cmap="magma",
        )
        ax.set(xlabel="Candidate angle (deg)", ylabel="Source angle (deg)", title=title)
    fig.colorbar(im, ax=axes, label="Normalized manifold coherence")
    fig.savefig(out / "ambiguity.png", dpi=140)
    plt.close(fig)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
