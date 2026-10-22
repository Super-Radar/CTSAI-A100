"""Fixed perturbations across cutoffs: metric preference is part of the protocol."""

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from logio import default_source, load, xy
from study import gospa

ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0)


def analytic_case():
    # Points are separated enough that each shifted point has one possible
    # below-cutoff partner for every cutoff here. Formulas need no assignment.
    reference = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    missing = reference[:2]
    shifted = reference + [0.3, 0]
    reports = []
    for cutoff in CUTOFFS:
        a = gospa(reference, missing, cutoff)
        b = gospa(reference, shifted, cutoff)
        expected_a = cutoff / np.sqrt(2)
        expected_b = np.sqrt(3) * min(cutoff, 0.3)
        if not np.allclose(
            [a["distance"], b["distance"]], [expected_a, expected_b], rtol=1e-12
        ):
            raise RuntimeError("Closed-form cutoff oracle failed")
        reports.append(
            {
                "cutoff": cutoff,
                "one_missing": a,
                "all_shifted": b,
                "expected_missing": expected_a,
                "expected_shifted": expected_b,
                "preferred": "one_missing"
                if a["distance"] < b["distance"]
                else "all_shifted",
            }
        )
    return reports


def main():
    path = default_source()
    source_root = ROOT.parent / "a100-indoor-cart-mti"
    with (source_root / "data/manifest.csv").open(encoding="utf-8") as stream:
        manifest = list(csv.DictReader(stream))
    entry = next(
        r
        for r in manifest
        if (source_root / r["relative_path"]).resolve() == path.resolve()
    )
    if hashlib.sha256(path.read_bytes()).hexdigest() != entry["sha256"]:
        raise ValueError("Source hash mismatch")
    frames, meta = load(path)
    if meta["rejected_rows"] or any(not f["complete"] for f in frames):
        raise ValueError("Complete valid source required")
    rows, duplicates, empty_frames = [], 0, 0
    for order, frame in enumerate(frames):
        original = xy(frame["points"])
        reference = np.unique(original, axis=0)
        duplicates += len(original) - len(reference)
        if not len(reference):
            empty_frames += 1
            continue
        # Both alternatives are fixed before sweeping c, with no fitting.
        estimates = {
            "one_third_missing": reference[np.arange(len(reference)) % 3 != 0],
            "all_shifted_0.3m": reference + [0.3, 0],
        }
        for cutoff in CUTOFFS:
            for method, estimate in estimates.items():
                score = gospa(reference, estimate, cutoff)
                rows.append(
                    dict(
                        order=order,
                        cutoff=cutoff,
                        method=method,
                        reference_points=len(reference),
                        estimate_points=len(estimate),
                        **{k: v for k, v in score.items() if k != "pairs"},
                    )
                )
    aggregate = []
    fields = ("distance", "localization_squared", "missed_squared", "false_squared")
    for cutoff in CUTOFFS:
        methods = {}
        for method in ("one_third_missing", "all_shifted_0.3m"):
            selected = [
                r for r in rows if r["cutoff"] == cutoff and r["method"] == method
            ]
            methods[method] = {
                "frames": len(selected),
                **{
                    f"mean_{k}": float(np.mean([r[k] for r in selected]))
                    for k in fields
                },
            }
        aggregate.append({"cutoff": cutoff, "methods": methods})
    report = {
        "input": meta,
        "duplicate_locations_removed": duplicates,
        "skipped_empty_frames": empty_frames,
        "evaluated_frames": len(frames) - empty_frames,
        "analytic": analytic_case(),
        "measured_reference": aggregate,
        "limits": "Cutoffs are declared evaluation preferences, not fitted detector parameters. Point snapshots are numeric references; perturbations and three-point oracle are synthetic. Same snapshots/estimates at every cutoff; no object truth, temporal identity metric, or claim that a specific cutoff is correct. Mean distances and mean squared components are reported separately.",
    }
    out = ROOT / "results/cutoff-stress"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    with (out / "frames.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for method in ("one_missing", "all_shifted"):
        axes[0].plot(
            CUTOFFS,
            [r[method]["distance"] for r in report["analytic"]],
            "o-",
            label=method,
        )
    for method in ("one_third_missing", "all_shifted_0.3m"):
        axes[1].plot(
            CUTOFFS,
            [r["methods"][method]["mean_distance"] for r in aggregate],
            "o-",
            label=method,
        )
    for ax, title in zip(
        axes,
        (
            "Three-point closed-form oracle",
            "207 recorded snapshots, synthetic perturbations",
        ),
    ):
        ax.set(
            xlabel="Cutoff c (m)",
            ylabel="GOSPA distance (m)",
            xscale="log",
            title=title,
        )
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.savefig(out / "cutoff.png", dpi=140)
    plt.close(fig)
    print(
        json.dumps(
            {"analytic": report["analytic"], "measured_reference": aggregate}, indent=2
        )
    )


if __name__ == "__main__":
    main()
