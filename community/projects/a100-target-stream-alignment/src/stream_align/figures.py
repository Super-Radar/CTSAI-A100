"""Figures for the stream-alignment study (Agg backend, English labels).

All randomness (subsampling of match connectors in fig1) uses the fixed seed
``20261003`` so every figure is byte-reproducible from the same data.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from stream_align.align import AlignResult
from stream_align.parser import StreamLog

SEED = 20261003

C_RAW = "#1f77b4"   # blue
C_TRA = "#d62728"   # red
C_UNMATCHED = "#ff9900"
C_BATCH_A = "#1f77b4"  # 2025-11-20 batch
C_BATCH_B = "#d62728"  # 2026-05 batch
C_METHODS = {"A": "#7f7f7f", "B": "#1f77b4", "C": "#2ca02c"}


def _epochs(log: StreamLog) -> np.ndarray:
    return np.array([f.epoch_ms for f in log.frames], dtype=np.float64)


def fig1_timeband(
    raw: StreamLog,
    tra: StreamLog,
    result: AlignResult,
    session: str,
    out_path: Path,
    max_connectors: int = 400,
) -> None:
    """Dual-stream time band: Raw/Tra frame instants, sampled match connectors."""
    rng = np.random.default_rng(SEED)
    e_raw, e_tra = _epochs(raw), _epochs(tra)
    t0 = min(e_raw.min(), e_tra.min())
    x_raw = (e_raw - t0) / 1000.0
    x_tra = (e_tra - t0) / 1000.0

    matched_tra = {p[1] for p in result.pairs}
    unmatched_tra = np.array(
        [i for i in range(tra.n_frames) if i not in matched_tra], dtype=int
    )

    fig, ax = plt.subplots(figsize=(12, 3.8))
    ax.scatter(x_raw, np.ones_like(x_raw), s=2.5, color=C_RAW, alpha=0.7,
               label=f"Raw frames (n={raw.n_frames})")
    matched_mask = np.ones(tra.n_frames, dtype=bool)
    matched_mask[unmatched_tra] = False
    ax.scatter(x_tra[matched_mask], np.zeros(int(matched_mask.sum())),
               s=2.5, color=C_TRA, alpha=0.7,
               label=f"Tra frames matched at |dt|<=tol (n={int(matched_mask.sum())})")
    if unmatched_tra.size:
        ax.scatter(x_tra[unmatched_tra], np.zeros_like(x_tra[unmatched_tra]),
                   s=14, marker="x", color=C_UNMATCHED, alpha=0.9, zorder=3,
                   linewidths=0.8,
                   label=f"Tra frames without Raw pair (n={unmatched_tra.size})")

    if result.pairs:
        idx = np.arange(len(result.pairs))
        if len(idx) > max_connectors:
            idx = np.sort(rng.choice(idx, size=max_connectors, replace=False))
        for k in idx:
            r_i, t_i = result.pairs[k]
            ax.plot([x_raw[r_i], x_tra[t_i]], [1, 0], color="#888888",
                    lw=0.25, alpha=0.4, zorder=1)

    ax.set_ylim(-0.4, 1.85)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["TraTarget", "RawTarget"])
    ax.set_xlabel("Time since session start (s)")
    ax.set_ylabel("Stream")
    tol_txt = result.method.split("tol")[-1]
    ax.set_title(
        f"Fig 1. Dual-stream time band - {session.split('_Tatget')[0]} "
        f"(method C, tol = {tol_txt})", pad=26
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=8,
              ncol=3, framealpha=0.9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig2_tol_coverage(
    curves: dict[str, list[dict]], batches: dict[str, str], out_path: Path
) -> None:
    """Coverage vs tolerance for every session; solid = Raw side, dashed = Tra side."""
    fig, ax = plt.subplots(figsize=(9, 5))
    era_label = {"2025": "2025-11-20 recordings", "2026": "2026-05 recordings"}
    seen_batch_line: dict[str, bool] = {}
    for session, rows in sorted(curves.items()):
        batch = batches.get(session, "?")
        color = C_BATCH_A if batch.startswith("2025") else C_BATCH_B
        lab = era_label.get(batch, batch)
        tols = [r["tol_ms"] for r in rows]
        ax.plot(tols, [r["cov_raw"] for r in rows], "-", color=color, alpha=0.75,
                lw=1.4,
                label=f"{lab} (Raw side)" if not seen_batch_line.get(batch + "raw") else None)
        seen_batch_line[batch + "raw"] = True
        ax.plot(tols, [r["cov_tra"] for r in rows], "--", color=color, alpha=0.75,
                lw=1.4,
                label=f"{lab} (Tra side)" if not seen_batch_line.get(batch + "tra") else None)
        seen_batch_line[batch + "tra"] = True
    ax.set_xlabel("Tolerance tol (ms)")
    ax.set_ylabel("Paired fraction of stream frames")
    ax.set_ylim(0.78, 1.04)
    ax.set_title("Fig 2. Method C coverage vs matching tolerance (20 sessions)")
    ax.annotate(
        "All Raw-side curves and all 2025-11-20 sessions lie at 1.0:\n"
        "fully paired already at tol = 0 ms",
        xy=(25, 1.0), xytext=(6, 0.885), fontsize=8,
        arrowprops=dict(arrowstyle="->", lw=0.8),
        bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
    )
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.14),
              ncol=2, framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig3_residual_hist(
    resid_a: list[int], resid_c: list[int], tol_c: int, out_path: Path
) -> None:
    """|dt| histograms: FrameNb pairing (A) vs tolerance pairing (C)."""
    a = np.abs(np.asarray(resid_a, dtype=np.float64))
    c = np.abs(np.asarray(resid_c, dtype=np.float64))
    hi = max(a.max() if a.size else 1.0, 1.0)
    bins = np.linspace(0, hi, 60)
    top = 1.8 * max(
        np.histogram(a, bins=bins)[0].max() if a.size else 1.0,
        np.histogram(c, bins=bins)[0].max() if c.size else 1.0,
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, arr, color, name in (
        (axes[0], a, C_METHODS["A"], "Method A: FrameNb equality"),
        (axes[1], c, C_METHODS["C"], f"Method C: tolerance {tol_c} ms"),
    ):
        n = int(arr.size)
        med = float(np.median(arr)) if arr.size else float("nan")
        p95 = float(np.percentile(arr, 95)) if arr.size else float("nan")
        ax.hist(arr, bins=bins, color=color, edgecolor="white", lw=0.2)
        ax.set_yscale("log")
        ax.set_ylim(0.5, top)
        ax.set_title(f"{name} (n={n})", fontsize=10)
        ax.text(
            0.97, 0.95,
            f"median |dt| = {med:.0f} ms\np95 |dt| = {p95:.0f} ms",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            bbox=dict(boxstyle="round", fc="white", ec="0.7", alpha=0.9),
        )
        ax.set_xlabel("Pair residual |dt| (ms)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Number of matched pairs")
    fig.suptitle("Fig 3. Matched-pair residual distribution, all sessions pooled")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def fig4_coverage_by_method(
    sessions: list[str],
    covs: dict[str, dict[str, tuple[float, float]]],
    batches: dict[str, str],
    out_path: Path,
) -> None:
    """Per-session paired fraction for methods A/B/C, Raw and Tra sides."""
    order = sorted(sessions)
    x = np.arange(len(order))
    width = 0.26
    fig, axes = plt.subplots(2, 1, figsize=(12, 7.2), sharex=True)
    for row, (ax, side, which) in enumerate(
        [(axes[0], 0, "Raw side"), (axes[1], 1, "Tra side")]
    ):
        for k, method in enumerate(("A", "B", "C")):
            vals = [covs[s][method][side] for s in order]
            ax.bar(x + (k - 1) * width, vals, width,
                   color=C_METHODS[method],
                   label={"A": "A: FrameNb", "B": "B: exact time",
                          "C": "C: tolerance (best)"}[method])
        ax.set_ylim(0, 1.12)
        ax.set_ylabel(f"Paired fraction ({which})")
        ax.grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8, ncol=3, loc="lower center",
                   bbox_to_anchor=(0.5, 1.001), framealpha=0.9)
    # batch separators
    for ax in axes:
        prev = batches[order[0]]
        for i, s in enumerate(order):
            b = batches[s]
            if b != prev:
                ax.axvline(i - 0.5, color="black", lw=0.8, ls=":")
                prev = b
    labels = [s.replace("_Tatget_CTLRR-300-V2_1", "")[:19] for s in order]
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
    axes[1].set_xlabel("Session")
    fig.suptitle("Fig 4. Session-level pairing coverage by method (A/B/C)",
                 y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
