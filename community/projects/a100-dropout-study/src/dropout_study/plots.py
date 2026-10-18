"""Matplotlib figures for the dropout study (headless Agg, PNG output).

Styling follows a validated categorical palette (blue / orange / aqua / yellow
in fixed slot order; strategy and model identities keep their colour across
every figure).  Line charts draw the median with a translucent IQR band;
every quantity shown here also exists in the CSV tables under ``results/``
(the table view is the accessibility fallback for the sub-3:1 contrast
slots).  Chinese labels use the WenQuanYi Micro Hei font when present.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .corruption import MODEL_AXIS, MODEL_LABEL  # noqa: E402
from .metrics import to_db  # noqa: E402

# --- reference palette (light mode) -----------------------------------------
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]  # blue, orange, aqua, yellow
STRAT_COLOR = {"zerofill": SLOT[0], "reject": SLOT[1]}
STRAT_LS = {"zerofill": "-", "reject": "--"}
STRAT_LABEL = {"zerofill": "zero-fill in place (official style)", "reject": "reject incomplete chirps"}
MODEL_COLOR = {
    "random_drop": SLOT[0],
    "zero_segment": SLOT[1],
    "chirp_zero": SLOT[2],
    "tail_trunc": SLOT[3],
}
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
CMAP_DB = LinearSegmentedColormap.from_list("a100_blue", BLUE_RAMP)

LOGX_MODELS = {"random_drop", "tail_trunc"}


def _setup():
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "font.family": ["WenQuanYi Micro Hei", "Microsoft YaHei", "DejaVu Sans"],
            "axes.unicode_minus": False,
            "text.color": INK2,
            "axes.labelcolor": INK2,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.edgecolor": AXIS,
            "axes.linewidth": 0.9,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 150,
        }
    )


_setup()


def _style(ax, xlabel=None, ylabel=None):
    ax.grid(True, axis="y")
    ax.grid(False, axis="x")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)
    ax.tick_params(labelsize=8)


def _series(agg, dataset, model, strategy, metric):
    """Sorted (severity, median, q25, q75) for one curve; None medians -> NaN."""
    rows = [
        r for r in agg
        if r["dataset"] == dataset and r["model"] == model and r["strategy"] == strategy
    ]
    rows.sort(key=lambda r: r["severity"])
    sev = np.array([r["severity"] for r in rows], dtype=float)
    med = np.array([np.nan if r[f"{metric}_med"] is None else r[f"{metric}_med"] for r in rows])
    q25 = np.array([np.nan if r[f"{metric}_q25"] is None else r[f"{metric}_q25"] for r in rows])
    q75 = np.array([np.nan if r[f"{metric}_q75"] is None else r[f"{metric}_q75"] for r in rows])
    return sev, med, q25, q75


def _plot_median_band(ax, x, med, q25, q75, color, ls, label=None, marker="o"):
    ax.plot(x, med, ls, color=color, lw=1.8, marker=marker, ms=4.0,
            mfc=color, mec=SURFACE, mew=1.0, label=label, zorder=3)
    ax.fill_between(x, q25, q75, color=color, alpha=0.16, lw=0, zorder=2)


STRAT_MARKER = {"zerofill": "o", "reject": "s"}


def _clip_note(ax, y_top, raw_max, unit):
    """Annotate when curves / IQR bands are clipped by the chosen y-range."""
    if np.isfinite(raw_max) and raw_max > y_top:
        ax.text(
            0.03, 0.97,
            f"curves clipped at {y_top:.0f} {unit} (max {raw_max:.0f})",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=7.5, color=INK2,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.5},
        )


def _label_ref_line(ax, text="1 dB"):
    """Small muted label at the left end of the 1-unit reference line."""
    import matplotlib.transforms as mtransforms

    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    ax.text(0.01, 1.0, f"{text} ", transform=tr, ha="left", va="bottom",
            fontsize=7.5, color=MUTED,
            bbox={"facecolor": SURFACE, "edgecolor": "none", "alpha": 0.85, "pad": 1.0})


def fig_power_curves(agg, dataset, models, out_path, title):
    """Median |peak power error| vs severity, one panel per corruption model.

    The y-range follows the *medians*; IQR bands of catastrophic runs (maps
    with almost no surviving chirps) are clipped and the clipping is
    annotated - the full numbers stay in the CSV tables.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.6), constrained_layout=True)
    first_ax = None
    for ax, model in zip(axes.ravel(), models):
        med_max = 0.0
        band_max = 0.0
        for strat in ("zerofill", "reject"):
            sev, med, q25, q75 = _series(agg, dataset, model, strat, "peak_power_error_db")
            if sev.size == 0:
                continue
            med, q25, q75 = np.abs(med), np.abs(q25), np.abs(q75)
            med_max = max(med_max, float(np.nanmax(med)))
            band_max = max(band_max, float(np.nanmax(q75)))
            _plot_median_band(ax, sev, med, q25, q75,
                              STRAT_COLOR[strat], STRAT_LS[strat], label=STRAT_LABEL[strat],
                              marker=STRAT_MARKER[strat])
            first_ax = first_ax or ax
        if model in LOGX_MODELS:
            ax.set_xscale("log")
        ax.margins(x=0.06)
        ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
        _label_ref_line(ax)
        # Cap the y-range for readability: catastrophic reject-strategy runs
        # (a handful of surviving chirps) reach hundreds of dB and would crush
        # the interesting sub-10 dB region.  The full values stay in the CSVs
        # and the clipping is annotated on the panel.
        y_top = max(2.0, min(1.25 * med_max, 10.0))
        ax.set_ylim(0.0, y_top)
        _clip_note(ax, y_top, max(band_max, med_max), "dB")
        _style(ax, xlabel=MODEL_AXIS[model], ylabel="median |peak power error| (dB)")
        ax.set_title(MODEL_LABEL[model], fontsize=10)
    if first_ax is not None:
        handles, labels = first_ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
                   bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(title, fontsize=12, y=1.10)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_drift_spurious(agg_a, dataset_a, agg_b, dataset_b, models, out_path, title):
    """Peak drift and Doppler spurious delta vs the *affected word fraction*.

    The four corruption models have different severity axes (fraction, word
    count, chirp count); the common x-axis is the fraction of the whole word
    stream that the corruption touches, so curves compare damage at equal
    amounts of affected data.
    """
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0), constrained_layout=True)
    panels = (
        (axes[0, 0], agg_a, dataset_a, "drift_max_bins", "median peak drift (bins)", 4.0, "bins"),
        (axes[0, 1], agg_a, dataset_a, "doppler_spurious_delta", "median Doppler spurious delta (count)", 12.0, "counts"),
        (axes[1, 0], agg_b, dataset_b, "drift_max_bins", "median peak drift (bins)", 4.0, "bins"),
        (axes[1, 1], agg_b, dataset_b, "doppler_spurious_delta", "median Doppler spurious delta (count)", 12.0, "counts"),
    )
    for ax, agg, dataset, metric, ylabel, cap, unit in panels:
        med_max = 0.0
        for model in models:
            for strat in ("zerofill", "reject"):
                sev, med, _, _ = _series(agg, dataset, model, strat, metric)
                if sev.size == 0:
                    continue
                med_max = max(med_max, float(np.nanmax(med)))
                x = _affected_fraction(sev, model, dataset)
                ax.plot(x, med, STRAT_LS[strat], color=MODEL_COLOR[model],
                        lw=1.8, marker=STRAT_MARKER[strat], ms=3.5,
                        mec=SURFACE, mew=0.7)
        ax.set_xscale("log")
        ax.margins(x=0.06)
        y_top = min(cap, med_max * 1.15) if np.isfinite(med_max) and med_max > 0 else cap
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
        if med_max > y_top:
            ax.set_ylim(top=y_top)
            _clip_note(ax, y_top, med_max, unit)
        _style(ax, xlabel="affected word fraction of the stream", ylabel=ylabel)
        ax.set_title(dataset, fontsize=10)
    legend_handles = [plt.Line2D([], [], color=MODEL_COLOR[m], lw=2.0,
                                 marker="o", ms=5, label=MODEL_LABEL[m])
                      for m in models]
    legend_handles += [
        plt.Line2D([], [], color=INK2, lw=1.8, ls="-", marker="o", ms=5,
                   label="zero-fill"),
        plt.Line2D([], [], color=INK2, lw=1.8, ls="--", marker="s", ms=5,
                   label="reject incomplete chirps"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.06))
    fig.suptitle(title, fontsize=12, y=1.11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def _affected_fraction(sev: np.ndarray, model: str, dataset: str) -> np.ndarray:
    """Map a model's severity values onto the fraction of the word stream affected."""
    s = np.asarray(sev, dtype=float)
    if model in ("random_drop", "tail_trunc"):
        return s
    if model == "zero_segment":
        # both init0 and init1 payloads are 256*512 = 128*1024 = 131072 words
        return s / 131072.0
    if model == "chirp_zero":
        # chirp count / nchirp: init0 -> /256, init1 -> /128
        nchirp = 128 if "init1" in dataset else 256
        return s / nchirp
    raise ValueError(model)


def fig_example_maps(examples, out_path, title):
    """Side-by-side dB RV maps (shared colour scale) as qualitative evidence.

    ``examples``: list of ``(heading, note, power_map)``.
    """
    n = len(examples)
    fig, axes = plt.subplots(1, n, figsize=(3.15 * n, 3.7), constrained_layout=True)
    axes = np.atleast_1d(axes)
    clean_db = to_db(examples[0][2])
    vmax = float(np.percentile(clean_db, 99.5))
    vmin = vmax - 45.0
    im = None
    for ax, (heading, note, mp) in zip(axes, examples):
        db = to_db(mp)
        im = ax.imshow(db, origin="lower", aspect="auto", cmap=CMAP_DB,
                       vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(heading, fontsize=10)
        ax.set_xlabel("Doppler bin", fontsize=8)
        ax.set_ylabel("Range bin", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.text(0.5, -0.16, note, transform=ax.transAxes, ha="center", va="top",
                fontsize=8, color=INK2)
    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.015)
    cbar.set_label("power (dB)", fontsize=8)
    cbar.ax.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def fig_position_study(agg_pos, out_path, title):
    """Zero-segment position comparison: median |peak power error| vs segment length."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4), constrained_layout=True)
    med_max, band_max = 0.0, 0.0
    for color, pos in ((SLOT[0], "head"), (SLOT[1], "middle"), (SLOT[2], "tail")):
        rows = [r for r in agg_pos if r["position"] == pos]
        rows.sort(key=lambda r: r["severity"])
        if not rows:
            continue
        sev = np.array([r["severity"] for r in rows], dtype=float)
        med = np.abs(np.array([np.nan if r["peak_power_error_db_med"] is None
                               else r["peak_power_error_db_med"] for r in rows]))
        q25 = np.abs(np.array([np.nan if r["peak_power_error_db_q25"] is None
                               else r["peak_power_error_db_q25"] for r in rows]))
        q75 = np.abs(np.array([np.nan if r["peak_power_error_db_q75"] is None
                               else r["peak_power_error_db_q75"] for r in rows]))
        med_max = max(med_max, float(np.nanmax(med)))
        band_max = max(band_max, float(np.nanmax(q75)))
        _plot_median_band(ax, sev, med, q25, q75, color, "-", label=f"segment at {pos}")
    ticks = [1, 2, 4, 8, 16, 32, 64]
    ax.set_xscale("log", base=2)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(v) for v in ticks])
    ax.margins(x=0.08)
    ax.axhline(1.0, color=MUTED, lw=0.9, ls=(0, (4, 3)))
    _label_ref_line(ax)
    y_top = max(1.5, 1.25 * med_max)
    ax.set_ylim(0.0, y_top)
    _clip_note(ax, y_top, max(band_max, med_max), "dB")
    if med_max < 1e-3:
        # Explicit null result: all three series coincide at ~0 dB.
        ax.text(
            0.5, 0.5,
            f"所有位置/长度下中位数 |误差| ≈ 0 dB\n（最大 {max(med_max, band_max):.2e} dB）",
            transform=ax.transAxes, ha="center", va="center",
            fontsize=9, color=INK2,
        )
    _style(ax, xlabel="zero-segment length (words)", ylabel="median |peak power error| (dB)")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.set_title(title, fontsize=11)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


__all__ = [
    "fig_power_curves", "fig_drift_spurious", "fig_example_maps",
    "fig_position_study",
]
