"""The five report figures (matplotlib, Agg backend, English labels).

Fixed categorical palette (Okabe-Ito, colourblind-safe); colour always
follows the entity: round = blue, trunc = vermillion, synthetic = green,
real data = gray family, thresholds = dashed gray.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

C_ROUND = "#0072B2"      # blue
C_TRUNC = "#D55E00"      # vermillion
C_SYNTH = "#009E73"      # green
C_REAL = "#666666"       # gray family (real files)
C_REF = "#000000"        # reference / threshold lines

plt.rcParams.update({
    "figure.dpi": 130,
    "savefig.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def fig1_time_rmse(ks, curves: dict, out: str) -> None:
    """curves: label -> (k array, rmse array, color, linestyle).

    Exact-identity points (RMSE == 0, e.g. k = 16 or the measured data's
    k = 14 grid identity) are masked to NaN so the line breaks instead of
    plunging off a log axis.
    """
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for label, (k, v, color, ls) in curves.items():
        ax.plot(k, _mask_exact(v), ls, color=color, marker="o", markersize=3.5,
                linewidth=1.6, label=label)
    ax.set_yscale("log")
    ax.set_xlabel("requantisation word length k (bits)")
    ax.set_ylabel("time-domain RMSE (LSB)")
    ax.set_xticks(list(ks))
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def _mask_exact(v) -> np.ndarray:
    """RMSE == 0 (exact identity) -> NaN so log axes break, not plunge."""
    v = np.array(v, dtype=float).copy()
    v[v == 0.0] = np.nan
    return v


def fig2_spectrum_rmse(ks, rng_curves: dict, rd_curves: dict, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharex=True)
    for ax, curves, ylab in ((axes[0], rng_curves, "range-spectrum RMSE / peak"),
                             (axes[1], rd_curves, "RD-map RMSE / peak")):
        for label, (k, v, color, ls) in curves.items():
            ax.plot(k, _mask_exact(v), ls, color=color, marker="o",
                    markersize=3.5, linewidth=1.6, label=label)
        ax.set_yscale("log")
        ax.set_xlabel("requantisation word length k (bits)")
        ax.set_ylabel(ylab)
        ax.set_xticks(list(ks))
        ax.set_title(ylab.split(" RMSE")[0], fontsize=9)
        ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig3_band_retention(ks, curves: dict, out: str) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    for label, (k, v, color, ls) in curves.items():
        ax.plot(k, v, ls, color=color, marker="o", markersize=3, linewidth=1.4,
                label=label)
    ax.axhline(0.99, color=C_REF, linestyle="--", linewidth=1.0)
    ax.text(list(ks)[0], 0.995, "99 % criterion", fontsize=7.0,
            color=C_REF, va="bottom")
    ax.set_xlabel("requantisation word length k (bits)")
    ax.set_ylabel("band energy retention (top-1 % bins, ratio)")
    ax.set_xticks(list(ks))
    ax.set_ylim(0.0, 1.12)
    ax.legend(fontsize=7.0, loc="lower right", ncol=2)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)


def fig4_two_tone(maps: dict, ks_shown, rng_bins: int,
                   out: str, weak_bin=None, zoom: int = 30) -> None:
    """Weak-tone visibility across k: full RD map (top) + zoom (bottom).

    The zoomed row centres on the weak tone (dotted crosshair) and uses its
    own colour scale so the weak spot stays discernible while it sinks into
    the rising requantisation floor.
    """
    n = len(ks_shown)
    fig, axes = plt.subplots(2, n, figsize=(2.5 * n, 6.0),
                             sharex="col")
    wr, wv = (weak_bin if weak_bin is not None else (0, 0))
    r0, r1 = max(0, wr - zoom), min(rng_bins, wr + zoom + 1)
    im_top = im_bot = None
    for col, k in enumerate(ks_shown):
        m = maps[k]
        db = 20 * np.log10(np.maximum(m, 1e-12) / np.max(m))
        ax = axes[0, col]
        im_top = ax.imshow(db, aspect="auto", origin="lower",
                           cmap="viridis", vmin=-80, vmax=0)
        ax.axvline(wv, color="white", linewidth=0.6, linestyle=":")
        ax.axhline(wr, color="white", linewidth=0.6, linestyle=":")
        if col == 0:
            ax.set_ylabel(f"range bin (of {rng_bins})\nfull map")
        ax.set_title(f"k = {k} bits", fontsize=9)
        ax = axes[1, col]
        im_bot = ax.imshow(db[r0:r1, :], aspect="auto", origin="lower",
                           cmap="viridis", vmin=-95, vmax=-25)
        ax.axvline(wv, color="white", linewidth=0.6, linestyle=":")
        ax.axhline(wr - r0, color="white", linewidth=0.6, linestyle=":")
        ax.set_xlabel("Doppler bin index")
        if col == 0:
            ax.set_ylabel(f"range bin\nzoom +/-{zoom} around weak tone")
    c1 = fig.colorbar(im_top, ax=axes[0, :], fraction=0.025, pad=0.02)
    c1.set_label("RD magnitude (dB rel. map peak)")
    c2 = fig.colorbar(im_bot, ax=axes[1, :], fraction=0.025, pad=0.02)
    c2.set_label("zoom RD magnitude (dB rel. map peak)")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)


def fig5_peak_drift(ks, amp_curves: dict, shift_curves: dict, out: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.5), sharex=True)
    ax = axes[0]
    for label, (k, v, color, ls) in amp_curves.items():
        ax.plot(k, v, ls, color=color, marker="o", markersize=3.5,
                linewidth=1.6, label=label)
    ax.axhline(0.0, color=C_REF, linestyle="--", linewidth=1.0)
    ax.set_xlabel("requantisation word length k (bits)")
    ax.set_ylabel("peak amplitude change (dB rel. k=16)")
    ax.set_xticks(list(ks))
    ax.legend(fontsize=7.5, loc="best")
    ax = axes[1]
    for label, (k, v, color, ls) in shift_curves.items():
        ax.plot(k, v, ls, color=color, marker="o", markersize=3.5,
                linewidth=1.6, label=label)
    ax.set_xlabel("requantisation word length k (bits)")
    ax.set_ylabel("peak range-bin shift (bins)")
    ax.set_xticks(list(ks))
    ax.legend(fontsize=7.5, loc="best")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
