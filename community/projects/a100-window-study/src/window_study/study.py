"""The three experiments of the A100 window study.

1. :func:`exp1_synthetic` -- synthetic single-/dual-tone benchmark: amplitude
   recovery after coherent-gain correction, sidelobe level vs theory, -3 dB
   peak width vs the window-kernel prediction, Parseval checks.
2. :func:`exp2_real` -- real far-wave (f0) range spectra under
   rect/hann/hamming on one chirp: uncompensated coherent-gain offset, leakage
   skirts, and weak-target visibility (natural-candidate scan + a clearly
   declared injected-target demonstration).
3. :func:`exp3_normalization` -- dB presentation conventions for the *same*
   spectrum (|X|, |X|/N, |X|/sum(w), 20log10|X|, 10log10|X|^2, official
   MATLAB |X|/255) and the 1021-valid-samples vs forced-1024 window comparison.

Every figure is a PNG (matplotlib Agg backend, < 300 KB each), every table a
CSV under ``results/``.  All runs are deterministic (fixed seed, no randomness
is actually needed -- the benchmark tones are analytic).
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from . import pipeline, windows

logger = logging.getLogger("window_study")

# fixed presentation palette (colorblind-validated categorical order, light surface);
# each window keeps its color in every figure of this project.
COLORS = {
    "rect": "#2a78d6",
    "hann": "#eb6834",
    "hamming": "#1baf7a",
    "blackman": "#eda100",
}
INK = "#1a1a19"
INK_MUTED = "#6f6e66"
GRID = "#d9d8cf"
ACCENT = "#4a3aa7"

MAX_PNG_BYTES = 300_000
WINDOWS_REAL = ("rect", "hann", "hamming")


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "text.color": INK,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.6,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.frameon": False,
            "legend.fontsize": 8,
            "lines.linewidth": 1.4,
        }
    )


def db20(x) -> np.ndarray:
    return 20.0 * np.log10(np.abs(x) + 1e-300)


def _save(fig, path: Path) -> int:
    """Save a figure as PNG, stepping the dpi down until it is < 300 KB."""
    for dpi in (130, 110, 95, 80):
        fig.savefig(path, dpi=dpi)
        if path.stat().st_size <= MAX_PNG_BYTES:
            break
    plt.close(fig)
    size = path.stat().st_size
    if size > MAX_PNG_BYTES:
        logger.warning("figure %s exceeds 300 KB (%.0f KB)", path.name, size / 1024)
    return size


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers: list[str] = list(rows[0].keys())
    for r in rows[1:]:
        for kx in r:
            if kx not in headers:
                headers.append(kx)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=headers, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _dirs(project_dir) -> tuple[Path, Path]:
    fig_dir = Path(project_dir) / "figures"
    res_dir = Path(project_dir) / "results"
    fig_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir, res_dir


def _local_maxima(db: np.ndarray) -> np.ndarray:
    idx = np.where((db[1:-1] > db[:-2]) & (db[1:-1] >= db[2:]))[0] + 1
    return idx


# ---------------------------------------------------------------------------
# Experiment 1: synthetic benchmark
# ---------------------------------------------------------------------------
def exp1_synthetic(project_dir, nfft: int = 1024, seed: int = 42) -> dict:
    fig_dir, res_dir = _dirs(project_dir)
    np.random.default_rng(seed)  # reserved for optional noise; tones are analytic

    npad = 8 * nfft
    metric_rows: list[dict] = []
    kernels: dict[str, np.ndarray] = {}
    for sym, variant in ((False, "periodic"), (True, "symmetric")):
        for name in windows.WINDOW_NAMES:
            w = windows.make_window(name, nfft, sym=sym)
            W = windows.window_kernel(w, npad)
            cg = windows.coherent_gain(w)
            row = dict(
                window=name,
                variant=variant,
                n=nfft,
                cg=cg,
                cg_db=20.0 * np.log10(cg),
                enbw_bins=windows.enbw_bins(w),
                sidelobe_db_meas=windows.kernel_sidelobe_db(W),
                width3db_bins_kernel=windows.kernel_3db_width_bins(w, npad),
            )
            if not sym:
                th = windows.analytic_metrics(name)
                kernels[name] = w
                row.update(
                    cg_theory=th["cg"],
                    enbw_bins_theory=th["enbw_bins"],
                    sidelobe_db_theory=th["sidelobe_db"],
                    cg_abs_err=abs(row["cg"] - th["cg"]),
                    enbw_abs_err=abs(row["enbw_bins"] - th["enbw_bins"]),
                    sidelobe_abs_err=abs(row["sidelobe_db_meas"] - th["sidelobe_db"]),
                )
            metric_rows.append(row)
    write_csv(res_dir / "exp1_window_metrics.csv", metric_rows)

    # ---- figure 1: window shapes and kernels -------------------------------
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    for name, w in kernels.items():
        ax.plot(np.arange(nfft), w, color=COLORS[name], label=name, lw=1.2)
    ax.set_xlabel("sample n")
    ax.set_ylabel("w[n]")
    ax.set_title("(a) window shapes (periodic, N = %d)" % nfft)
    ax.legend(ncol=2)
    ax = axes[1]
    for name, w in kernels.items():
        Wfull = np.fft.fft(w, npad)
        db = db20(Wfull)
        db -= db[0]
        per_bin = npad / nfft
        sel = int(8 * per_bin)
        freqs = np.concatenate(
            [-np.arange(1, sel + 1)[::-1] / per_bin, np.arange(sel + 1) / per_bin]
        )
        m = np.concatenate([db[-sel:], db[: sel + 1]])
        ax.plot(freqs, m, color=COLORS[name], lw=1.2,
                label=f"{name}: SLL {row_sll(name, metric_rows):.1f} dB")
    ax.set_ylim(-100, 3)
    ax.set_xlabel("frequency offset (DFT bins)")
    ax.set_ylabel("window kernel (dB rel. peak)")
    ax.set_title("(b) window kernels; SLL = highest sidelobe, measured")
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save(fig, fig_dir / "fig1_windows_and_kernels.png")

    # ---- tone amplitude recovery + peak width ------------------------------
    n = np.arange(nfft)
    k1, k2 = max(4, nfft // 10), 3 * nfft // 10  # 102 / 307 for nfft = 1024
    tone_sets = {"single": ((k1, 1.0),), "dual": ((k1, 1.0), (k2, 0.25))}
    recovery_rows: list[dict] = []
    for name, w in kernels.items():
        sw = float(w.sum())
        pred_width = windows.kernel_3db_width_bins(w, npad)
        for case, tones in tone_sets.items():
            x = np.zeros(nfft)
            for k, a in tones:
                x += a * np.cos(2.0 * np.pi * k * n / nfft)
            xw = x * w
            X = np.fft.fft(xw)
            db = db20(X[: nfft // 2])
            # the -3 dB width is measured on an 8x zero-padded spectrum
            # (frequency interpolation): 1-bin sampling + linear interpolation
            # cannot trace a ~1.4-bin mainlobe, and a bin-aligned rect tone has
            # exactly zero leakage at every unpadded DFT bin.
            X8 = np.fft.fft(xw, 8 * nfft)
            db8 = db20(X8[: 4 * nfft])
            for k, a in tones:
                a_est = 2.0 * abs(X[k]) / sw
                recovery_rows.append(
                    dict(
                        case=case,
                        window=name,
                        bin=k,
                        amp_true=a,
                        amp_est=a_est,
                        rel_err=abs(a_est - a) / a,
                        width3db_meas_bins=windows.measured_3db_width_bins(db8, 8 * k) / 8.0,
                        width3db_pred_bins=pred_width,
                    )
                )
    write_csv(res_dir / "exp1_tone_recovery.csv", recovery_rows)

    # ---- weak-neighbour masking demo (synthetic) ---------------------------
    kw = k1 + 6  # weak tone 6 bins away from the strong tone
    weak_db = -35.0
    x_mask = np.cos(2.0 * np.pi * k1 * n / nfft) + 10 ** (weak_db / 20.0) * np.cos(
        2.0 * np.pi * kw * n / nfft
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    ax = axes[0]
    lo_a, hi_a = max(1, k1 - 40), min(k2 + 40, nfft // 2 - 1)
    for name, w in kernels.items():
        X = np.fft.fft(x_dual_signal(tone_sets["dual"], n, nfft) * w)[: nfft // 2]
        amp_db = db20(2.0 * np.abs(X) / w.sum())
        ax.plot(
            np.arange(lo_a, hi_a),
            amp_db[lo_a:hi_a],
            color=COLORS[name],
            lw=1.2,
            label=name,
        )
    ax.axvline(k1, color=INK_MUTED, lw=0.7, ls=":")
    ax.axvline(k2, color=INK_MUTED, lw=0.7, ls=":")
    ax.text(k1, ax.get_ylim()[0] + 5, "k=%d" % k1, ha="center", fontsize=8, color=INK_MUTED)
    ax.text(k2, ax.get_ylim()[0] + 5, "k=%d" % k2, ha="center", fontsize=8, color=INK_MUTED)
    ax.set_xlabel("DFT bin")
    ax.set_ylabel("amplitude (dB, CG-compensated)")
    ax.set_title("(a) dual tone 1.0 / 0.25, recovered after 2|X|/sum(w)")
    ax.legend()
    ax = axes[1]
    lo_b, hi_b = max(1, k1 - 8), min(kw + 11, nfft // 2 - 1)
    for name, w in kernels.items():
        X = np.fft.fft(x_mask * w)[: nfft // 2]
        amp_db = db20(2.0 * np.abs(X) / w.sum())
        ax.plot(np.arange(lo_b, hi_b), amp_db[lo_b:hi_b], color=COLORS[name], lw=1.2, label=name)
    ax.axhline(weak_db, color=INK_MUTED, lw=0.8, ls="--")
    ax.text(lo_b + 1, weak_db + 1.0, "weak tone true level (-35 dB)", fontsize=8, color=INK_MUTED)
    ax.annotate(
        "rect sidelobes of the strong tone\nsit above the weak tone",
        xy=(kw, -25.0),
        xytext=(kw + 2, -12.0),
        fontsize=8,
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.8),
    )
    ax.set_xlabel("DFT bin")
    ax.set_ylabel("amplitude (dB, CG-compensated)")
    ax.set_title("(b) weak neighbour at -35 dB, 6 bins away")
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, fig_dir / "fig2_synthetic_tones.png")

    # ---- Parseval checks ----------------------------------------------------
    k0 = max(4, nfft // 13)
    z = np.exp(1j * 2.0 * np.pi * k0 * n / nfft)  # constant modulus -> exact ratio
    parseval_rows = []
    for name, w in kernels.items():
        xw = z * w
        e_raw = float(np.sum(np.abs(z) ** 2))
        e_win = float(np.sum(np.abs(xw) ** 2))
        ratio = e_win / e_raw
        expect = float(np.sum(np.square(w)) / nfft)
        X = np.fft.fft(xw)
        fft_ratio = float(np.sum(np.abs(X) ** 2) / (nfft * e_win))
        parseval_rows.append(
            dict(
                window=name,
                energy_ratio_windowed_over_raw=ratio,
                sum_w2_over_n=expect,
                abs_err=abs(ratio - expect),
                fft_parseval_ratio=fft_ratio,
                fft_abs_err=abs(fft_ratio - 1.0),
            )
        )
    write_csv(res_dir / "exp1_parseval.csv", parseval_rows)

    per = {(r["window"], r["variant"]): r for r in metric_rows}
    summary = {
        "worst_cg_abs_err": max(r["cg_abs_err"] for r in metric_rows if "cg_abs_err" in r),
        "worst_enbw_abs_err": max(r["enbw_abs_err"] for r in metric_rows if "enbw_abs_err" in r),
        "worst_sidelobe_abs_err": max(r["sidelobe_abs_err"] for r in metric_rows if "sidelobe_abs_err" in r),
        "worst_amp_rel_err": max(r["rel_err"] for r in recovery_rows),
        "worst_parseval_err": max(r["abs_err"] for r in parseval_rows),
        "worst_fft_parseval_err": max(r["fft_abs_err"] for r in parseval_rows),
        "metrics": {name: per[(name, "periodic")] for name in windows.WINDOW_NAMES},
    }
    return summary


def row_sll(name: str, rows: list[dict]) -> float:
    for r in rows:
        if r["window"] == name and r["variant"] == "periodic":
            return float(r["sidelobe_db_meas"])
    return float("nan")


def x_dual_signal(tones, n, nfft) -> np.ndarray:
    x = np.zeros(nfft)
    for k, a in tones:
        x += a * np.cos(2.0 * np.pi * k * n / nfft)
    return x


# ---------------------------------------------------------------------------
# Experiment 2: real-data range spectra under different windows
# ---------------------------------------------------------------------------
def scan_natural_weak_targets(
    db_h: np.ndarray,
    db_r: np.ndarray,
    range_m: np.ndarray,
    strong_bin: int,
    min_prominence_db: float = 8.0,
    near_bins: int = 45,
    strong_excess_db: float = 20.0,
) -> list[dict]:
    """Local maxima of the hann spectrum that rise ``min_prominence_db`` above
    the median hann floor while sitting within ``near_bins`` of a >=
    ``strong_excess_db`` stronger peak -- i.e. inside a rect leakage skirt.

    Every candidate is flagged: peaks inside the close-target leakage tail
    (``bin < strong_bin + near_bins`` for the dominant near-range target) may be
    part of the target/scenario itself, so they are *ambiguous*; only a clean
    candidate would count as an unambiguous natural example.
    """
    floor = float(np.median(db_h))
    loc = _local_maxima(db_h)
    cands = []
    for b in loc:
        prom = db_h[b] - floor
        if prom < min_prominence_db:
            continue
        near = loc[(np.abs(loc - b) <= near_bins) & (db_h[loc] - db_h[b] >= strong_excess_db)]
        if not near.size:
            continue
        b_strong = int(near[np.argmax(db_h[near])])
        inside_tail = b < strong_bin + near_bins
        cands.append(
            dict(
                bin=int(b),
                range_m=float(range_m[b]),
                hann_db=float(db_h[b]),
                rect_db=float(db_r[b]),
                prominence_db=float(prom),
                strongest_neighbour_bin=b_strong,
                rect_minus_hann_db=float(db_r[b] - db_h[b]),
                verdict="ambiguous: inside close-target leakage tail"
                if inside_tail
                else "clean natural weak peak",
            )
        )
    return cands


def injection_demo(x, cfg: pipeline.RadarConfig, db_h, comp_h, comp_r, strong_bin, floor_h):
    """Declared synthetic two-tone injection on the real chirp background.

    The real frame's near zone is cluttered by the close target's own leakage
    tail, so the weak-target masking mechanism is demonstrated with a *declared*
    synthetic scene built on the real data's noise background:

    * a strong synthetic tone in a spectrally clean zone (17 bins all within
      floor + 2 dB, far from the real strong peak), at the same gain-corrected
      amplitude as the real strongest peak, placed at a *fractional* bin
      (zone + 0.5) so that its leakage skirt is that of a generic
      (non-bin-aligned) target;
    * a weak synthetic tone 27 dB below it, 7 bins away.

    In the gain-corrected domain (2|X|/sum(w), the apparent amplitude is then
    window-independent): hann's sidelobes (-31.5 dB first sidelobe, falling
    ~18 dB/octave) leave the weak tone a clean local peak, while rect's skirt
    (-13.3 dB falling only ~6 dB/octave) buries it.  Resolution = level at the
    weak bin >= 3 dB above the maximum of its +/-3-bin neighbours.  All
    numbers are reported as found.
    """
    zone = None
    for b0 in range(140, 300, 2):
        if np.max(db_h[b0 - 8 : b0 + 9]) <= floor_h + 2.0:
            zone = b0
            break
    if zone is None:
        raise RuntimeError("no spectrally clean zone found for the injection demo")

    weak_bin = zone + 7
    delta_db = 27.0
    strong_bin_f = zone + 0.5
    a_strong = 10.0 ** (float(comp_h[strong_bin]) / 20.0)
    a_weak = a_strong * 10.0 ** (-delta_db / 20.0)
    n = np.arange(cfg.rng_nfft)
    x_inj = (
        x
        + a_strong * np.cos(2.0 * np.pi * strong_bin_f * n / cfg.rng_nfft)
        + a_weak * np.cos(2.0 * np.pi * weak_bin * n / cfg.rng_nfft)
    )

    def _prom(comp, b):
        nb = [i for i in range(b - 3, b + 4) if i != b and 0 <= i < len(comp)]
        return float(comp[b] - max(comp[i] for i in nb))

    Xi, comp_i = {}, {}
    for name in WINDOWS_REAL:
        X, w = pipeline.range_fft(x_inj, cfg, name)
        Xi[name] = dict(X=X, sum_w=float(w.sum()))
        comp_i[name] = db20(2.0 * np.abs(X) / float(w.sum()))
    hann_prom = _prom(comp_i["hann"], weak_bin)
    rect_prom = _prom(comp_i["rect"], weak_bin)
    return dict(
        strong_bin=strong_bin_f,
        weak_bin=weak_bin,
        delta_db=delta_db,
        a_strong=a_strong,
        a_weak=a_weak,
        hann_prom_db=hann_prom,
        rect_prom_db=rect_prom,
        hann_resolves=bool(hann_prom >= 3.0),
        rect_resolves=bool(rect_prom >= 3.0),
        rect_burial_db=float(comp_r[weak_bin] - 20.0 * np.log10(a_weak)),
        x_inj=x_inj,
        Xi=Xi,
        comp_i=comp_i,
    )


def inject_weak_target(x, cfg: pipeline.RadarConfig, bin_target, floor_h_db, margin_db=8.0):
    """Return (modified chirp, cosine amplitude) adding a real tone at DFT bin
    ``bin_target`` whose *raw* hann-spectrum peak lands ``margin_db`` above the
    hann noise floor."""
    amp_target = 10.0 ** ((floor_h_db + margin_db) / 20.0)
    _, wh = pipeline.range_fft(x, cfg, "hann")
    amp_cos = 2.0 * amp_target / float(wh.sum())
    n = np.arange(cfg.rng_nfft)
    return x + amp_cos * np.cos(2.0 * np.pi * bin_target * n / cfg.rng_nfft), amp_cos


def exp2_real(data_dir, project_dir, cfg=pipeline.FAR_F0, channel: int = 0, chirp_index: int = 0) -> dict:
    fig_dir, res_dir = _dirs(project_dir)
    path = pipeline.find_data_file(data_dir, cfg, channel)
    sha = pipeline.sha256_of(path)
    adc, info = pipeline.load_channel(path, cfg)
    x = adc[chirp_index]

    spectra = {}
    for name in WINDOWS_REAL:
        X, w = pipeline.range_fft(x, cfg, name)  # symmetric windows, as MATLAB
        spectra[name] = dict(X=X, w=w, sum_w=float(w.sum()))
    range_m = pipeline.range_axis(cfg)

    rows = []
    for name, d in spectra.items():
        db_raw = db20(d["X"])
        amp = 2.0 * np.abs(d["X"]) / d["sum_w"]
        pk = int(np.argmax(db_raw))
        rows.append(
            dict(
                window=name,
                sum_w=d["sum_w"],
                peak_bin=pk,
                peak_range_m=float(range_m[pk]),
                peak_db_raw=float(db_raw[pk]),
                peak_db_amp_compensated=float(db20(amp)[pk]),
                floor_db_median=float(np.median(db_raw)),
            )
        )
    write_csv(res_dir / "exp2_real_spectra.csv", rows)
    (res_dir / "data_sha256.txt").write_text(
        f"{sha}  {path.name}\n# {info['values_read']} values read, payload {info['payload_len']}, "
        f"dropped tail {info['dropped_tail']}\n",
        encoding="utf-8",
    )

    cg_offset_db = 20.0 * np.log10(spectra["hann"]["sum_w"] / spectra["rect"]["sum_w"])
    db_h = db20(spectra["hann"]["X"])
    db_r = db20(spectra["rect"]["X"])
    strong_bin = int(np.argmax(db_h))
    floor_h = float(np.median(db_h))

    # ---- figure 3: window comparison on the real chirp ---------------------
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    for name in WINDOWS_REAL:
        ax.plot(range_m, db20(spectra[name]["X"]), color=COLORS[name], lw=1.1, label=name)
    pk_rect = float(db20(spectra["rect"]["X"])[strong_bin])
    pk_hann = float(db_h[strong_bin])
    ax.annotate(
        "",
        xy=(range_m[strong_bin], pk_rect),
        xytext=(range_m[strong_bin], pk_hann),
        arrowprops=dict(arrowstyle="<->", color=ACCENT, lw=1.0),
    )
    ax.text(
        range_m[strong_bin] + 3.0,
        0.5 * (pk_rect + pk_hann),
        "uncompensated window gain rect $\\to$ hann:\n"
        "systematic CG part %.2f dB, gap at this bin %.2f dB\n"
        "(residual = target structure / scalloping)"
        % (cg_offset_db, pk_rect - pk_hann),
        fontsize=8,
        color=ACCENT,
    )
    ax.set_xlim(0, 80)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("|X| (dB, uncompensated)")
    ax.set_title(
        "(a) far-wave f0, Rx%d chirp %d: raw spectra" % (channel, chirp_index)
    )
    ax.legend()
    ax = axes[1]
    comp_hann = db20(2.0 * np.abs(spectra["hann"]["X"]) / spectra["hann"]["sum_w"])
    ref_db = float(comp_hann[strong_bin])
    for name in WINDOWS_REAL:
        d = spectra[name]
        amp_db = db20(2.0 * np.abs(d["X"]) / d["sum_w"]) - ref_db
        ax.plot(range_m, amp_db, color=COLORS[name], lw=1.1, label=name)
    ax.axvspan(range_m[strong_bin], range_m[strong_bin + 40], color=GRID, alpha=0.45, lw=0)
    ax.text(
        range_m[strong_bin + 20],
        ax.get_ylim()[0] + 6,
        "rect leakage skirt dominates\nweak-target visibility here",
        ha="center",
        fontsize=8,
        color=INK,
    )
    ax.set_xlim(0, 80)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("amplitude (dB, CG-compensated, common ref)")
    ax.set_title("(b) same spectra after 2|X|/sum(w) gain correction")
    ax.legend()
    fig.tight_layout()
    _save(fig, fig_dir / "fig3_real_range_spectra.png")

    # ---- natural weak-target scan -----------------------------------------
    cands = scan_natural_weak_targets(db_h, db_r, range_m, strong_bin)
    write_csv(res_dir / "exp2_natural_candidates.csv", cands)
    clean = [c for c in cands if c["verdict"].startswith("clean")]

    # ---- declared synthetic two-tone injection on the real background ------
    comp_h = db20(2.0 * np.abs(spectra["hann"]["X"]) / spectra["hann"]["sum_w"])
    comp_r = db20(2.0 * np.abs(spectra["rect"]["X"]) / spectra["rect"]["sum_w"])
    demo = injection_demo(x, cfg, db_h, comp_h, comp_r, strong_bin, floor_h)
    inj_rows = [
        dict(
            declared="two synthetic tones injected into the real chirp (declared)",
            strong_bin=demo["strong_bin"],
            weak_bin=demo["weak_bin"],
            weak_range_m=float(range_m[demo["weak_bin"]]),
            weak_rel_strong_db=demo["delta_db"],
            separation_bins=demo["weak_bin"] - demo["strong_bin"],
            hann_prom_over_neighbours_db=demo["hann_prom_db"],
            rect_prom_over_neighbours_db=demo["rect_prom_db"],
            hann_resolves=demo["hann_resolves"],
            rect_resolves=demo["rect_resolves"],
            strong_amplitude_lsb=demo["a_strong"],
            weak_amplitude_lsb=demo["a_weak"],
        )
    ]
    write_csv(res_dir / "exp2_injection.csv", inj_rows)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.4))
    sb, wb = demo["strong_bin"], demo["weak_bin"]
    lo_bin, hi_bin = max(int(sb) - 6, 0), min(wb + 9, cfg.rng_nfft // 2 - 1)
    for name in ("rect", "hann"):
        sum_w = spectra[name]["sum_w"]
        ax.plot(
            range_m[lo_bin:hi_bin],
            (db20(2.0 * np.abs(spectra[name]["X"]) / sum_w) - comp_h[strong_bin])[lo_bin:hi_bin],
            color=COLORS[name],
            lw=1.3,
            label=f"{name}, real chirp",
        )
        ci = demo["comp_i"][name]
        ax.plot(
            range_m[lo_bin:hi_bin],
            (ci - ci[strong_bin])[lo_bin:hi_bin],
            color=COLORS[name],
            lw=1.1,
            ls="--",
            label=f"{name}, real + two declared synthetic tones",
        )
    inj_level_rel = float(demo["comp_i"]["hann"][wb] - demo["comp_i"]["hann"][strong_bin])
    ax.axhline(inj_level_rel, color=INK_MUTED, lw=0.7, ls=":")
    ax.annotate(
        "declared weak tone @ bin %d (%.1f m), %.0f dB below the synthetic\n"
        "strong tone, %d bins away:\nhann: %+.1f dB over neighbours -> resolved\n"
        "rect: %+.1f dB over neighbours -> buried under the skirt"
        % (
            wb,
            range_m[wb],
            demo["delta_db"],
            wb - sb,
            demo["hann_prom_db"],
            demo["rect_prom_db"],
        ),
        xy=(range_m[wb], inj_level_rel),
        xytext=(range_m[lo_bin] + 0.2, -56),
        fontsize=8,
        ha="left",
        va="bottom",
        arrowprops=dict(arrowstyle="->", color=INK_MUTED, lw=0.8),
    )
    ax.set_ylim(-58, 4)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("amplitude (dB rel. strong peak)")
    ax.set_title(
        "weak-target visibility: real far-wave chirp + declared injected tone"
        " (injection is synthetic and labelled)"
    )
    ax.legend(loc="center right")
    fig.tight_layout()
    _save(fig, fig_dir / "fig4_weak_target_visibility.png")

    return {
        "file": path.name,
        "sha256": sha,
        "strong_bin": strong_bin,
        "strong_range_m": float(range_m[strong_bin]),
        "cg_offset_rect_to_hann_db": cg_offset_db,
        "natural_candidates": cands,
        "clean_natural_found": len(clean) > 0,
        "injection_strong_bin": demo["strong_bin"],
        "injection_weak_bin": demo["weak_bin"],
        "injection_weak_range_m": float(range_m[demo["weak_bin"]]),
        "injection_delta_db": demo["delta_db"],
        "injection_hann_prom_db": demo["hann_prom_db"],
        "injection_rect_prom_db": demo["rect_prom_db"],
        "hann_resolves": demo["hann_resolves"],
        "rect_resolves": demo["rect_resolves"],
    }


# ---------------------------------------------------------------------------
# Experiment 3: normalisation conventions + 1021-vs-1024 zero fill
# ---------------------------------------------------------------------------
def exp3_normalization(data_dir, project_dir, cfg=pipeline.FAR_F0, channel: int = 0, chirp_index: int = 0) -> dict:
    fig_dir, res_dir = _dirs(project_dir)
    path = pipeline.find_data_file(data_dir, cfg, channel)
    adc, _ = pipeline.load_channel(path, cfg)
    x = adc[chirp_index]
    X, w = pipeline.range_fft(x, cfg, "hann")
    nfft = cfg.rng_nfft
    sum_w = float(w.sum())
    k = int(np.argmax(np.abs(X)))
    db_raw = db20(X)
    range_m = pipeline.range_axis(cfg)

    conventions = [
        ("C1 raw |X|", "20*log10|X|", 0.0),
        ("C2 |X|/N", "20*log10(|X|/%d)" % nfft, -20.0 * np.log10(nfft)),
        ("C3 |X|/sum(w)", "20*log10(|X|/%.1f)" % sum_w, -20.0 * np.log10(sum_w)),
        ("C4 20*log10|X|", "numerically identical to C1", 0.0),
        ("C5 10*log10|X|^2", "identical to C4 (10log10 x^2 = 20log10 x)", 0.0),
        (
            "official MATLAB |X|/255",
            "20*log10(|X|/255)  [rng_fft_scalar = 0xff]",
            -20.0 * np.log10(pipeline.OFFICIAL_RNG_FFT_SCALAR),
        ),
    ]
    conv_rows = [
        dict(
            convention=c,
            formula=f,
            peak_bin=k,
            peak_db=float(db_raw[k] + off),
            offset_vs_c1_db=off,
        )
        for c, f, off in conventions
    ]
    write_csv(res_dir / "exp3_normalization.csv", conv_rows)

    apply_style()
    fig, ax = plt.subplots(figsize=(10, 4.2))
    styles = {
        "C1 raw |X|": (COLORS["hann"], "-"),
        "C2 |X|/N": (COLORS["rect"], "-"),
        "C3 |X|/sum(w)": (COLORS["hamming"], "-"),
        "official MATLAB |X|/255": (ACCENT, "--"),
    }
    for c, _f, off in conventions:
        if c not in styles:
            continue
        color, ls = styles[c]
        ax.plot(
            range_m,
            db_raw + off,
            color=color,
            lw=1.1,
            ls=ls,
            label=f"{c}  (offset {off:+.2f} dB vs C1)",
        )
    ax.set_xlim(0, 80)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("presented level (dB)")
    ax.set_title(
        "same hann spectrum under different dB conventions -- fixed offsets, "
        "C4 = C1 and C5 = C4 numerically"
    )
    ax.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, fig_dir / "fig5_normalization_conventions.png")

    # ---- 1021 valid samples vs forced 1024-sample window --------------------
    Xa, wa = pipeline.range_fft(x, cfg, "hann", valid_n=cfg.adc_sample_num)
    Xb, wb = pipeline.range_fft(x, cfg, "hann", valid_n=cfg.rng_nfft)
    amp_a = 2.0 * np.abs(Xa) / float(wa.sum())
    amp_b = 2.0 * np.abs(Xb) / float(wb.sum())
    db_a, db_b = db20(amp_a), db20(amp_b)
    ka, kb = int(np.argmax(amp_a)), int(np.argmax(amp_b))
    width_a = windows.measured_3db_width_bins(db_a, ka)
    width_b = windows.measured_3db_width_bins(db_b, kb)
    delta = db_b - db_a
    zerofill_row = dict(
        variant_a=f"hann({cfg.adc_sample_num}) on {cfg.adc_sample_num} valid samples, FFT {nfft} (3 zeros)",
        variant_b=f"hann({cfg.rng_nfft}) forced over all {cfg.rng_nfft} samples",
        valid_n=cfg.adc_sample_num,
        nfft=nfft,
        peak_bin_a=ka,
        peak_bin_b=kb,
        peak_delta_db=20.0 * np.log10(amp_b[kb] / amp_a[ka]),
        width3db_a_bins=width_a,
        width3db_b_bins=width_b,
        width3db_delta_bins=width_b - width_a,
        enbw_bins_a=windows.enbw_bins(wa, nfft),
        enbw_bins_b=windows.enbw_bins(wb, nfft),
        delta_db_median=float(np.median(np.abs(delta))),
        delta_db_p95=float(np.percentile(np.abs(delta), 95)),
        delta_db_max=float(np.max(np.abs(delta))),
    )
    write_csv(res_dir / "exp3_zerofill.csv", [zerofill_row])

    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    lo, hi = max(ka - 8, 0), min(ka + 9, nfft // 2 - 1)
    ax.plot(range_m[lo:hi], db_a[lo:hi] - db_a[ka], color=COLORS["hann"], lw=1.4,
            label=f"hann({cfg.adc_sample_num}) + 3 zeros  (width {width_a:.2f} bins)")
    ax.plot(range_m[lo:hi], db_b[lo:hi] - db_b[kb], color=COLORS["rect"], lw=1.4, ls="--",
            label=f"hann({cfg.rng_nfft}) forced       (width {width_b:.2f} bins)")
    ax.axhline(-10.0 * np.log10(2.0), color=INK_MUTED, lw=0.7, ls=":")
    ax.text(lo + 0.2, -10.0 * np.log10(2.0) + 0.4, "-3.01 dB", fontsize=8, color=INK_MUTED)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("amplitude (dB rel. own peak)")
    ax.set_title("(a) mainlobe at the strongest peak")
    ax.legend(loc="lower right")
    ax = axes[1]
    ax.plot(range_m[1:], delta[1:], color=ACCENT, lw=1.0)
    ax.axhline(0.0, color=INK_MUTED, lw=0.7)
    ax.set_xlabel("range (m)")
    ax.set_ylabel("dB(forced 1024) - dB(valid 1021)")
    ax.set_title(
        "(b) spectrum difference, median |d| = %.2f dB, p95 = %.2f dB"
        % (zerofill_row["delta_db_median"], zerofill_row["delta_db_p95"])
    )
    fig.tight_layout()
    _save(fig, fig_dir / "fig6_zerofill_1021_vs_1024.png")

    return {
        "conventions": conv_rows,
        "offset_c2_db": conv_rows[1]["offset_vs_c1_db"],
        "offset_c3_db": conv_rows[2]["offset_vs_c1_db"],
        "offset_official_db": conv_rows[5]["offset_vs_c1_db"],
        "zerofill": zerofill_row,
    }
