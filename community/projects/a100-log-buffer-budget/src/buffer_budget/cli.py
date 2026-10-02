"""One-command reproduction of the a100-log-buffer-budget study.

Run from the project root::

    PYTHONPATH=src python3 -m buffer_budget.cli

Pipeline: parse every ``*_CanData.asc`` in the repository data directory,
keep the files that actually contain messages, compute multi-scale
burstiness statistics, sweep a deterministic consumer queue over service
rates and capacities, then write all CSV tables, five figures and the
Chinese ``results/summary.md``.

Everything is deterministic (see SEED note below); rerunning reproduces the
shipped ``results/`` and ``figures/`` byte-for-byte except for PNG encoder
timestamps.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must import after use("Agg"))
from matplotlib.colors import LogNorm  # noqa: E402

from . import bursts
from .asc_parser import AscLog, parse_asc_dir, sha256_of
from .queue_sim import (
    QueueResult,
    group_arrivals,
    min_rate_no_overflow,
    required_capacity,
    simulate,
)

#: Fixed RNG seed for every stochastic choice in the pipeline (figure-2
#: segment selection among the 3 busiest 100 ms windows).  All statistics
#: and simulations themselves are fully deterministic.
SEED = 20261003

REPRESENTATIVE_FILE = "2026_05_20_16_24_32_Tatget__CanData.asc"

#: Consumer-rate sweep points, as multiples of each file's mean arrival rate.
RATE_MULTIPLIERS: tuple[float, ...] = (
    0.8, 0.9, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0, 3.0, 5.0,
)

#: Capacity grid in messages for the representative-file heatmap.
CAPACITY_GRID: tuple[int, ...] = (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)

_TOP_ID_BARS = 20


# --------------------------------------------------------------------------- paths
def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    # <repo>/community/projects/a100-log-buffer-budget/src/buffer_budget/cli.py
    repo_root = Path(__file__).resolve().parents[5]
    return repo_root / "RadarTools" / "RadarTools_Release" / "data"


# --------------------------------------------------------------------------- csv io
def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _fmt_rate(x: float) -> str:
    return f"{x:.6g}"


# --------------------------------------------------------------------------- figures
def _apply_axis_style(ax, xlabel: str, ylabel: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, which="both")


def fig1_multiscale_envelope(log: AscLog, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    mean_rate = log.mean_rate_msgs_per_s
    for w in bursts.WINDOW_S:
        centers, mr, _ = bursts.sliding_window_rates(log.time_s, log.n_bytes, w, w / 10.0)
        # Quiet windows (rate 0) are plotted as gaps: log-y cannot show zero.
        mr = np.where(mr > 0, mr, np.nan)
        ax.step(centers, mr, where="mid", label=f"{w * 1000:g} ms window", linewidth=1.1)
    ax.axhline(mean_rate, color="k", linestyle="--", linewidth=1.0,
               label=f"mean = {mean_rate:.0f} msg/s")
    ax.set_yscale("log")
    ax.set_ylim(bottom=0.5)
    _apply_axis_style(ax, "Time since log start (s)", "Message arrival rate (messages/s)")
    ax.set_title(
        f"Sliding-window arrival-rate envelope at five scales\n{log.path.name}"
    )
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig2_burst_segment(log: AscLog, out: Path, rng: np.random.Generator) -> None:
    # Deterministically rank 100 ms windows by message count, then pick one of
    # the 3 busiest with the fixed seed (documented stochastic choice).
    centers, mr, _ = bursts.sliding_window_rates(log.time_s, log.n_bytes, 0.10, 0.01)
    top3 = np.argsort(-mr, kind="stable")[:3]
    pick = top3[rng.integers(0, top3.size)]
    seg_center = float(centers[pick])
    half = 1.0
    lo, hi = seg_center - half, seg_center + half

    fig, ax_left = plt.subplots(figsize=(8.5, 5.0))
    c10, r10, _ = bursts.sliding_window_rates(log.time_s, log.n_bytes, 0.01, 0.001)
    mask = (c10 >= lo) & (c10 <= hi)
    r10 = np.where(r10 > 0, r10, np.nan)
    ax_left.step(c10[mask], r10[mask], where="mid", color="tab:blue",
                 label="rate, 10 ms window / 1 ms hop")
    _apply_axis_style(ax_left, f"Time since log start (s), segment [{lo:.2f}, {hi:.2f}]",
                      "Message arrival rate (messages/s)")

    ax_right = ax_left.twinx()
    ev, sizes = _batches_in(log.time_s, lo, hi)
    ax_right.vlines(ev, 0, sizes, color="tab:red", alpha=0.55, linewidth=1.1,
                    label="batch size at each timestamp")
    ax_right.set_ylabel("Same-timestamp batch size (messages)", color="tab:red")
    ax_right.tick_params(axis="y", labelcolor="tab:red")
    ax_right.set_ylim(bottom=0)
    imax = int(np.argmax(sizes))
    ax_right.annotate(
        f"max batch = {int(sizes[imax])} messages\nat t = {ev[imax]:.3f} s",
        xy=(ev[imax], sizes[imax]),
        xytext=(0.03, 0.92), textcoords="axes fraction",
        fontsize=8, color="tab:red",
        arrowprops=dict(arrowstyle="->", color="tab:red", lw=0.8),
    )
    ax_left.set_title(
        "Arrival-rate time series inside one burst segment\n"
        f"(red stems: messages sharing one timestamp) — {log.path.name}"
    )
    lines = ax_left.get_lines() + ax_right.get_lines()
    ax_left.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def _batches_in(time_s: np.ndarray, lo: float, hi: float):
    ev, sizes = group_arrivals(time_s)
    m = (ev >= lo) & (ev <= hi)
    return ev[m], sizes[m]


def fig3_backlog_vs_rate(sweep: dict[str, list[QueueResult]], mean_rates: dict[str, float],
                         out: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    for name, results in sweep.items():
        ratio = [res.rate / mean_rates[name] for res in results]
        backlog = [res.max_backlog for res in results]
        order = np.argsort(ratio)
        ax.plot(np.asarray(ratio)[order], np.asarray(backlog)[order],
                marker="o", markersize=3, linewidth=1.0, label=name)
    ax.axvline(1.0, color="k", linestyle="--", linewidth=1.0,
               label="consumer rate = mean arrival rate")
    ax.set_xscale("log")
    ax.set_yscale("log")
    _apply_axis_style(ax,
                      "Assumed consumer rate / mean arrival rate (dimensionless)",
                      "Peak queue backlog (messages)")
    ax.set_title("Peak backlog of an unbounded queue vs assumed consumer rate\n"
                 "(deterministic single-server simulation, one curve per ASC file)")
    ax.legend(fontsize=6.5, loc="upper right", ncol=2, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig4_capacity_heatmap(rate_grid: list[float], mean_rate: float,
                          cap_grid: tuple[int, ...], results: list[QueueResult],
                          out: Path) -> None:
    nr, nc = len(rate_grid), len(cap_grid)
    mat = np.zeros((nr, nc), dtype=np.int64)
    for i, _r in enumerate(rate_grid):
        for j in range(nc):
            mat[i, j] = results[i * nc + j].overflow

    masked = np.where(mat == 0, np.nan, mat.astype(np.float64))
    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    vmax = max(float(mat.max()), 1.0)
    im = ax.imshow(masked, aspect="auto", origin="lower", interpolation="nearest",
                   norm=LogNorm(vmin=1.0, vmax=vmax),
                   extent=[-0.5, nc - 0.5, -0.5, nr - 0.5], cmap="viridis")

    # Compact cell annotations; 0-overflow cells stay white with a "0".
    for i in range(nr):
        for j in range(nc):
            v = mat[i, j]
            txt = "0" if v == 0 else (f"{v / 1000:.0f}k" if v >= 1000 else f"{v}")
            color = "black" if v == 0 else "white"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.0, color=color)

    # Zero-overflow frontier: between the last overflowing capacity and the
    # first zero-overflow capacity per rate column (index coordinates).
    fx, fy = [], []
    for i in range(nr):
        zero_rows = np.flatnonzero(mat[i] == 0)
        if zero_rows.size:
            fx.append(zero_rows[0] - 0.5)
            fy.append(i)
    if fx:
        ax.plot(fx, fy, color="crimson", linewidth=1.8, label="zero-overflow frontier")

    ax.set_xticks(range(nc))
    ax.set_xticklabels([f"{c}" for c in cap_grid])
    ax.set_yticks(range(nr))
    ax.set_yticklabels([f"{r / mean_rate:.2f}x\n({r:.0f})" for r in rate_grid])
    _apply_axis_style(ax,
                      "Queue capacity (messages)",
                      "Consumer rate\n(multiple of mean rate / msg/s)")
    ax.set_title("Dropped messages (overflow) over consumer rate x queue capacity\n"
                 "representative file "
                 f"{REPRESENTATIVE_FILE} — white cells: no overflow")
    cbar = fig.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Dropped messages (log scale)")
    if fx:
        ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig5_per_id_share(log: AscLog, out: Path) -> None:
    all_ids, _, _ = bursts.per_id_counts(log.can_id)
    total_ids = all_ids.size
    ids, counts, fractions = bursts.per_id_counts(log.can_id)
    n = min(_TOP_ID_BARS, ids.size)
    ids, counts, fractions = ids[:n], counts[:n], fractions[:n]
    labels = [f"0x{i:03X}" for i in ids]

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    ypos = np.arange(n)[::-1]
    ax.barh(ypos, fractions * 100.0, color="tab:blue", edgecolor="none")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    for y, frac, cnt in zip(ypos, fractions, counts):
        ax.text(frac * 100.0 + 0.6, y, f"{cnt:,} ({frac * 100:.1f}%)",
                va="center", fontsize=7)
    _apply_axis_style(ax, "Share of messages (%)", "CAN-FD identifier (hex)")
    ax.set_xlim(0, max(fractions) * 100.0 * 1.28)
    ax.set_title(f"Per-identifier message share, top {n} of {total_ids}\n"
                 f"{log.path.name}")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- summary
def _md_table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * len(header)) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def write_summary_md(
    out: Path,
    logs: list[AscLog],
    stats: dict[str, dict],
    rep_sweep: list[QueueResult],
    cap_results: list[QueueResult],
    rep_lambda: float,
    rep_required_cap: dict[float, int],
) -> None:
    rep = next(l for l in logs if l.path.name == REPRESENTATIVE_FILE)
    rep_stats = stats[REPRESENTATIVE_FILE]
    total_msgs = sum(len(l) for l in logs)
    total_bytes = sum(l.total_bytes for l in logs)

    fac1 = [stats[l.path.name]["burst_factor_w1s"] for l in logs]
    fac10 = [stats[l.path.name]["burst_factor_w10ms"] for l in logs]

    # Zero-overflow minimum rate per capacity (from the swept grid).
    per_cap: dict[int, float] = {}
    for c in CAPACITY_GRID:
        col = [res for res in cap_results if res.capacity == c]
        r = min_rate_no_overflow(col)
        if r is not None:
            per_cap[c] = r

    underrun = next(res for res in rep_sweep
                    if abs(res.rate / rep_lambda - 0.8) < 1e-9)

    demand_rows = []
    for mult in (1.0, 1.05, 1.1, 1.2, 1.5, 2.0, 5.0):
        c_req = rep_required_cap[round(rep_lambda * mult, 6)]
        demand_rows.append([f"{mult:.2f}x ({rep_lambda * mult:.0f} msg/s)",
                            f"{c_req}", f"{c_req * 64 / 1024:.1f} KiB"])

    cap_rows = []
    for c in (16, 32, 64, 128, 256, 1024):
        if c in per_cap:
            r = per_cap[c]
            cap_rows.append([f"{c}", f"{r:.0f}", f"{r / rep_lambda:.2f}x"])

    per_file_rows = []
    for l in logs:
        s = stats[l.path.name]
        per_file_rows.append([
            l.path.name, f"{len(l):,}", f"{s['duration_s']:.2f}",
            f"{s['mean_msg_s']:.1f}", f"{s['peak_1s']:.0f}",
            f"{s['burst_factor_w1s']:.1f}", f"{s['burst_factor_w10ms']:.1f}",
            f"{s['max_batch']}",
        ])

    md = f"""# a100-log-buffer-budget 关键结论

本文件由 `cli.py` 自动生成（全部结论基于**假设的离线消费器**与**内存队列模拟**，
与真实总线占用率无关；见 README「局限」）。

## 1. 数据总览

- 含报文的 ASC 日志 **{len(logs)}** 个，共 **{total_msgs:,}** 条 CAN-FD 记录、
  **{total_bytes / 1e6:.2f} MB**（{total_bytes:,} 字节）日志载荷字节。
- 代表文件 `{REPRESENTATIVE_FILE}`：{len(rep):,} 条 / {rep_stats['duration_s']:.2f} s，
  平均到达率 {rep_lambda:.1f} msg/s。

{_md_table(
    ["文件", "报文数", "时长 (s)", "均值 (msg/s)", "峰值@1s (msg/s)", "突发倍数@1s", "突发倍数@10ms", "最大同时间戳批次"],
    per_file_rows)}

## 2. 突发性

- 1 s 尺度突发倍数（峰值/均值）范围 **{min(fac1):.1f}–{max(fac1):.1f}**，
  10 ms 尺度范围 **{min(fac10):.0f}–{max(fac10):.0f}**：短窗峰值远高于均值，
  日志到达呈强突发而非均匀流。
- 代表文件最大同时间戳批次 **{rep_stats['max_batch']}** 条
  （{rep_stats['n_batches']:,} 个批次，平均 {rep_stats['mean_batch']:.1f} 条/批）：
  日志器把整簇报文记在同一时间戳，任何按消息逐条消费的假设消费器在瞬时必须吸收整批。
- 1 s 窗口到达率变异系数（CV）代表文件为 {rep_stats['cv_1s']:.2f}。

## 3. 假设消费器的最低消费速率与容量建议（代表文件）

对确定性单服务台内存队列（服务速率 r，容量 c，超出即丢弃）：

- **容量充足所需的最低消费速率**：r < 平均到达率（{rep_lambda:.0f} msg/s）时积压发散，
  例如 r = 0.8x（{0.8 * rep_lambda:.0f} msg/s）时日志结束时积压达
  **{underrun.final_backlog:,}** 条；r ≥ 平均到达率后峰值积压才收敛。
- **给定 r 时零溢出的最小容量**（精确值 = 无界运行的需求峰值）：

{_md_table(["消费速率 r", "零溢出最小容量 c (条)", "折合 64B 报文内存"], demand_rows)}

- 反过来，**给定容量 c 时零溢出的最低消费速率**（来自扫描网格，为网格下限）：

{_md_table(["容量 c (条)", "最低 r (msg/s)", "r / 平均到达率"], cap_rows)}

- 网格中 r = 5x（{5 * rep_lambda:.0f} msg/s）时最小容量即降为最大批次
  {rep_required_cap[round(rep_lambda * 5.0, 6)]} 条：消费速率足够快时，
  缓冲预算由**同时间戳批次规模**而非平均流量决定。

## 4. 图

- `figures/fig1_multiscale_rate_envelope.png`：代表文件五尺度到达率包络（log-y）。
- `figures/fig2_burst_segment_batch_stems.png`：突发片段到达率与同时间戳批次。
- `figures/fig3_backlog_vs_rate.png`：最大积压 vs 消费速率（10 文件）。
- `figures/fig4_capacity_overflow_heatmap.png`：容量×速率溢出热图与零溢出边界。
- `figures/fig5_per_id_share.png`：按 ID 报文量占比。

## 5. 适用边界（必须与结论一起读）

以上"最低消费速率/容量"针对**假设的离线消费者**从 ASC 日志回放得到的内存队列，
不是 CAN-FD 总线带宽利用率，也不含驱动/内核缓冲。时间戳分辨率受日志器限制
（同簇报文共享时间戳），真实突发间隔可能更短。本分析未做硬件实测。
"""
    out.write_text(md, encoding="utf-8")


# --------------------------------------------------------------------------- main
def run(data_dir: Path, out_root: Path) -> int:
    results_dir = out_root / "results"
    figures_dir = out_root / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    print(f"[1/6] parsing ASC logs in {data_dir}")
    all_logs = parse_asc_dir(data_dir)
    logs = [l for l in all_logs if len(l) > 0]
    print(f"      {len(all_logs)} ASC files, {len(logs)} with messages, "
          f"{sum(len(l) for l in logs):,} records total")
    for l in logs:
        print(f"      {l.path.name}: {len(l):,} msgs, {l.duration_s:.3f} s")

    with (results_dir / "data_sha256.txt").open("w", encoding="utf-8") as fh:
        fh.write("# sha256  n_messages  filename (Vector ASC logs used by this study)\n")
        for l in logs:
            fh.write(f"{sha256_of(l.path)}  {len(l)}  {l.path.name}\n")

    print("[2/6] burstiness statistics")
    stats: dict[str, dict] = {}
    for l in logs:
        scale = bursts.window_scale_stats(l.time_s, l.n_bytes)
        mean_msg, mean_bytes, duration = bursts.mean_rates(l.time_s, l.n_bytes)
        n_batches, max_batch, mean_batch = bursts.batch_stats(l.time_s)
        stats[l.path.name] = {
            "n": len(l), "bytes": l.total_bytes,
            "duration_s": duration, "mean_msg_s": mean_msg, "mean_bytes_s": mean_bytes,
            "peak_10ms": scale[0.01]["peak_msg_s"],
            "peak_50ms": scale[0.05]["peak_msg_s"],
            "peak_100ms": scale[0.10]["peak_msg_s"],
            "peak_500ms": scale[0.50]["peak_msg_s"],
            "peak_1s": scale[1.00]["peak_msg_s"],
            "peak_bytes_1s": scale[1.00]["peak_bytes_s"],
            "burst_factor_w1s": bursts.burst_factor(scale[1.00]["peak_msg_s"], mean_msg),
            "burst_factor_w10ms": bursts.burst_factor(scale[0.01]["peak_msg_s"], mean_msg),
            "cv_1s": bursts.cv_1s(l.time_s, l.n_bytes),
            "n_batches": n_batches, "max_batch": max_batch, "mean_batch": mean_batch,
        }
    _write_csv(
        results_dir / "burst_summary.csv",
        ["file", "n_messages", "total_bytes", "duration_s", "mean_rate_msgs_per_s",
         "peak_rate_msgs_per_s_w10ms", "peak_rate_msgs_per_s_w50ms",
         "peak_rate_msgs_per_s_w100ms", "peak_rate_msgs_per_s_w500ms",
         "peak_rate_msgs_per_s_w1s", "burst_factor_w1s", "burst_factor_w10ms",
         "cv_1s", "mean_rate_bytes_per_s", "peak_rate_bytes_per_s_w1s",
         "n_batches", "max_batch_same_ts", "mean_batch"],
        [[l.path.name, s["n"], s["bytes"], f"{s['duration_s']:.6f}",
          f"{s['mean_msg_s']:.6f}", f"{s['peak_10ms']:.6f}", f"{s['peak_50ms']:.6f}",
          f"{s['peak_100ms']:.6f}", f"{s['peak_500ms']:.6f}", f"{s['peak_1s']:.6f}",
          f"{s['burst_factor_w1s']:.6f}", f"{s['burst_factor_w10ms']:.6f}",
          f"{s['cv_1s']:.6f}", f"{s['mean_bytes_s']:.6f}", f"{s['peak_bytes_1s']:.6f}",
          s["n_batches"], s["max_batch"], f"{s['mean_batch']:.6f}"]
         for l, s in ((l, stats[l.path.name]) for l in logs)],
    )
    _write_csv(
        results_dir / "per_id_counts.csv",
        ["file", "can_id_hex", "can_id_dec", "count", "fraction"],
        [[l.path.name, f"0x{i:03X}", int(i), int(c), f"{f:.9f}"]
         for l in logs
         for i, c, f in zip(*bursts.per_id_counts(l.can_id))],
    )

    print("[3/6] queue sweep: backlog vs consumer rate (unbounded queue)")
    sweep: dict[str, list[QueueResult]] = {}
    mean_rate_of: dict[str, float] = {}
    backlog_rows: list[list] = []
    for l in logs:
        lam = l.mean_rate_msgs_per_s
        mean_rate_of[l.path.name] = lam
        rates = [lam * m for m in RATE_MULTIPLIERS]
        results = [simulate(l.time_s, r, capacity=None) for r in rates]
        sweep[l.path.name] = results
        for m, res in zip(RATE_MULTIPLIERS, results):
            backlog_rows.append([l.path.name, _fmt_rate(res.rate), f"{m:.6g}",
                                 res.max_backlog, res.final_backlog, res.demand_peak])
    _write_csv(results_dir / "backlog_vs_rate.csv",
               ["file", "rate_msgs_per_s", "rate_over_mean", "max_backlog",
                "final_backlog", "demand_peak"],
               backlog_rows)

    print("[4/6] queue sweep: capacity x rate grid (representative file)")
    rep = next(l for l in logs if l.path.name == REPRESENTATIVE_FILE)
    rep_lambda = rep.mean_rate_msgs_per_s
    rep_rate_grid = [rep_lambda * m for m in RATE_MULTIPLIERS]
    cap_results = [simulate(rep.time_s, r, c)
                   for r in rep_rate_grid for c in CAPACITY_GRID]
    _write_csv(results_dir / "capacity_overflow.csv",
               ["rate_msgs_per_s", "rate_over_mean", "capacity", "overflow",
                "max_backlog", "demand_peak"],
               [[_fmt_rate(res.rate), f"{res.rate / rep_lambda:.6g}", int(res.capacity),
                 res.overflow, res.max_backlog, res.demand_peak] for res in cap_results])
    rep_required_cap = {round(r, 6): required_capacity(rep.time_s, r)
                        for r in rep_rate_grid}

    print("[5/6] figures")
    fig1_multiscale_envelope(rep, figures_dir / "fig1_multiscale_rate_envelope.png")
    fig2_burst_segment(rep, figures_dir / "fig2_burst_segment_batch_stems.png", rng)
    fig3_backlog_vs_rate(sweep, mean_rate_of, figures_dir / "fig3_backlog_vs_rate.png")
    fig4_capacity_heatmap(rep_rate_grid, rep_lambda, CAPACITY_GRID, cap_results,
                          figures_dir / "fig4_capacity_overflow_heatmap.png")
    fig5_per_id_share(rep, figures_dir / "fig5_per_id_share.png")

    print("[6/6] summary.md")
    write_summary_md(results_dir / "summary.md", logs, stats,
                     sweep[REPRESENTATIVE_FILE], cap_results,
                     rep_lambda, rep_required_cap)

    print("done.")
    print(f"  results : {results_dir}")
    print(f"  figures : {figures_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="buffer_budget.cli",
        description="Reproduce the a100-log-buffer-budget offline study.",
    )
    parser.add_argument("--data-dir", type=Path, default=default_data_dir(),
                        help="directory containing *_CanData.asc files "
                             "(default: repository RadarTools data directory)")
    parser.add_argument("--out-dir", type=Path, default=project_root(),
                        help="project root under which results/ and figures/ are written")
    args = parser.parse_args(argv)
    if not args.data_dir.is_dir():
        parser.error(f"data directory not found: {args.data_dir}")
    return run(args.data_dir, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
