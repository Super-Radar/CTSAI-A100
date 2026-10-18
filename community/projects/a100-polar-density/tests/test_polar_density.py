"""Tests for the a100-polar-density study.

Covers: sector-area formula against a hand-computed sample, the half-open
bin-membership convention, conservation of the re-binning transfer, uniform
annulus flatness, hotspot recovery, the multi-frame CSV parser (all three
observed header-width variants, 58/59/62 columns) and empty-frame handling.
"""

from __future__ import annotations

import numpy as np
import pytest

from polar_density import binning as bn
from polar_density import conservation as cons
from polar_density.parser import parse_capture

RNG_SEED = 20261003


# --------------------------------------------------------------------------- #
# Hand-computed geometry                                                      #
# --------------------------------------------------------------------------- #


def test_sector_area_hand_computed():
    # 90-degree sector of the annulus r in [1, 2]:
    # A = (pi/2) * (2^2 - 1^2) / 2 = 3*pi/4
    area = bn.sector_areas(np.array([1.0, 2.0]), np.array([0.0, 90.0]))
    assert area.shape == (1, 1)
    assert area[0, 0] == pytest.approx(3 * np.pi / 4, rel=1e-14)

    # full ring r in [2, 3] over 360 deg: A = pi * (3^2 - 2^2) = 5*pi
    full_ring = bn.sector_areas(np.array([2.0, 3.0]), np.array([0.0, 360.0]))
    assert full_ring[0, 0] == pytest.approx(5 * np.pi, rel=1e-14)

    # areas are additive: splitting a ring angularly sums to the ring area
    split = bn.sector_areas(np.array([2.0, 3.0]), np.array([0.0, 10.0, 350.0, 360.0]))
    assert split.sum() == pytest.approx(5 * np.pi, rel=1e-14)


def test_equal_area_grid_cells_identical():
    r_eq = bn.equal_area_r_edges(5.0, 80.0, 30)
    theta = np.arange(-75.0, 76.0, 5.0)
    areas = bn.sector_areas(r_eq, theta)
    assert np.all(areas > 0)
    # every annulus has identical area -> every sector has identical area
    assert np.allclose(areas, areas[0, 0], rtol=1e-12)
    assert areas.max() / areas.min() == pytest.approx(1.0, rel=1e-12)
    # total area equals the covered 150-degree annulus
    expected_total = np.deg2rad(150.0) * (80.0**2 - 5.0**2) / 2
    assert areas.sum() == pytest.approx(expected_total, rel=1e-12)


# --------------------------------------------------------------------------- #
# Bin membership convention                                                   #
# --------------------------------------------------------------------------- #


def test_half_open_bin_convention():
    edges = np.array([0.0, 1.0, 2.0, 3.0])
    idx = bn.assign_bins(np.array([0.0, 0.5, 1.0, 2.999, 3.0, -0.1, 4.0]), edges)
    # left-closed / right-open: boundary point joins the bin starting there,
    # the global right edge and out-of-range values are excluded (-1)
    assert idx.tolist() == [0, 0, 1, 2, -1, -1, -1]

    # consistency: boundary points are counted exactly once, never dropped
    targets = np.column_stack(
        [
            np.zeros(6),
            np.array([0.0, 1.0, 2.0, 1.5, 2.5, 0.5]),  # range
            np.zeros(6),
            np.array([0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),  # angle
            np.zeros(6),
        ]
    )
    counts = bn.polar_histogram(targets, edges, np.array([0.0, 1.0, 2.0]))
    assert counts.sum() == 6  # nothing double-counted or lost
    assert counts[0, 0] == 1  # r=0 lands in bin 0
    assert counts[1, 0] == 1  # r=1.0 is an interior edge -> bin 1, not bin 0
    assert counts[2, 1] == 2  # r=2.0 edge -> bin 2; includes r=2.5
    assert counts[0, 1] == 1 and counts[1, 1] == 1  # theta=1.0 edge -> col 1


# --------------------------------------------------------------------------- #
# Conservation                                                                #
# --------------------------------------------------------------------------- #


def test_density_identity_single_frame():
    rng = np.random.default_rng(RNG_SEED)
    r = np.sqrt(np.linspace(2.0**2, 9.0**2, 500))  # area-uniform-ish
    theta = rng.uniform(-30.0, 30.0, 500)
    targets = np.column_stack([np.zeros(500), r, np.zeros(500), theta, np.zeros(500)])
    r_edges = np.arange(2.0, 10.0, 1.0)
    t_edges = np.arange(-30.0, 31.0, 2.0)
    counts = bn.polar_histogram(targets, r_edges, t_edges)
    areas = bn.sector_areas(r_edges, t_edges)
    density = bn.density_from_counts(counts, areas, n_frames=1)
    report = cons.check_conservation(density, areas, counts, n_frames=1)
    assert report.passed
    assert report.rel_error < 1e-12
    # exposure normalisation scales density by 1/F
    density7 = bn.density_from_counts(counts, areas, n_frames=7)
    assert np.allclose(density7 * 7, density, rtol=1e-14)
    assert cons.check_conservation(density7, areas, counts, n_frames=7).passed


def test_rebin_conservation():
    rng = np.random.default_rng(RNG_SEED)
    r_src = np.arange(2.0, 20.0, 0.5)
    t_src = np.arange(-60.0, 61.0, 2.0)
    density = rng.random((len(r_src) - 1, len(t_src) - 1)) * 10.0
    areas_src = bn.sector_areas(r_src, t_src)

    # identity re-binning (same grid) must return the field unchanged
    same = bn.rebin_polar(density, r_src, t_src, r_src, t_src)
    assert np.allclose(same, density, rtol=1e-12)

    # to an equal-area grid of the same coverage
    r_dst = bn.equal_area_r_edges(2.0, 19.5, 12)
    t_dst = np.arange(-60.0, 61.0, 6.0)
    rebinned = bn.rebin_polar(density, r_src, t_src, r_dst, t_dst)
    report = cons.check_rebin_conservation(
        density, areas_src, rebinned, bn.sector_areas(r_dst, t_dst)
    )
    assert report.passed
    assert report.rel_error < 1e-12

    # and to a finer nested grid (halved bins sharing the source domain)
    r_half = np.arange(2.0, 19.75, 0.25)  # last edge 19.5 == source right edge
    t_half = np.arange(-60.0, 61.0, 1.0)
    fine = bn.rebin_polar(density, r_src, t_src, r_half, t_half)
    report2 = cons.check_rebin_conservation(
        density, areas_src, fine, bn.sector_areas(r_half, t_half)
    )
    assert report2.passed


def test_cartesian_rebin_same_total_weight():
    rng = np.random.default_rng(RNG_SEED)
    r = np.sqrt(rng.uniform(2.0**2, 12.0**2, 4000))
    theta = rng.uniform(-70.0, 70.0, 4000)
    targets = np.column_stack([np.zeros(4000), r, np.zeros(4000), theta, np.zeros(4000)])
    r_edges = np.arange(2.0, 13.0, 1.0)
    t_edges = np.arange(-70.0, 71.0, 2.0)
    polar_counts = bn.polar_histogram(targets, r_edges, t_edges)
    x = r * np.cos(np.deg2rad(theta))
    y = r * np.sin(np.deg2rad(theta))
    x_edges = bn.default_edges(x, 1.0)
    y_edges = bn.default_edges(y, 1.0)
    cart_counts, cart_density = bn.cartesian_density(
        targets, x_edges, y_edges, n_frames=3
    )
    # same points, both grids cover the data: identical total weight
    assert cart_counts.sum() == polar_counts.sum() == 4000
    areas_cart = np.outer(np.diff(x_edges), np.diff(y_edges))
    assert cons.check_conservation(
        cart_density, areas_cart, cart_counts, n_frames=3
    ).passed


# --------------------------------------------------------------------------- #
# Synthetic benchmarks                                                        #
# --------------------------------------------------------------------------- #


def _annulus_targets(rng, n, r_inner, r_outer, theta_lo, theta_hi):
    u = rng.random(n)
    r = np.sqrt(r_inner**2 + u * (r_outer**2 - r_inner**2))
    theta = rng.uniform(theta_lo, theta_hi, n)
    return np.column_stack([np.zeros(n), r, np.zeros(n), theta, np.zeros(n)])


def test_uniform_annulus_flat_density():
    rng = np.random.default_rng(RNG_SEED)
    targets = _annulus_targets(rng, 600_000, 5.0, 80.0, -75.0, 75.0)
    r_edges = np.arange(5.0, 80.0, 1.0)
    t_edges = np.arange(-75.0, 75.0, 1.0)
    counts = bn.polar_histogram(targets, r_edges, t_edges)
    areas = bn.sector_areas(r_edges, t_edges)
    n_frames = 100
    density = bn.density_from_counts(counts, areas, n_frames)
    assert cons.check_conservation(density, areas, counts, n_frames).passed

    ring_counts = counts.sum(axis=1)
    ring_density = ring_counts / areas.sum(axis=1)
    interior = slice(5, -5)
    flatness = np.max(np.abs(ring_density[interior] - ring_density[interior].mean()))
    flatness /= ring_density[interior].mean()
    # density flat within counting noise (max |dev| over ~70 interior rings)
    assert flatness < 0.10
    # raw counts, in contrast, grow steeply with radius (area bias)
    assert ring_counts[-8] > 5 * ring_counts[1]


def test_hotspot_recovery():
    rng = np.random.default_rng(RNG_SEED)
    background = _annulus_targets(rng, 120_000, 5.0, 80.0, -75.0, 75.0)
    n_hot = 15_000
    r_hs = rng.normal(30.0, 1.0, n_hot)
    t_hs = rng.normal(-10.0, 1.5, n_hot)
    hotspot = np.column_stack(
        [np.zeros(n_hot), r_hs, np.zeros(n_hot), t_hs, np.zeros(n_hot)]
    )
    targets = np.vstack([background, hotspot])
    r_edges = np.arange(5.0, 80.0, 1.0)
    t_edges = np.arange(-75.0, 75.0, 1.0)
    counts = bn.polar_histogram(targets, r_edges, t_edges)
    areas = bn.sector_areas(r_edges, t_edges)
    density = bn.density_from_counts(counts, areas, n_frames=1)

    peak_flat = int(np.argmax(density))
    i, j = np.unravel_index(peak_flat, density.shape)
    r_centers = bn.cell_centers(r_edges)
    t_centers = bn.cell_centers(t_edges)
    assert abs(r_centers[i] - 30.0) <= 1.0  # within one range bin
    assert abs(t_centers[j] - (-10.0)) <= 1.0  # within one angle bin

    # the far-ring area bias is why counting alone misranks hotspots:
    # in the background-only field the largest raw counts sit in the outer
    # rings purely because those sectors are geometrically bigger
    bg_counts = bn.polar_histogram(background, r_edges, t_edges)
    bg_ring = bg_counts.sum(axis=1)
    assert int(np.argmax(bg_ring)) >= len(r_edges) - 12
    # but the density peak of the combined field is at the hotspot, not outer
    assert i < len(r_edges) - 12


# --------------------------------------------------------------------------- #
# Parser                                                                      #
# --------------------------------------------------------------------------- #


def _make_capture_text(n_cols: int, frames: list[list[tuple[str, ...]]]) -> str:
    """Build a synthetic RawTarget capture with a given header width.

    Reproduces the on-disk quirks: single header line, START/timestamp/
    heightOffset frame preamble, per-target rows, ``END,Total:N`` terminator,
    mixed CR/LF endings, dangling trailing commas and stray whitespace.
    """
    header = ",".join(
        ["ObjId", "range", "speed", "angle", "snr"]
        + [f"col{i}" for i in range(5, n_cols)]
    )
    lines = [header]
    endings = ["\n", "\r\n"]
    for nb, rows in enumerate(frames):
        lines.append(f"START,FrameNb:{nb},CameraNb:2029,HeSaiNb:0")
        lines.append("2026/03/10 15:34:39:090---->")
        lines.append(
            "heightOffset:0,1773128079090,0,selfCaliBufCnt:0,Polynom:-0.001023"
        )
        for row in rows:
            padded = [f" {tok} " for tok in row]  # stray whitespace
            cells = padded + ["0"] * (n_cols - len(row)) + [""]  # dangling comma
            lines.append(",".join(cells).rstrip(" ").replace(" ,", ","))
        lines.append(f"END,Total:{len(rows)}")
        lines.append("")  # tolerated blank line between frames
    return "".join(line + endings[i % 2] for i, line in enumerate(lines))


@pytest.mark.parametrize("n_cols", [57, 58, 59, 62])
def test_parser_header_variants(tmp_path, n_cols):
    """Header widths measured across the 22 in-repo RawTarget captures:
    57, 58, 59 and 62 columns (main input: 57)."""
    frames = [
        [
            ("0", "2.31", "0", "-22.45", "36.5"),
            ("1", "14.2", "0", "64.85", "24.5"),
        ],
        [
            ("0", "95.17", "-1.5", "33.85", "16.5"),
        ],
    ]
    path = tmp_path / f"capture_{n_cols}.csv"
    path.write_text(_make_capture_text(n_cols, frames), encoding="utf-8")

    log = parse_capture(path)
    assert log.header.startswith("ObjId,range,speed,angle,snr")
    assert log.header.count(",") + 1 == n_cols
    assert log.n_frames == 2
    assert log.n_targets == 3
    assert len(log.frames[0]) == 2 and len(log.frames[1]) == 1
    np.testing.assert_allclose(
        log.frames[0].targets[0], [0.0, 2.31, 0.0, -22.45, 36.5]
    )
    assert log.frames[0].epoch_ms == 1773128079090.0
    assert log.frames[0].frame_nb == 0 and log.frames[1].frame_nb == 1
    assert log.all_targets().shape == (3, 5)


def test_parser_empty_frame_counted(tmp_path):
    frames = [
        [("0", "10.0", "0", "5.0", "20.0")],
        [],  # START ... END,Total:0 with no target rows
        [("1", "12.0", "0", "6.0", "21.0")],
    ]
    path = tmp_path / "capture_empty.csv"
    path.write_text(_make_capture_text(58, frames), encoding="utf-8")

    log = parse_capture(path)
    assert log.n_frames == 3  # empty frame still contributes exposure
    assert len(log.frames[1]) == 0
    assert log.frames[1].targets.shape == (0, 5)
    assert log.n_targets == 2
    assert log.all_targets().shape == (2, 5)

    # exposure normalisation must use 3 frames, not 2
    r_edges = np.array([9.0, 11.0, 13.0])
    t_edges = np.array([4.0, 5.5, 7.0])
    counts = bn.polar_histogram(log.all_targets(), r_edges, t_edges)
    areas = bn.sector_areas(r_edges, t_edges)
    density = bn.density_from_counts(counts, areas, n_frames=log.n_frames)
    assert density[0, 0] == pytest.approx(1.0 / (areas[0, 0] * 3))
    assert cons.check_conservation(density, areas, counts, n_frames=3).passed


def test_parser_plain_lf_and_no_spaces(tmp_path):
    # minimal, clean file: LF endings, no whitespace quirks, single frame
    text = (
        "ObjId,range,speed,angle,snr\n"
        "START,FrameNb:0\n"
        "heightOffset:0,42\n"
        "7,3.5,1.0,-2.0,9.0\n"
        "END,Total:1\n"
    )
    path = tmp_path / "clean.csv"
    path.write_text(text, encoding="utf-8")
    log = parse_capture(path)
    assert log.n_frames == 1 and log.n_targets == 1
    np.testing.assert_allclose(log.frames[0].targets[0], [7.0, 3.5, 1.0, -2.0, 9.0])
    assert log.frames[0].epoch_ms == 42.0
