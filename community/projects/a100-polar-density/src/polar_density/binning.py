"""Polar binning, area normalisation and conservation-preserving re-binning.

Conventions used throughout this module
---------------------------------------
* ``r_edges`` / ``theta_edges`` are *increasing* 1-D boundary arrays.
  Angles are expressed in **degrees** everywhere in the public API (matching
  the CSV data); radian conversion happens internally.
* Bin membership follows the half-open convention ``[edge_i, edge_{i+1})``
  (left-closed, right-open) along both axes.  Values exactly on an interior
  boundary belong to the bin that *starts* at that boundary; values outside
  ``[edges[0], edges[-1])`` are excluded (index -1).
* A sector ``(i, j)`` spans the annulus fragment ``r in [r_i, r_{i+1})`` and
  ``theta in [theta_j, theta_{j+1})``; its geometric area is::

      A[i, j] = deg2rad(dtheta_j) * (r_{i+1}^2 - r_i^2) / 2

* Density is exposure-normalised: ``density = counts / (A * n_frames)``,
  i.e. detections per square metre per frame [detections/(m^2*frame)].
* Grids are parameterised: ``dr`` (m) and ``dtheta`` (deg) defaults are 1 m
  and 1 degree, but any positive bin width / explicit edge array is accepted.
"""

from __future__ import annotations

import numpy as np

# --------------------------------------------------------------------------- #
# Bin assignment (half-open [a, b) convention)                                #
# --------------------------------------------------------------------------- #


def assign_bins(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map values to bin indices with the left-closed/right-open convention.

    Returns integer indices in ``[0, n_edges-2)``; out-of-range values get
    ``-1``.  ``values == edges[k]`` lands in bin ``k`` (for interior ``k``),
    and ``values == edges[-1]`` is excluded, consistently with ``[a, b)``.
    """
    values = np.asarray(values, dtype=float)
    edges = np.asarray(edges, dtype=float)
    idx = np.searchsorted(edges, values, side="right") - 1
    idx[(values < edges[0]) | (values >= edges[-1])] = -1
    return idx


def cell_centers(edges: np.ndarray) -> np.ndarray:
    """Midpoints of adjacent bin edges."""
    edges = np.asarray(edges, dtype=float)
    return 0.5 * (edges[:-1] + edges[1:])


# --------------------------------------------------------------------------- #
# Geometry                                                                    #
# --------------------------------------------------------------------------- #


def sector_areas(r_edges: np.ndarray, theta_edges: np.ndarray) -> np.ndarray:
    """Geometric area of every polar sector, shape ``(n_r, n_theta)``.

    ``A[i, j] = deg2rad(theta_{j+1} - theta_j) * (r_{i+1}^2 - r_i^2) / 2``
    (exact annular-sector area; angles in degrees).
    """
    r_edges = np.asarray(r_edges, dtype=float)
    theta_edges = np.asarray(theta_edges, dtype=float)
    half_du = 0.5 * np.diff(r_edges**2)              # (r2^2 - r1^2) / 2
    dtheta = np.deg2rad(np.diff(theta_edges))        # radians
    return np.outer(half_du, dtheta)                 # shape (n_r, n_theta)


def equal_area_r_edges(r_min: float, r_max: float, n_r: int) -> np.ndarray:
    """Radial edges whose annuli all have equal area (sqrt-spaced).

    With uniform angular division into ``n_theta`` sectors every cell of the
    resulting grid then has *identical* area ``pi (r_max^2 - r_min^2) /
    (n_r * n_theta)``, which removes the r-proportional cell-area bias of an
    equal-angle grid and makes densities directly comparable across range.
    """
    return np.sqrt(np.linspace(float(r_min) ** 2, float(r_max) ** 2, int(n_r) + 1))


# --------------------------------------------------------------------------- #
# Histogramming                                                               #
# --------------------------------------------------------------------------- #


def polar_histogram(
    targets: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
) -> np.ndarray:
    """Accumulate per-sector detection counts from ``(N, 5)`` target rows.

    ``targets`` columns are ``(objid, range_m, speed_ms, angle_deg, snr_db)``
    as produced by :func:`polar_density.parser.parse_capture`.  Points that
    fall outside either edge range are excluded (consistent with the half-open
    convention).
    """
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    r = targets[:, 1]
    theta = targets[:, 3]
    ir = assign_bins(r, r_edges)
    it = assign_bins(theta, theta_edges)
    valid = (ir >= 0) & (it >= 0)
    counts = np.zeros(
        (len(r_edges) - 1, len(theta_edges) - 1), dtype=np.int64
    )
    if valid.any():
        np.add.at(counts, (ir[valid], it[valid]), 1)
    return counts


def cartesian_histogram(
    targets: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
) -> np.ndarray:
    """Counts of the same detections on a Cartesian grid (re-binding check).

    Converts every polar detection ``(range, angle_deg)`` to
    ``x = r cos(theta)``, ``y = r sin(theta)`` and histograms it with the same
    half-open convention, so polar and Cartesian maps carry identical total
    weight over the same covered points.
    """
    targets = np.atleast_2d(np.asarray(targets, dtype=float))
    r = targets[:, 1]
    theta = np.deg2rad(targets[:, 3])
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    ix = assign_bins(x, x_edges)
    iy = assign_bins(y, y_edges)
    valid = (ix >= 0) & (iy >= 0)
    counts = np.zeros((len(x_edges) - 1, len(y_edges) - 1), dtype=np.int64)
    if valid.any():
        np.add.at(counts, (ix[valid], iy[valid]), 1)
    return counts


def density_from_counts(
    counts: np.ndarray, areas: np.ndarray, n_frames: int
) -> np.ndarray:
    """Exposure- and area-normalised density [detections/(m^2*frame)].

    ``density = counts / (areas * n_frames)``: per-sector detections divided
    by the sector's geometric area and by the number of frames the capture
    was exposed for (empty frames included).
    """
    counts = np.asarray(counts, dtype=float)
    areas = np.asarray(areas, dtype=float)
    if counts.shape != areas.shape:
        raise ValueError(
            f"counts shape {counts.shape} does not match areas shape {areas.shape}"
        )
    if n_frames <= 0:
        raise ValueError("n_frames must be a positive exposure length")
    return counts / (areas * float(n_frames))


# --------------------------------------------------------------------------- #
# Conservation-preserving re-binning                                          #
# --------------------------------------------------------------------------- #


def _interval_overlap(src_edges: np.ndarray, dst_edges: np.ndarray) -> np.ndarray:
    """Overlap lengths between source and destination 1-D intervals."""
    lo = np.maximum.outer(src_edges[:-1], dst_edges[:-1])
    hi = np.minimum.outer(src_edges[1:], dst_edges[1:])
    return np.clip(hi - lo, 0.0, None)


def rebin_polar(
    density: np.ndarray,
    r_src: np.ndarray,
    theta_src: np.ndarray,
    r_dst: np.ndarray,
    theta_dst: np.ndarray,
) -> np.ndarray:
    """Re-bin a density field between two polar grids sharing the origin.

    The intersection of two annular sectors (same origin) is again an annular
    sector, so the overlap area factorises exactly into a radial part in
    ``r^2`` and an angular part; the transfer is a pair of matrix products
    with **no approximation**::

        W[k, l]   = sum_{i, j} density[i, j] * Ou[i, k] * Ot[j, l]
        density'  = W / A_dst

    where ``Ou`` holds half the ``r^2``-interval overlaps and ``Ot`` the
    angular overlaps in radians.  This preserves ``sum(density * A)`` exactly
    (up to floating point) whenever the destination grid covers the source
    domain.
    """
    density = np.asarray(density, dtype=float)
    u_src = np.asarray(r_src, dtype=float) ** 2
    u_dst = np.asarray(r_dst, dtype=float) ** 2
    ou = _interval_overlap(u_src, u_dst) * 0.5                    # (n_src_r, n_dst_r)
    ot = _interval_overlap(
        np.deg2rad(np.asarray(theta_src, dtype=float)),
        np.deg2rad(np.asarray(theta_dst, dtype=float)),
    )                                                          # (n_src_t, n_dst_t)
    weight = ou.T @ density @ ot                               # (n_dst_r, n_dst_t)
    areas = sector_areas(r_dst, theta_dst)
    return weight / areas


def cartesian_density(
    targets: np.ndarray,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    n_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Cartesian density map of the same detections (representation control).

    Returns ``(counts, density)`` with ``density = counts / (dx * dy *
    n_frames)`` on the Cartesian grid, so the polar and Cartesian
    representations of one capture can be compared with identical total
    weight.
    """
    counts = cartesian_histogram(targets, x_edges, y_edges)
    areas = np.outer(
        np.diff(np.asarray(x_edges, dtype=float)),
        np.diff(np.asarray(y_edges, dtype=float)),
    )
    return counts, density_from_counts(counts, areas, n_frames)


def default_edges(
    values: np.ndarray, step: float, *, snap: str = "out"
) -> np.ndarray:
    """Uniform edges covering ``values`` with the requested bin width.

    ``snap="out"`` extends the range to whole multiples of ``step`` so the
    default grid (dr = 1 m, dtheta = 1 deg) covers the measured data range.
    """
    values = np.asarray(values, dtype=float)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("cannot build edges from an empty value set")
    lo = np.floor(finite.min() / step) * step if snap == "out" else finite.min()
    hi = np.ceil(finite.max() / step) * step if snap == "out" else finite.max()
    if hi <= lo:
        hi = lo + step
    n = int(np.ceil((hi - lo) / step - 1e-9))
    return lo + step * np.arange(n + 1)
