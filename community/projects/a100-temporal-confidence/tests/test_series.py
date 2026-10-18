"""Series-construction tests: target counts, grid occupancy, cell selection rule."""

import numpy as np

from temporal_ci.parser import Frame, ParseResult
from temporal_ci.series import (
    cell_center,
    occupancy_grid,
    occupancy_series,
    select_representative_cells,
    target_count_series,
)


def _frame(nb: int, r, a) -> Frame:
    r = np.asarray(r, dtype=float)
    a = np.asarray(a, dtype=float)
    return Frame(nb, None, None, np.arange(r.size), r, np.zeros_like(r), a, np.zeros_like(r))


def test_target_count_series():
    res = ParseResult(
        frames=[_frame(0, [2.5], [-30.0]), _frame(1, [1.0, 6.0, 11.0], [0.0, 10.0, -70.0]), _frame(2, [], [])],
        header=["ObjId"],
    )
    assert list(target_count_series(res)) == [1.0, 3.0, 0.0]


def test_occupancy_grid_cells():
    frames = [
        _frame(0, [2.0, 12.0], [-30.0, -30.0]),
        _frame(1, [12.0], [-30.0]),
        _frame(2, [7.0], [15.0]),
    ]
    occ, info = occupancy_grid(ParseResult(frames=frames, header=[]))
    # r_max = 12 -> n_r = 3 bins [0,15); theta bins: -30 -> tb = 15, 15 -> tb = 19
    assert info["n_r"] == 3 and info["n_theta"] == 36 and info["n_cells"] == 108
    assert occ.sum() == 4  # per-frame occupancies: 2 + 1 + 1
    assert len(np.flatnonzero(occ.sum(axis=0))) == 3  # 3 distinct cells overall
    assert occ[0, 0 * 36 + 15] == 1 and occ[1, 2 * 36 + 15] == 1 and occ[2, 1 * 36 + 19] == 1
    assert info["n_detections_out_of_grid"] == 0


def test_select_representative_cells_rule():
    occ = np.zeros((100, 4), dtype=np.uint8)
    occ[:50, 0] = 1  # frac 0.50 -> closest to 0.5
    occ[:34, 1] = 1  # frac 0.34 -> second closest
    occ[:, 2] = 1  # always occupied -> excluded (frac = 1.0)
    occ[:3, 3] = 1  # hits = 3 < MIN hits -> excluded
    cells, meta = select_representative_cells(occ)
    assert cells[0] == 0 and cells[1] == 1
    assert meta["band"] == (0.05, 0.95)
    assert abs(meta["fraction"][0] - 0.5) < 1e-12
    assert abs(meta["fraction"][1] - 0.34) < 1e-12


def test_cell_center_mapping():
    info = {"n_theta": 36, "r_step_m": 5.0, "theta_step_deg": 10.0}
    r_c, th_c = cell_center(2 * 36 + 15, info)
    assert r_c == 12.5 and th_c == -25.0


def test_occupancy_series_binary():
    occ = np.zeros((5, 2), dtype=np.uint8)
    occ[[0, 2, 4], 1] = 1
    assert list(occupancy_series(occ, 1)) == [1.0, 0.0, 1.0, 0.0, 1.0]
