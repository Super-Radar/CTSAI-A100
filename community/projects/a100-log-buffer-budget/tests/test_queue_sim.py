"""Queue-simulation tests against closed-form / analytic expectations."""

import math

import numpy as np

from buffer_budget.queue_sim import (
    group_arrivals,
    required_capacity,
    simulate,
)


def _clustered_arrivals(seed: int = 7, n_batches: int = 60) -> np.ndarray:
    """Deterministic clustered arrival sequence (same-timestamp batches)."""
    rng = np.random.default_rng(seed)
    times = np.sort(rng.uniform(0.0, 40.0, size=n_batches))
    sizes = rng.integers(1, 12, size=n_batches)
    return np.repeat(times, sizes)


def test_group_arrivals_sizes_sum_to_total():
    t = np.array([2.0, 0.0, 0.0, 1.0])
    ev, sizes = group_arrivals(t)
    assert ev.tolist() == [0.0, 1.0, 2.0]
    assert sizes.tolist() == [2, 1, 1]
    assert int(sizes.sum()) == t.size


def test_periodic_flow_at_service_rate_equal_arrival_rate():
    # One message per second arrives, consumer takes one per second:
    # each message is finished just as the next one arrives.
    t = np.arange(5) * 1.0
    res = simulate(t, rate=1.0)
    assert res.overflow == 0
    assert res.max_backlog == 1
    assert res.completed + res.final_backlog == 5


def test_periodic_underprovisioned_matches_analytic_ceiling():
    # p=1 s, r=0.5 msg/s, N=5: fluid backlog after n arrivals is
    # n - (n-1)*p*r, so peak = 5 - 4*0.5 = 3.0; the message-level server
    # must match it to ceil accuracy and here exactly.
    t = np.arange(5) * 1.0
    res = simulate(t, rate=0.5)
    fluid_peak = 5 - 4 * 1.0 * 0.5
    assert abs(res.max_backlog - fluid_peak) <= 1.0  # ceil-level agreement
    assert res.max_backlog == math.ceil(fluid_peak) == 3

    # Second closed form: p=0.1 s, r=6 msg/s (lambda=10), N=25.
    t2 = np.arange(25) * 0.1
    res2 = simulate(t2, rate=6.0)
    fluid2 = 25 - 24 * 0.1 * 6  # 10.6
    assert res2.max_backlog == math.ceil(fluid2) == 11


def test_same_timestamp_batch_enqueued_atomically():
    # 10 messages at one instant: the whole batch enters the queue at once
    # (nothing can be served "during" a zero-length instant).
    t = np.zeros(10)
    res = simulate(t, rate=100.0)
    assert res.max_backlog == 10
    assert res.completed == 0
    assert res.final_backlog == 10
    res_c4 = simulate(t, rate=100.0, capacity=4)
    assert res_c4.max_backlog == 4
    assert res_c4.overflow == 6
    assert res_c4.final_backlog == 4


def test_known_burst_analytic():
    # 10 messages at t=0, 20 more at t=8, consumer at 1 msg/s:
    # 8 served in between -> peak backlog = (10-8) + 20 = 22.
    t = np.array([0.0] * 10 + [8.0] * 20)
    res = simulate(t, rate=1.0)
    assert res.max_backlog == 22
    assert res.completed == 8
    assert res.final_backlog == 22
    assert res.overflow == 0


def test_capacity_zero_drops_everything():
    t = np.array([0.0, 0.0, 1.0, 2.0])
    res = simulate(t, rate=10.0, capacity=0)
    assert res.overflow == 4
    assert res.max_backlog == 0
    assert res.completed == 0
    assert res.final_backlog == 0


def test_conservation_arrivals_equal_enqueued_plus_overflow():
    t = _clustered_arrivals()
    for rate in (0.5, 2.0, 13.7):
        for cap in (None, 0, 3, 16, 10_000):
            res = simulate(t, rate=rate, capacity=cap)
            assert res.completed + res.final_backlog + res.overflow == t.size
            if cap is not None:
                assert 0 <= res.max_backlog <= cap


def test_max_backlog_monotone_nonincreasing_in_rate():
    t = _clustered_arrivals(seed=11)
    rates = (0.3, 1.0, 2.0, 5.0, 10.0, 100.0, 1000.0)
    backlogs = [simulate(t, rate=r).max_backlog for r in rates]
    assert all(b_next <= b_prev for b_prev, b_next in zip(backlogs, backlogs[1:]))
    demands = [simulate(t, rate=r).demand_peak for r in rates]
    assert all(d_next <= d_prev for d_prev, d_next in zip(demands, demands[1:]))


def test_required_capacity_is_exact_zero_overflow_frontier():
    t = _clustered_arrivals(seed=3)
    rate = 3.7
    c_req = required_capacity(t, rate)
    assert simulate(t, rate, capacity=c_req).overflow == 0
    assert simulate(t, rate, capacity=c_req - 1).overflow > 0
    assert c_req == simulate(t, rate).demand_peak


def test_invalid_arguments_raise():
    t = np.zeros(3)
    for bad in (0.0, -1.0):
        try:
            simulate(t, rate=bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"rate={bad} must raise ValueError")
    try:
        simulate(t, rate=1.0, capacity=-2)
    except ValueError:
        pass
    else:
        raise AssertionError("negative capacity must raise ValueError")
