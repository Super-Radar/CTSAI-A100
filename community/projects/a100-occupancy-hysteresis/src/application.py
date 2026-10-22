"""Application boundary: local receipt clock, source ordering and explicit epochs.

Point rows use the existing reader's [range_m, angle_radians, speed_mps].
This module schedules no background work; callers must poll on a monotonic clock.
"""

import math
from dataclasses import dataclass

import numpy as np

from study import Occupancy


@dataclass(frozen=True)
class CaptureClock:
    """Externally established device-to-local clock mapping and error bound.

    Error grows away from the anchor by drift_ppm. This class does not infer
    synchronization from packet arrival and cannot validate the supplied bound.
    """

    device_anchor: float
    local_anchor: float
    error_seconds: float = 0.0
    rate: float = 1.0
    drift_ppm: float = 0.0

    def __post_init__(self):
        values = (
            self.device_anchor,
            self.local_anchor,
            self.error_seconds,
            self.rate,
            self.drift_ppm,
        )
        if (
            not np.isfinite(values).all()
            or self.error_seconds < 0
            or self.rate <= 0
            or not 0 <= self.drift_ppm < 1e6
        ):
            raise ValueError(
                "Finite clock anchor, nonnegative error/drift and positive rate required"
            )

    def bounds(self, device_time):
        delta = (float(device_time) - self.device_anchor) * self.rate
        estimate = self.local_anchor + delta
        roundoff = (
            4
            * (math.ulp(float(device_time)) + math.ulp(self.device_anchor))
            * self.rate
        )
        roundoff += 4 * (
            math.ulp(self.local_anchor) + math.ulp(delta) + math.ulp(estimate)
        )
        error = self.error_seconds + abs(delta) * self.drift_ppm * 1e-6 + roundoff
        lower, upper = estimate - error, estimate + error
        if not np.isfinite((lower, upper)).all():
            raise ValueError("Capture clock mapping outside finite range")
        return lower, upper


@dataclass(frozen=True)
class PolarROI:
    near: float = 2.5
    far: float = 3.1
    left_degrees: float = -20.0
    right_degrees: float = 5.0

    def __post_init__(self):
        values = (self.near, self.far, self.left_degrees, self.right_degrees)
        if not np.isfinite(values).all() or not (
            0 <= self.near < self.far
            and -180 <= self.left_degrees < self.right_degrees <= 180
        ):
            raise ValueError("Finite ordered range/angle bounds required")

    def count(self, points):
        points = np.asarray(points, dtype=float)
        if (
            points.ndim != 2
            or points.shape[1] != 3
            or not np.isfinite(points).all()
            or np.any(points[:, 0] < 0)
            or np.any(np.abs(points[:, 1]) > np.pi)
        ):
            raise ValueError("Expected finite Nx3 range/angle-radians/speed rows")
        return int(
            np.count_nonzero(
                (points[:, 0] >= self.near)
                & (points[:, 0] <= self.far)
                & (points[:, 1] >= np.deg2rad(self.left_degrees))
                & (points[:, 1] <= np.deg2rad(self.right_degrees))
            )
        )


class OccupancyApplication:
    """Reject old packets without renewing their lease or inferring a restart.

    Call start_epoch only after an externally confirmed source restart. A lower
    device timestamp alone cannot distinguish restart from delayed delivery.
    """

    def __init__(self, roi=None, *, capture_clock=None, **parameters):
        if capture_clock is not None and not isinstance(capture_clock, CaptureClock):
            raise TypeError("capture_clock must be a CaptureClock")
        self.roi = roi if roi is not None else PolarROI()
        self.machine = Occupancy(**parameters)
        self.capture_clock = capture_clock
        self.requires_capture_clock = capture_clock is not None
        self.capture_deadline = None
        self.epoch = 0
        self.device_time = None
        self.clock = None
        self.events = []
        self._reported = None

    def _clock(self, now):
        if not np.isfinite(now) or (self.clock is not None and now < self.clock):
            self.machine.reset()
            self.capture_deadline = None
            raise ValueError("Local receipt/poll clock must be finite and monotonic")
        self.clock = float(now)

    def _result(self, now, reason, accepted=False, count=None, force=False):
        state = self.machine.state
        fresh_until = None
        if self.machine.last is not None:
            fresh_until = self.machine.last + self.machine.gap
            if self.capture_deadline is not None:
                fresh_until = min(fresh_until, self.capture_deadline)
        if force or state != self._reported:
            self.events.append(
                {
                    "received_at_s": float(now),
                    "epoch": self.epoch,
                    "state": state,
                    "reason": reason,
                }
            )
            self._reported = state
        return {
            "state": state,
            "reason": reason,
            "accepted": accepted,
            "count": count,
            "epoch": self.epoch,
            "fresh_until_s": fresh_until,
            "freshness_basis": "capture_bound"
            if self.requires_capture_clock
            else "receipt",
        }

    def poll(self, now):
        self._clock(now)
        expired = (
            self.machine.last is not None and now - self.machine.last > self.machine.gap
        )
        capture_expired = (
            self.capture_deadline is not None and now > self.capture_deadline
        )
        if capture_expired:
            self.machine.reset()
            self.capture_deadline = None
        self.machine.value_at(now)
        return self._result(
            now,
            "capture_expired" if capture_expired else "stale" if expired else "poll",
        )

    def start_epoch(self, now, *, capture_clock=None):
        if capture_clock is not None and not isinstance(capture_clock, CaptureClock):
            raise TypeError("capture_clock must be a CaptureClock")
        self._clock(now)
        self.epoch += 1
        self.device_time = None
        self.machine.reset()
        self.capture_deadline = None
        self.capture_clock = capture_clock
        self.requires_capture_clock = (
            self.requires_capture_clock or capture_clock is not None
        )
        return self._result(now, "source_restart", force=True)

    def observe(self, now, device_time, points, *, epoch=0, complete=True):
        self.poll(now)
        if type(epoch) is not int or epoch != self.epoch:
            return self._result(now, "wrong_epoch")
        if device_time is None or not np.isfinite(device_time):
            self.machine.reset()
            self.capture_deadline = None
            return self._result(now, "invalid_device_time")
        if self.device_time is not None and device_time <= self.device_time:
            return self._result(now, "duplicate_or_out_of_order")
        deadline = None
        if self.requires_capture_clock:
            if self.capture_clock is None:
                return self._result(now, "missing_capture_clock")
            try:
                lower, _ = self.capture_clock.bounds(device_time)
            except (TypeError, ValueError, OverflowError):
                return self._result(now, "invalid_capture_clock")
            if lower > now:
                return self._result(now, "future_capture")
            deadline = lower + self.machine.gap
            if now > deadline:
                return self._result(now, "old_or_uncertain_capture")
        self.device_time = float(device_time)
        if not complete:
            self.machine.reset()
            self.capture_deadline = None
            return self._result(now, "incomplete_frame")
        try:
            count = self.roi.count(points)
        except (TypeError, ValueError):
            self.machine.reset()
            self.capture_deadline = None
            return self._result(now, "invalid_points")
        self.machine.update(now, count)
        self.capture_deadline = deadline
        return self._result(now, "measurement", accepted=True, count=count)
