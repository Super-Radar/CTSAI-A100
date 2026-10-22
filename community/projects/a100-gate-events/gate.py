"""Timestamp-based, per-identity gate events downstream of a tracker."""

import math
from dataclasses import dataclass


@dataclass
class Track:
    last: float
    stable: int | None = None
    pending: int | None = None
    since: float | None = None
    confirmed_at: float | None = None
    armed: bool = True


class GateCounter:
    def __init__(self, band=0.1, confirm_seconds=0.3, rearm_seconds=0.6, max_gap=0.5):
        if not all(
            math.isfinite(v) for v in (band, confirm_seconds, rearm_seconds, max_gap)
        ):
            raise ValueError("Finite gate parameters required")
        if min(band, confirm_seconds, rearm_seconds) < 0 or max_gap <= 0:
            raise ValueError("Nonnegative band/durations and positive max_gap required")
        self.band, self.confirm, self.rearm, self.gap = (
            band,
            confirm_seconds,
            rearm_seconds,
            max_gap,
        )
        self.epoch = -1
        self.reset()

    def reset(self):
        """Start a distinct aggregation epoch, including explicit source resets."""
        self.epoch += 1
        self.tracks = {}
        self.last_time = None

    def update(self, time, distances):
        """One mapping of persistent track ID -> signed distance per data frame.

        Positive-to-negative is 'negative'; negative-to-positive is 'positive'.
        Missing tracks expire after max_gap. A timestamp reset starts a new
        epoch; callers must retain that boundary when aggregating events.
        """
        if not math.isfinite(time) or any(
            not math.isfinite(d) for d in distances.values()
        ):
            self.reset()
            raise ValueError("Finite timestamp/distances required")
        if self.last_time is not None and time <= self.last_time:
            self.reset()
        self.last_time = time
        self.tracks = {
            key: value
            for key, value in self.tracks.items()
            if time - value.last <= self.gap
        }
        events = []
        # An explicit frame without this identity is not evidence of a
        # continuous dwell. Keep the last stable side until expiry, but require
        # fresh uninterrupted confirmation/rearming when the track reappears.
        for key, track in self.tracks.items():
            if key not in distances:
                track.pending = None
                track.since = None
                if not track.armed:
                    track.confirmed_at = None
        for key, distance in distances.items():
            track = self.tracks.setdefault(key, Track(last=time))
            track.last = time
            side = 1 if distance > self.band else -1 if distance < -self.band else 0
            if side == 0:
                track.pending = None
                track.since = None
                # Time in the ambiguous strip does not rearm a crossing.
                if not track.armed:
                    track.confirmed_at = None
                continue
            if side == track.stable:
                track.pending = None
                track.since = None
                if not track.armed:
                    if track.confirmed_at is None:
                        track.confirmed_at = time
                    if time - track.confirmed_at >= self.rearm - 1e-12:
                        track.armed = True
                continue
            if track.pending != side:
                track.pending = side
                track.since = time
                # A tentative return interrupts the uninterrupted rearm dwell.
                if not track.armed:
                    track.confirmed_at = None
            if time - track.since < self.confirm - 1e-12:
                continue
            if track.stable is not None:
                if track.armed:
                    events.append(
                        {
                            "track_id": key,
                            "epoch": self.epoch,
                            "time_s": time,
                            "direction": "positive" if side > 0 else "negative",
                            "confirmation_started_s": track.since,
                        }
                    )
                track.armed = False
            track.stable = side
            track.confirmed_at = time
            track.pending = None
            track.since = None
        return events
