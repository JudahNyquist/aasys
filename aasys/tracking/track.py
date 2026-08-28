"""A single track and its lifecycle.

Tracks are deliberately not created confident. A new measurement opens a
*tentative* track that must earn confirmation by being hit again on
subsequent looks. That rule is what disposes of optical phantom volumes for
free: a ghost is the intersection of silhouettes from a particular target
geometry, so it moves erratically and vanishes as soon as the geometry
shifts, and it never accumulates the hits confirmation requires. A real
target does.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

_ids = itertools.count(1)


class TrackState(Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    COASTING = "coasting"
    DELETED = "deleted"


@dataclass
class TrackHistory:
    t: list = field(default_factory=list)
    position: list = field(default_factory=list)
    velocity: list = field(default_factory=list)
    nis: list = field(default_factory=list)
    trace_P: list = field(default_factory=list)
    model: list = field(default_factory=list)
    sensors: list = field(default_factory=list)


class Track:
    def __init__(self, filt, t: float, sensor_id: str = "") -> None:
        self.id = next(_ids)
        self.filter = filt
        self.state = TrackState.TENTATIVE
        self.t_created = t
        # Two distinct clocks, and conflating them double-predicts the
        # filter: `t_state` is the time the estimate is valid for and every
        # prediction advances it, while `t_last_hit` is the last time real
        # evidence arrived and only a measurement moves it.
        self.t_state = t
        self.t_last_hit = t
        self.t_updated = t
        self.hits = 1
        self.misses = 0
        self.consecutive_misses = 0
        # Rolling window of hit/miss outcomes for the M-of-N rule.
        self.window: list[bool] = [True]
        self.history = TrackHistory()
        self.sensor_hits: dict[str, int] = {sensor_id: 1} if sensor_id else {}
        self.last_sensor = sensor_id

    # ------------------------------------------------------------- geometry
    @property
    def position(self) -> np.ndarray:
        return self.filter.position

    @property
    def velocity(self) -> np.ndarray:
        return self.filter.velocity

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    @property
    def P(self) -> np.ndarray:
        return self.filter.P

    @property
    def position_cov(self) -> np.ndarray:
        return self.filter.P[:3, :3]

    @property
    def position_sigma(self) -> float:
        """One-sigma radius of the position uncertainty, in metres."""
        return float(np.sqrt(max(np.trace(self.position_cov) / 3.0, 0.0)))

    @property
    def confirmed(self) -> bool:
        return self.state in (TrackState.CONFIRMED, TrackState.COASTING)

    @property
    def age(self) -> float:
        return self.t_updated - self.t_created

    def age_since_update(self, t: float) -> float:
        """Seconds since real evidence last arrived."""
        return t - self.t_last_hit

    def dominant_model(self) -> str:
        fn = getattr(self.filter, "dominant_model", None)
        return fn() if fn else self.filter.model.name

    def model_probabilities(self) -> dict:
        fn = getattr(self.filter, "model_probabilities", None)
        return fn() if fn else {self.filter.model.name: 1.0}

    # -------------------------------------------------------------- updates
    def predict_to(self, t: float) -> None:
        """Advance the estimate to time `t`. Idempotent: calling it twice
        with the same `t` is a no-op rather than a second prediction."""
        dt = t - self.t_state
        if dt > 0:
            self.filter.predict(dt)
            self.t_state = t

    def predict(self, dt: float) -> None:
        if dt > 0:
            self.filter.predict(dt)
            self.t_state += dt

    def register_hit(self, meas, window: int) -> float:
        nis = self.filter.update(meas)
        self.hits += 1
        self.consecutive_misses = 0
        self.t_updated = meas.t
        self.t_last_hit = meas.t
        self.window.append(True)
        del self.window[:-window]
        if meas.sensor_id:
            self.sensor_hits[meas.sensor_id] = \
                self.sensor_hits.get(meas.sensor_id, 0) + 1
            self.last_sensor = meas.sensor_id
        return nis

    def register_miss(self, window: int) -> None:
        self.misses += 1
        self.consecutive_misses += 1
        self.window.append(False)
        del self.window[:-window]

    def record(self, t: float) -> None:
        h = self.history
        h.t.append(t)
        h.position.append(self.position.copy())
        h.velocity.append(self.velocity.copy())
        h.nis.append(self.filter.last_nis)
        h.trace_P.append(float(np.trace(self.position_cov)))
        h.model.append(self.dominant_model())
        h.sensors.append(self.last_sensor)

    def __repr__(self) -> str:
        p = self.position
        return (f"<Track {self.id} {self.state.value} "
                f"pos=({p[0]:.0f},{p[1]:.0f},{p[2]:.0f}) "
                f"v={self.speed:.0f} sigma={self.position_sigma:.1f}m "
                f"hits={self.hits}>")
