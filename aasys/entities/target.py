"""Flying targets.

Powered vehicles track their profile's commanded velocity through a
first-order lag bounded by an acceleration limit; unpowered ones follow
gravity and drag alone. The acceleration limit is what stops a profile
demanding a turn no airframe could fly, which matters because the tracking
filters are tuned against the manoeuvres these limits permit.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..core.atmosphere import density, gravity_vector
from ..core.integrate import rk4_step
from .flight_profiles import Ballistic, FlightProfile

_ids = itertools.count(1)


class Target:
    def __init__(self, position, velocity, profile: FlightProfile | None = None,
                 radius: float = 0.35, rcs: float = 0.02, mass: float = 2.0,
                 drag_coeff: float = 0.6, area: float | None = None,
                 max_accel: float = 30.0, tau: float = 0.45,
                 powered: bool = True, name: str = "") -> None:
        self.id = next(_ids)
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.asarray(velocity, dtype=float).copy()
        self.profile = profile if profile is not None else Ballistic()
        self.radius = float(radius)
        # Radar cross-section. A small quadcopter really is ~0.01-0.05 m^2,
        # which is why radar detection range against one is so short.
        self.rcs = float(rcs)
        self.mass = float(mass)
        self.drag_coeff = float(drag_coeff)
        self.area = float(area) if area is not None else np.pi * self.radius ** 2
        self.max_accel = float(max_accel)
        self.tau = float(tau)
        self.powered = bool(powered)
        self.name = name or f"tgt{self.id}"
        self.alive = True
        self.t_killed: float | None = None
        self.accel = np.zeros(3)

    @property
    def state(self) -> np.ndarray:
        return np.concatenate([self.position, self.velocity])

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def _accel(self, t: float, pos: np.ndarray, vel: np.ndarray) -> np.ndarray:
        v_des = self.profile.desired_velocity(t, pos, vel)

        if v_des is None or not self.powered:
            # Unpowered: gravity plus quadratic drag.
            rho = float(density(pos[2]))
            speed = float(np.linalg.norm(vel))
            drag = np.zeros(3)
            if speed > 1e-9:
                drag = (-0.5 * rho * self.drag_coeff * self.area * speed
                        * vel / self.mass)
            return gravity_vector() + drag

        # Powered: lift trims gravity, so the profile commands directly.
        a = (np.asarray(v_des, dtype=float) - vel) / self.tau
        mag = float(np.linalg.norm(a))
        if mag > self.max_accel:
            a *= self.max_accel / mag
        return a

    def step(self, t: float, dt: float) -> None:
        if not self.alive:
            return

        def deriv(tt, y):
            return np.concatenate([y[3:6], self._accel(tt, y[:3], y[3:6])])

        y = rk4_step(deriv, t, self.state, dt)
        self.accel = self._accel(t, y[:3], y[3:6])
        self.position, self.velocity = y[:3], y[3:6]

        if self.position[2] <= 0.0:
            self.position[2] = 0.0
            if not self.powered:
                self.alive = False

    def kill(self, t: float) -> None:
        """Convert to unpowered wreckage rather than deleting outright, so
        the tracker has to notice the track die instead of being told."""
        self.alive = True
        self.powered = False
        self.profile = Ballistic()
        self.t_killed = t
        self.drag_coeff = 1.1

    @property
    def destroyed(self) -> bool:
        return self.t_killed is not None
