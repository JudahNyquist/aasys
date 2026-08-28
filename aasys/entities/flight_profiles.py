"""Target flight behaviours.

A profile answers one question per step: what velocity does this vehicle
*want* right now? The vehicle then tracks that command through a
first-order lag bounded by its acceleration limit, so no profile can make a
target turn harder than its airframe allows. Keeping "intent" separate from
"dynamics" means a new behaviour is a few lines and never has to restate
the physics.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.vecmath import unit


class FlightProfile(ABC):
    """Returns a desired velocity, or None for unpowered ballistic flight."""

    @abstractmethod
    def desired_velocity(self, t: float, pos: np.ndarray,
                         vel: np.ndarray) -> np.ndarray | None:
        ...

    def done(self, t: float, pos: np.ndarray) -> bool:
        return False


class Ballistic(FlightProfile):
    """Unpowered. Gravity and drag only -- a shell or a dead drone."""

    def desired_velocity(self, t, pos, vel):
        return None


class Cruise(FlightProfile):
    """Straight and level on a fixed bearing."""

    def __init__(self, velocity: np.ndarray) -> None:
        self.velocity = np.asarray(velocity, dtype=float)

    def desired_velocity(self, t, pos, vel):
        return self.velocity


class Waypoint(FlightProfile):
    """Fly a list of waypoints at a fixed speed, then hold the last one."""

    def __init__(self, points, speed: float = 25.0,
                 capture_radius: float = 12.0, loop: bool = False) -> None:
        self.points = [np.asarray(p, dtype=float) for p in points]
        self.speed = float(speed)
        self.capture_radius = float(capture_radius)
        self.loop = loop
        self.i = 0

    def desired_velocity(self, t, pos, vel):
        if self.i >= len(self.points):
            return np.zeros(3)
        tgt = self.points[self.i]
        d = tgt - pos
        if np.linalg.norm(d) < self.capture_radius:
            self.i += 1
            if self.loop:
                self.i %= len(self.points)
            elif self.i >= len(self.points):
                return np.zeros(3)
            tgt = self.points[min(self.i, len(self.points) - 1)]
            d = tgt - pos
        return unit(d) * self.speed

    def done(self, t, pos):
        return (not self.loop) and self.i >= len(self.points)


class Hover(FlightProfile):
    """Station-keep. Deliberately included: a hovering drone has near-zero
    radial velocity and falls straight into the radar's MTI clutter notch,
    which is exactly the case the optical channel has to carry alone."""

    def __init__(self, station: np.ndarray) -> None:
        self.station = np.asarray(station, dtype=float)

    def desired_velocity(self, t, pos, vel):
        d = self.station - pos
        n = float(np.linalg.norm(d))
        return unit(d) * min(n, 6.0)


class Orbit(FlightProfile):
    """Circle a point at fixed radius and altitude -- a loitering observer.

    Sustained turn, so it is the case where a constant-velocity filter model
    biases and an IMM earns its keep."""

    def __init__(self, center: np.ndarray, radius: float = 80.0,
                 speed: float = 22.0, clockwise: bool = False) -> None:
        self.center = np.asarray(center, dtype=float)
        self.radius = float(radius)
        self.speed = float(speed)
        self.sign = -1.0 if clockwise else 1.0

    def desired_velocity(self, t, pos, vel):
        r = pos - self.center
        r[2] = 0.0
        rn = float(np.linalg.norm(r))
        if rn < 1e-6:
            return np.array([self.speed, 0.0, 0.0])
        radial = r / rn
        tangent = self.sign * np.array([-radial[1], radial[0], 0.0])
        # Blend tangentially with a correction back onto the ring.
        correction = -radial * np.clip((rn - self.radius) / self.radius, -1, 1)
        climb = np.array([0.0, 0.0, (self.center[2] - pos[2]) * 0.25])
        return unit(tangent + 1.6 * correction) * self.speed + climb


class Jink(FlightProfile):
    """Weaving evasive flight toward a goal.

    The lateral oscillation is what breaks a constant-velocity tracker: the
    filter is always lagging a turn that reverses before it converges."""

    def __init__(self, goal: np.ndarray, speed: float = 30.0,
                 amplitude: float = 18.0, period: float = 4.0,
                 vertical: float = 0.35) -> None:
        self.goal = np.asarray(goal, dtype=float)
        self.speed = float(speed)
        self.amplitude = float(amplitude)
        self.period = float(period)
        self.vertical = float(vertical)

    def desired_velocity(self, t, pos, vel):
        along = unit(self.goal - pos)
        if np.linalg.norm(along) < 1e-9:
            return np.zeros(3)
        # Lateral axis perpendicular to travel, kept horizontal.
        lat = unit(np.cross(along, np.array([0.0, 0.0, 1.0])))
        w = 2.0 * np.pi / self.period
        swing = self.amplitude * np.sin(w * t)
        vert = self.vertical * self.amplitude * np.cos(0.7 * w * t)
        return along * self.speed + lat * swing + np.array([0.0, 0.0, vert])


class TerminalDive(FlightProfile):
    """Cruise to a standoff point, then accelerate into a diving attack."""

    def __init__(self, aim: np.ndarray, cruise_speed: float = 28.0,
                 dive_speed: float = 55.0, trigger_range: float = 140.0) -> None:
        self.aim = np.asarray(aim, dtype=float)
        self.cruise_speed = float(cruise_speed)
        self.dive_speed = float(dive_speed)
        self.trigger_range = float(trigger_range)
        self.diving = False

    def desired_velocity(self, t, pos, vel):
        d = self.aim - pos
        if np.linalg.norm(d) < self.trigger_range:
            self.diving = True
        speed = self.dive_speed if self.diving else self.cruise_speed
        return unit(d) * speed

    def done(self, t, pos):
        return bool(np.linalg.norm(self.aim - pos) < 3.0)
