"""Effectors: an unguided gun and a guided interceptor.

They fail in opposite ways, which is the reason to have both.

The **gun** commits everything at the trigger. Its rounds cannot correct, so
the miss distance is the track error at firing time plus dispersion,
propagated over the time of flight. It is the honest instrument for judging
the tracker: degrade the filter and the burst simply misses.

The **interceptor** keeps measuring and correcting, so it forgives a poor
launch solution but costs far more per shot and needs time to fly. It earns
its keep against manoeuvring targets the gun cannot lead.
"""

from __future__ import annotations

import itertools

import numpy as np

from ..core.atmosphere import G, density, gravity_vector
from ..core.integrate import rk4_step
from ..core.vecmath import segment_distance, unit
from .guidance import closing_speed, proportional_navigation
from .seeker import SeekState, Seeker, SeekerOut

_round_ids = itertools.count(1)
_msl_ids = itertools.count(1)


class Projectile:
    """One unguided round."""

    def __init__(self, position, velocity, ballistic_coeff: float = 900.0,
                 lethal_radius: float = 0.6, max_time: float = 12.0,
                 salvo_id: int = 0) -> None:
        self.id = next(_round_ids)
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.asarray(velocity, dtype=float).copy()
        self.ballistic_coeff = float(ballistic_coeff)
        self.lethal_radius = float(lethal_radius)
        self.max_time = float(max_time)
        self.salvo_id = salvo_id
        self.tracer = False        # every Nth round is drawn bright
        self.alive = True
        self.age = 0.0
        # Where the round was at the start of the current step. A round
        # crosses several metres per physics step, so hit detection has to
        # test the segment it swept rather than the point it landed on.
        self.prev_position = self.position.copy()
        # Set once the round has been consumed by an impact, as opposed to
        # merely having expired. A round that hits the ground or times out
        # part-way through a step still swept a segment that step, and that
        # segment is still allowed to have hit something.
        self.spent = False
        self.trail: list[np.ndarray] = []

    def _deriv(self, t, y):
        v = y[3:6]
        speed = float(np.linalg.norm(v))
        a = gravity_vector()
        if speed > 1e-9:
            rho = float(density(y[2]))
            a = a - (rho * speed / (2.0 * self.ballistic_coeff)) * v
        return np.concatenate([v, a])

    def step(self, t: float, dt: float) -> None:
        if not self.alive:
            return
        self.prev_position = self.position.copy()
        y = rk4_step(self._deriv, t, np.concatenate([self.position, self.velocity]), dt)
        self.position, self.velocity = y[:3], y[3:6]
        self.age += dt
        if len(self.trail) < 400:
            self.trail.append(self.position.copy())
        if self.position[2] <= 0.0 or self.age > self.max_time:
            self.alive = False


class Interceptor:
    """A guided pursuit vehicle flying proportional navigation.

    One airframe, two payloads. With missile tuning it is a fast, short-lived
    interceptor that corrects through its whole flight; with `uav=True` it is
    a slower, long-endurance counter-drone that has to *hunt* a manoeuvring
    target over a much longer flight.

    Both carry a seeker, and the difference between them is the cone rather
    than the presence of one. The chaser needs a wide, long-ranged head
    because it approaches slowly from whatever aspect it can get; the missile
    closes fast and nearly head-on, so it needs the last few hundred metres
    resolved precisely and nothing else. Until lock, either one is flying the
    ground track it was launched on, and the step change in guidance quality
    at handoff is the thing worth watching.
    """

    def __init__(self, position, velocity, target_track_id: int,
                 boost_accel: float = 220.0, boost_time: float = 1.4,
                 max_lateral: float = 180.0, nav_constant: float = 4.0,
                 fuze_radius: float = 5.0, max_time: float = 25.0,
                 seeker_range: float = 400.0,
                 seeker_fov_deg: float = 35.0,
                 drag_coeff: float = 0.35, area: float = 0.012,
                 mass: float = 9.0, uav: bool = False,
                 seeker: Seeker | None = None,
                 rng=None) -> None:
        self.id = next(_msl_ids)
        self.position = np.asarray(position, dtype=float).copy()
        self.velocity = np.asarray(velocity, dtype=float).copy()
        self.target_track_id = target_track_id
        self.boost_accel = float(boost_accel)
        self.boost_time = float(boost_time)
        self.max_lateral = float(max_lateral)
        self.N = float(nav_constant)
        self.fuze_radius = float(fuze_radius)
        self.max_time = float(max_time)
        self.seeker_range = float(seeker_range)
        self.seeker_cos = float(np.cos(np.radians(seeker_fov_deg)))
        self.drag_coeff = float(drag_coeff)
        self.area = float(area)
        self.mass = float(mass)

        self.uav = bool(uav)
        self.seeker = seeker
        self.rng = rng if rng is not None else np.random.default_rng(self.id)

        self.alive = True
        self.age = 0.0
        self.detonated = False
        self.miss_distance: float | None = None
        self.seeker_locked = False
        self.trail: list[np.ndarray] = []
        self._prev_range: dict[int, float] = {}
        self.prev_position = self.position.copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def seeker_sees(self, point: np.ndarray) -> bool:
        """Terminal handoff: the onboard seeker has a limited cone and range.

        Until it locks, the missile is flying on the ground track's estimate;
        after, on its own much better look. The moment of handoff is visible
        as a step change in guidance quality."""
        rel = np.asarray(point, dtype=float) - self.position
        d = float(np.linalg.norm(rel))
        if d > self.seeker_range or d < 1e-6:
            return False
        v = self.velocity
        if float(np.linalg.norm(v)) < 1e-6:
            return False
        return float(rel @ v) / (d * float(np.linalg.norm(v))) >= self.seeker_cos

    def guide(self, t: float, dt: float, aim_pos, aim_vel, aim_accel=None,
              seeker_out: SeekerOut | None = None) -> None:
        if not self.alive:
            return

        aim_pos = np.asarray(aim_pos, dtype=float)
        aim_vel = np.asarray(aim_vel, dtype=float)

        # Seeker handoff: while locked, the aim point is built from the
        # seeker's *measured* line of sight and range, not the ground track.
        # The velocity term still comes from the ground track -- a lagged
        # position is what a dropped seeker falls back to, and the miss that
        # follows is the point of the exercise.
        if seeker_out is not None:
            self.seeker_locked = seeker_out.locked
            if seeker_out.locked:
                aim_pos = self.position + seeker_out.r_meas * seeker_out.u_meas
        else:
            self.seeker_locked = self.seeker_sees(aim_pos)

        a = proportional_navigation(self.position, self.velocity, aim_pos,
                                    aim_vel, N=self.N, target_accel=aim_accel,
                                    max_accel=self.max_lateral)

        if self.age < self.boost_time:
            a = a + unit(self.velocity) * self.boost_accel

        rho = float(density(self.position[2]))
        speed = self.speed
        if speed > 1e-9:
            a = a - (0.5 * rho * self.drag_coeff * self.area * speed
                     / self.mass) * self.velocity
        a = a + gravity_vector()

        def deriv(tt, y):
            return np.concatenate([y[3:6], a])

        self.prev_position = self.position.copy()
        y = rk4_step(deriv, t, np.concatenate([self.position, self.velocity]), dt)
        self.position, self.velocity = y[:3], y[3:6]
        self.age += dt
        if len(self.trail) < 2000:
            self.trail.append(self.position.copy())

        if self.position[2] <= 0.0 or self.age > self.max_time:
            self.alive = False

    def check_fuze(self, true_target_pos: np.ndarray, key: int = 0) -> bool:
        """Proximity fuze: detonate at the closest point of approach.

        Firing on range alone would trigger on the way in; waiting for range
        to start increasing catches the actual minimum, which is what a real
        fuze does.

        `key` identifies which target this range belongs to. The caller polls
        the fuze against every live target in turn, so a single previous
        range would be overwritten by a different target between polls and
        the "opening" test would compare two unrelated distances -- correct
        against one target and meaningless against a raid.
        """
        if not self.alive or self.detonated:
            return False

        # Closest approach over the segment just flown, not the range at the
        # sample that happened to end the step. A missile closes at 200+ m/s
        # and steps at 120 Hz, so it jumps 1.7 m at a time through a 5 m fuze
        # radius: triggering on the first sample inside that radius detonates
        # it on the way in, one or two metres short of where it would
        # actually have passed. Every recorded miss distance then sits just
        # under the fuze radius no matter how good the guidance was, which
        # hides exactly the improvement a terminal seeker is bought for.
        endpoint = float(np.linalg.norm(true_target_pos - self.position))
        d = float(segment_distance(self.prev_position[None, :],
                                   self.position[None, :], true_target_pos)[0])
        prev = self._prev_range.get(key)
        opening = prev is not None and endpoint > prev
        self._prev_range[key] = endpoint

        if d <= self.fuze_radius or (opening and d <= self.fuze_radius * 3.0):
            self.detonated = True
            self.alive = False
            self.miss_distance = d
            return True
        return False


def step_projectiles(projectiles, t: float, dt: float) -> None:
    """Advance every live round in one vectorised RK4.

    Rounds are homogeneous and mutually independent -- only the ballistic
    coefficient varies between them -- so stepping them one at a time through
    a Python derivative closure is the same arithmetic done N times with N
    times the interpreter overhead. A single burst puts hundreds of rounds in
    the air at once, and at 120 Hz that closure was the second-largest cost
    in the whole simulation.

    The arithmetic is unchanged: same RK4, same drag law, same order of
    operations per round.
    """
    live = [p for p in projectiles if p.alive]
    if not live:
        return

    pos = np.array([p.position for p in live], dtype=float)
    vel = np.array([p.velocity for p in live], dtype=float)
    inv_2bc = 1.0 / (2.0 * np.array([p.ballistic_coeff for p in live],
                                    dtype=float))[:, None]
    g = gravity_vector()

    def accel(p_, v_):
        speed = np.linalg.norm(v_, axis=1, keepdims=True)
        rho = density(p_[:, 2])[:, None]
        return g - (rho * speed * inv_2bc) * v_

    k1p, k1v = vel, accel(pos, vel)
    k2p, k2v = vel + 0.5 * dt * k1v, accel(pos + 0.5 * dt * k1p,
                                           vel + 0.5 * dt * k1v)
    k3p, k3v = vel + 0.5 * dt * k2v, accel(pos + 0.5 * dt * k2p,
                                           vel + 0.5 * dt * k2v)
    k4p, k4v = vel + dt * k3v, accel(pos + dt * k3p, vel + dt * k3v)

    pos = pos + (dt / 6.0) * (k1p + 2.0 * k2p + 2.0 * k3p + k4p)
    vel = vel + (dt / 6.0) * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)

    for i, p in enumerate(live):
        p.prev_position = p.position
        p.position, p.velocity = pos[i], vel[i]
        p.age += dt
        if len(p.trail) < 400:
            p.trail.append(p.position.copy())
        if p.position[2] <= 0.0 or p.age > p.max_time:
            p.alive = False


def kill_probability(miss_distance: float, lethal_radius: float = 5.0,
                     sharpness: float = 2.0) -> float:
    """Damage falls off with miss distance rather than switching at a radius."""
    if miss_distance <= 0:
        return 1.0
    return float(np.exp(-((miss_distance / lethal_radius) ** sharpness)))
