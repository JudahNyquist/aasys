"""The interceptor's own sensor: a passive infrared seeker.

The architecture's one rule is that nothing downstream ever sees ground
truth; truth is only ever handed to a sensor, which returns a measurement,
and to scoring. This file is the sensor the profile of the whole design
leaves room for -- a third sensing channel, but riding on the effector
instead of the turret.

So the seeker reads truth **only inside `observe()`** and hands guidance a
noisy observation of line of sight. Nothing below that ever touches truth;
neither does anything above it. It is a measurement channel with the weight
of the radar or the optical rig, and dropping it into `fire_control` (rather
than `sensing`) is deliberate: it travels with the round, not the battery.

The realism being paid for is handoff. Before lock the chaser flies the
ground track it was launched on. After lock it steers on its own measured
line of sight -- typically far better, because it corrects with truth-derived
geometry. And on loss of lock it falls back to the ground track again, which
is exactly the dangerous moment: a target that manoeuvres while the seeker
is blind goes unmeasured, and a miss is the honest consequence rather than a
number plucked from the air.

Three independent ways to lose lock, matching the three physical causes:

* **Off-boresight / cone exit** -- the target steers outside the gimbal cone
  (or out of range). Geometry, no randomness.
* **Glare / contrast** -- detection is a probability that falls with range;
  a draw can fail anywhere, and fails more often far out and late in the
  terminal phase.
* **Slew** -- the seeker head has a rate limit. When the line of sight
  rotates faster than the head can track, lock drops. A fast crossing target
  at close range is exactly that case.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np

from ..core.vecmath import unit


class SeekState(Enum):
    SEEKING = "seeking"        # hasn't seen anything worth locking
    LOCKED = "locked"          # steering on its own measurement
    DROPPED = "dropped"        # lost lock; coasting on stale aim


@dataclass
class SeekerOut:
    """What guidance is allowed to consume: a measurement, nothing more."""
    state: SeekState
    u_meas: np.ndarray | None = None       # measured LOS direction (unit)
    r_meas: float = 0.0                    # measured range (m)
    sigma_ang: float = 0.0                 # LOS angle noise (rad)
    sigma_r: float = 0.0                   # range noise (m)

    @property
    def locked(self) -> bool:
        return self.state is SeekState.LOCKED


class Seeker:
    def __init__(self, rng, *, range_m: float = 1500.0,
                 fov_deg: float = 70.0,
                 ang_noise_mrad: float = 1.0, range_noise_frac: float = 0.04,
                 p_det_near: float = 0.99, p_det_far: float = 0.25,
                 slew_rate_deg_s: float = 55.0,
                 glare_strike_limit: int = 8) -> None:
        self.rng = rng
        self.range_m = float(range_m)
        self.cos_half_fov = float(np.cos(np.radians(fov_deg) / 2.0))
        self.sigma_ang = float(np.radians(ang_noise_mrad / 1000.0))
        self.sigma_r_frac = float(range_noise_frac)
        self.p_det_near = float(p_det_near)
        self.p_det_far = float(p_det_far)
        self.slew_rad_s = float(np.radians(slew_rate_deg_s))
        # Glare needs consecutive bad draws to break lock: a tracking loop
        # rides through a transient frame, and only a sustained bank of
        # smoke or low contrast knocks it out. Geometry and slew are
        # instantaneous by contrast -- the target is simply not there.
        self.glare_strike_limit = int(glare_strike_limit)

        self.state = SeekState.SEEKING
        self._prev_u: np.ndarray | None = None
        self._glare_strikes = 0

    # ------------------------------------------------------------------ sense
    def observe(self, t: float, dt: float, self_pos, self_vel,
                truth_targets, aim_hint) -> tuple[SeekerOut, str | None]:
        """Observe truth through the measurement channel and return (out, ev).

        `aim_hint` is the ground-track position the chaser is flying toward;
        the seeker locks whatever live target is nearest to it. The returned
        event string is only ever a lock/loss transition log line.
        """
        target = None
        best_a = float("inf")
        aim_hint = np.asarray(aim_hint, dtype=float)
        for g in truth_targets:
            if g.destroyed:
                continue
            a = float(np.linalg.norm(g.position - aim_hint))
            if a < best_a:
                best_a, target = a, g

        if target is None:
            return SeekerOut(SeekState.SEEKING), None

        pos = np.asarray(self_pos, dtype=float)
        vel = np.asarray(self_vel, dtype=float)
        rel = target.position - pos
        d = float(np.linalg.norm(rel))
        u = unit(rel)

        # Geometric gate: inside range and inside the gimbal cone. A chaser
        # still boosting nearly along its rail has no tunnel-vision yet.
        speed = float(np.linalg.norm(vel))
        cone = (d > 1e-6 and speed > 1e-6
                and float(u @ (vel / speed)) >= self.cos_half_fov)
        geom_ok = cone and 0.5 <= d <= self.range_m

        # Glare / contrast gate: detection probability falling with range.
        frac = float(np.clip(d / max(self.range_m, 1e-6), 0.0, 1.0))
        p_det = self.p_det_near + (self.p_det_far - self.p_det_near) * frac
        glare_ok = bool(self.rng.random() < p_det)

        # Slew gate: the head keeps rotating to follow the line of sight.
        slew_ok = True
        if self._prev_u is not None and dt > 0.0:
            c = float(np.clip(u @ self._prev_u, -1.0, 1.0))
            omega = float(np.arccos(c) / dt)
            slew_ok = omega <= self.slew_rad_s
        self._prev_u = u

        reason = None
        if not geom_ok:
            reason = "off-boresight"
            self._glare_strikes = 0
        elif not slew_ok:
            reason = "slew"
            self._glare_strikes = 0
        elif glare_ok:
            self._glare_strikes = 0
        else:
            self._glare_strikes += 1
            if self._glare_strikes >= self.glare_strike_limit:
                reason = "glare"

        event: str | None = None
        if reason is None:
            if self.state is not SeekState.LOCKED:
                event = f"SEEKER LOCK  range {d:.0f} m"
            self.state = SeekState.LOCKED
        elif self.state is SeekState.LOCKED:
            event = f"SEEKER LOSS  {reason}  range {d:.0f} m"
            self.state = SeekState.DROPPED
        else:
            # Never had lock: still merely seeking, not dropped.
            self.state = SeekState.SEEKING

        if self.state is not SeekState.LOCKED:
            return SeekerOut(self.state), event

        # Measurement: LOS direction plus range, each with honest noise.
        n1 = unit(np.cross(u, np.array([0.0, 0.0, 1.0])))
        if float(np.linalg.norm(n1)) < 1e-9:
            n1 = unit(np.cross(u, np.array([1.0, 0.0, 0.0])))
        n2 = np.cross(u, n1)
        u_meas = unit(u + self.sigma_ang * (n1 * self.rng.standard_normal()
                                            + n2 * self.rng.standard_normal()))
        r_meas = d * (1.0 + self.sigma_r_frac * self.rng.standard_normal())
        return (SeekerOut(SeekState.LOCKED, u_meas, r_meas,
                          self.sigma_ang, self.sigma_r_frac * d), event)