"""Weapon mounts and firing modes.

Two genuinely different weapons, not one with different numbers.

**AA machine gun.** A cyclic weapon: it lays down a continuous stream while
held on target, at a fixed rounds-per-minute, until the belt runs out or the
barrel overheats. Modelling that as one instantaneous burst misses what
actually determines effectiveness -- the gun walks its stream through the
target's future position over a second or more of sustained fire, and its
hit probability comes from how long it can hold a good solution rather than
from any single shot. Heat is what stops it firing forever.

**Surface-to-air tracking missile.** One round, guided all the way in. It
re-solves the intercept continuously instead of committing at launch, so it
tolerates a far worse initial solution -- but it costs a magazine slot and
needs flight time.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

from ..core.vecmath import unit
from .effectors import Projectile


class FireMode(Enum):
    """What the battery is allowed to engage with."""

    AUTO = "auto"          # layered: gun inside its envelope, missile beyond
    GUN = "gun"            # machine gun only
    MISSILE = "missile"    # tracking missiles only
    HOLD = "hold"          # weapons tight, track but do not fire

    @classmethod
    def parse(cls, s: str) -> "FireMode":
        try:
            return cls(str(s).lower())
        except ValueError:
            raise SystemExit(
                f"unknown fire mode {s!r}; choose from "
                + ", ".join(m.value for m in cls))


class GunMount:
    """A cyclic anti-aircraft gun.

    Fires continuously while commanded rather than in discrete volleys.
    Rounds are emitted at the cyclic rate using a fractional accumulator, so
    the stream is independent of the simulation step size -- at 900 rpm the
    gun puts out fifteen rounds per second whether physics runs at 60 Hz or
    240 Hz.
    """

    def __init__(self, position, muzzle_speed: float = 1000.0,
                 rate_rpm: float = 900.0, belt: int = 900,
                 dispersion_mrad: float = 1.7,
                 heat_per_round: float = 1.0,
                 cool_per_s: float = 5.0,
                 heat_limit: float = 90.0,
                 heat_resume: float = 35.0,
                 tracer_every: int = 4,
                 ballistic_coeff: float = 900.0,
                 rng=None) -> None:
        self.position = np.asarray(position, dtype=float)
        self.muzzle_speed = float(muzzle_speed)
        self.rate = float(rate_rpm) / 60.0
        self.belt = int(belt)
        self.ammo = int(belt)
        self.dispersion = float(dispersion_mrad) * 1e-3
        self.heat_per_round = float(heat_per_round)
        self.cool_per_s = float(cool_per_s)
        self.heat_limit = float(heat_limit)
        self.heat_resume = float(heat_resume)
        self.tracer_every = int(tracer_every)
        self.ballistic_coeff = float(ballistic_coeff)
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self.heat = 0.0
        self.overheated = False
        self.firing = False
        self._accum = 0.0
        self._fired_total = 0

    @property
    def ready(self) -> bool:
        return self.ammo > 0 and not self.overheated

    @property
    def heat_frac(self) -> float:
        return min(self.heat / max(self.heat_limit, 1e-6), 1.0)

    def _thermal(self, dt: float) -> None:
        self.heat = max(0.0, self.heat - self.cool_per_s * dt)
        if self.overheated and self.heat <= self.heat_resume:
            # Hysteresis: resuming at the same threshold that stopped it
            # would chatter on and off every step.
            self.overheated = False

    def update(self, t: float, dt: float, aim_direction=None) -> list[Projectile]:
        """Emit whatever rounds the cyclic rate produced this step.

        `aim_direction` None means hold fire; the mount still cools.
        """
        self._thermal(dt)

        if aim_direction is None or not self.ready:
            self.firing = False
            self._accum = 0.0
            return []

        self.firing = True
        self._accum += self.rate * dt
        n = int(self._accum)
        if n <= 0:
            return []
        self._accum -= n
        n = min(n, self.ammo)

        d = np.asarray(aim_direction, dtype=float)
        perp1 = unit(np.cross(d, [0.0, 0.0, 1.0]))
        if not np.any(perp1):
            perp1 = unit(np.cross(d, [0.0, 1.0, 0.0]))
        perp2 = np.cross(d, perp1)

        out: list[Projectile] = []
        for _ in range(n):
            # Every round is slightly different, which is what turns one
            # firing solution into a hit probability rather than a certainty.
            off = (self.rng.normal(0.0, self.dispersion) * perp1
                   + self.rng.normal(0.0, self.dispersion) * perp2)
            v = unit(d + off) * self.muzzle_speed
            p = Projectile(self.position, v,
                           ballistic_coeff=self.ballistic_coeff)
            self._fired_total += 1
            p.tracer = (self._fired_total % self.tracer_every) == 0
            out.append(p)

        self.ammo -= n
        self.heat += n * self.heat_per_round
        if self.heat >= self.heat_limit:
            self.overheated = True
        return out

    def reload(self) -> None:
        self.ammo = self.belt

    def status(self) -> str:
        state = ("OVERHEAT" if self.overheated
                 else "FIRING" if self.firing
                 else "READY" if self.ready else "EMPTY")
        return (f"gun {state} ammo={self.ammo}/{self.belt} "
                f"heat={self.heat_frac*100:.0f}%")
