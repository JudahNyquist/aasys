"""Guidance laws.

Proportional navigation is what real interceptors fly, and it is not the
obvious algorithm. Pure pursuit -- point at the target and chase -- forces
the interceptor into a tail chase it loses against anything fast. PN
instead commands acceleration proportional to the *rotation rate of the
line of sight*, which drives that rotation to zero. A line of sight that
holds a constant bearing while the range closes is a collision, so nulling
its rotation is precisely the condition for intercept, and it produces the
characteristic lead that beats a chase.
"""

from __future__ import annotations

import numpy as np


def los_rate(rel_pos: np.ndarray, rel_vel: np.ndarray) -> np.ndarray:
    """Angular velocity of the line of sight: omega = (r x v) / (r.r)."""
    r2 = float(rel_pos @ rel_pos)
    if r2 < 1e-9:
        return np.zeros(3)
    return np.cross(rel_pos, rel_vel) / r2


def closing_speed(rel_pos: np.ndarray, rel_vel: np.ndarray) -> float:
    n = float(np.linalg.norm(rel_pos))
    if n < 1e-9:
        return 0.0
    return float(-(rel_pos @ rel_vel) / n)


def proportional_navigation(missile_pos, missile_vel, target_pos, target_vel,
                            N: float = 4.0, target_accel=None,
                            max_accel: float = 200.0) -> np.ndarray:
    """True PN, with the APN term when target acceleration is available.

    `N` between 3 and 5 is the usual range: below 3 the interceptor
    under-leads, above 5 it saturates its own airframe on noise.
    """
    rel_pos = np.asarray(target_pos, dtype=float) - np.asarray(missile_pos, dtype=float)
    rel_vel = np.asarray(target_vel, dtype=float) - np.asarray(missile_vel, dtype=float)

    omega = los_rate(rel_pos, rel_vel)
    a_cmd = N * np.cross(omega, np.asarray(missile_vel, dtype=float))

    if target_accel is not None:
        # Augmented PN: feed the target's own manoeuvre forward instead of
        # waiting to observe its effect on the line of sight.
        a_cmd = a_cmd + 0.5 * N * np.asarray(target_accel, dtype=float)

    mag = float(np.linalg.norm(a_cmd))
    if mag > max_accel:
        a_cmd *= max_accel / mag
    return a_cmd
