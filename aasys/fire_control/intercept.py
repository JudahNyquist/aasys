"""Intercept solutions.

Everything here consumes a *track estimate*, never ground truth. That is
the whole point of the exercise: a filter that is 2 m off puts the burst 2 m
off, and for an unguided gun there is nothing downstream to recover it.

Two regimes:

* **Guided** -- the interceptor can correct in flight, so a first-order
  intercept point is enough to launch on and guidance closes the rest.
* **Unguided gun** -- the shot is committed at the trigger. It needs the
  target's future position *and* the projectile's time of flight, which
  depend on each other, so the solution is iterated to a fixed point.
"""

from __future__ import annotations

import numpy as np

from ..core.atmosphere import G, density


def predict_state(position, velocity, t, acceleration=None):
    """Propagate a track estimate forward, optionally with acceleration."""
    position = np.asarray(position, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    p = position + velocity * t
    if acceleration is not None:
        p = p + 0.5 * np.asarray(acceleration, dtype=float) * t * t
    return p


def intercept_time_constant_speed(rel_pos, rel_vel, speed):
    """Smallest positive t with |rel_pos + rel_vel*t| = speed*t.

    Expands to the quadratic

        (|v|^2 - s^2) t^2 + 2 (p.v) t + |p|^2 = 0

    Returns None when the interceptor simply cannot catch the target -- a
    real condition worth reporting rather than papering over, since it is
    how an engagement gets declined for being out of envelope.
    """
    p = np.asarray(rel_pos, dtype=float)
    v = np.asarray(rel_vel, dtype=float)
    a = float(v @ v) - float(speed) ** 2
    b = 2.0 * float(p @ v)
    c = float(p @ p)

    if abs(a) < 1e-9:
        if abs(b) < 1e-12:
            return None
        t = -c / b
        return t if t > 0 else None

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return None
    root = np.sqrt(disc)
    roots = sorted(t for t in ((-b - root) / (2 * a), (-b + root) / (2 * a))
                   if t > 1e-6)
    return roots[0] if roots else None


def solve_intercept(launcher_pos, target_pos, target_vel, speed,
                    target_accel=None, max_iter: int = 12, tol: float = 1e-3,
                    max_tof: float = 120.0):
    """Intercept point for a constant-speed interceptor.

    With no target acceleration this is the closed-form quadratic. With
    acceleration the closed form no longer applies, so it iterates the fixed
    point t <- |p_target(t) - p_launch| / speed.

    That iteration converges in a handful of steps for sensible geometry, and
    *diverges* otherwise -- which is not a hypothetical. The predicted
    position carries a term in a*t^2/2, so each pass grows t roughly as
    |a|t^2/(2*speed); once that factor exceeds one the sequence runs away
    superexponentially. A track whose estimated acceleration is transiently
    large -- a newly initiated track, or one mid-manoeuvre -- is enough to
    trigger it, and twelve passes are plenty to reach 1e200. The result was
    an aim point far outside any float range, an overflow inside the launcher
    selection that follows, and a missile committed to a garbage solution.

    Divergence is not an error to paper over: it means this interceptor
    cannot catch this target under this extrapolation, which is exactly the
    condition the constant-speed branch already reports by returning None. So
    a run that fails to converge, or converges past `max_tof`, is declined
    the same way and the caller counts it against the envelope.
    """
    launcher_pos = np.asarray(launcher_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    target_vel = np.asarray(target_vel, dtype=float)

    if target_accel is None:
        t = intercept_time_constant_speed(target_pos - launcher_pos,
                                          target_vel, speed)
        if t is None:
            return None, None
        return t, predict_state(target_pos, target_vel, t)

    t = intercept_time_constant_speed(target_pos - launcher_pos,
                                      target_vel, speed)
    if t is None:
        t = float(np.linalg.norm(target_pos - launcher_pos)) / max(speed, 1e-6)

    converged = False
    for _ in range(max_iter):
        aim = predict_state(target_pos, target_vel, t, target_accel)
        t_new = float(np.linalg.norm(aim - launcher_pos)) / max(speed, 1e-6)
        if not np.isfinite(t_new) or t_new > max_tof:
            return None, None          # diverging: no reachable intercept
        if abs(t_new - t) < tol:
            t = t_new
            converged = True
            break
        t = t_new
    if not (converged and np.isfinite(t)) or t <= 0:
        return None, None
    return t, predict_state(target_pos, target_vel, t, target_accel)


# ------------------------------------------------------------------ gunnery
def projectile_time_of_flight(range_m: float, muzzle_speed: float,
                              ballistic_coeff: float, altitude: float = 50.0,
                              steps: int = 24) -> float:
    """Time of flight for a drag-decelerated projectile.

    A vacuum solution (range/speed) is badly optimistic: drag can add tens
    of percent to the flight time at gun ranges, and time of flight is
    exactly what sets the lead angle. Integrating a 1-D drag model along the
    trajectory is cheap and captures the dominant effect.

    `ballistic_coeff` is m/(Cd*A) in kg/m^2 -- higher means better retained
    velocity.
    """
    rho = float(density(altitude))
    k = rho / (2.0 * max(ballistic_coeff, 1e-6))

    v = float(muzzle_speed)
    s = 0.0
    t = 0.0
    dt = range_m / max(muzzle_speed, 1e-6) / steps
    for _ in range(steps * 3):
        if s >= range_m or v <= 1.0:
            break
        v = v - k * v * v * dt
        s += v * dt
        t += dt
    if s < range_m:
        # Never reached: report the vacuum bound so callers see it as long.
        return float(range_m / max(muzzle_speed, 1e-6))
    return t


def solve_gun(gun_pos, target_pos, target_vel, muzzle_speed: float,
              ballistic_coeff: float = 900.0, target_accel=None,
              max_iter: int = 10, tol: float = 1e-3):
    """Firing solution for an unguided gun.

    Returns `(aim_direction, time_of_flight, predicted_impact)`.

    Two coupled unknowns: where the target will be, and how long the round
    takes to get there. Iterated to a fixed point, then the aim point is
    raised by the gravity drop over that flight time -- the superelevation
    a gunner would dial in.
    """
    gun_pos = np.asarray(gun_pos, dtype=float)
    target_pos = np.asarray(target_pos, dtype=float)
    target_vel = np.asarray(target_vel, dtype=float)

    t = float(np.linalg.norm(target_pos - gun_pos)) / max(muzzle_speed, 1e-6)
    aim = target_pos
    for _ in range(max_iter):
        aim = predict_state(target_pos, target_vel, t, target_accel)
        rng = float(np.linalg.norm(aim - gun_pos))
        t_new = projectile_time_of_flight(rng, muzzle_speed, ballistic_coeff,
                                          altitude=float(aim[2]))
        if abs(t_new - t) < tol:
            t = t_new
            break
        t = t_new

    if not np.isfinite(t) or t <= 0:
        return None, None, None

    # Superelevation: launch high by the drop accumulated over the flight.
    lofted = aim.copy()
    lofted[2] += 0.5 * G * t * t
    direction = lofted - gun_pos
    n = float(np.linalg.norm(direction))
    if n < 1e-9:
        return None, None, None
    return direction / n, t, aim
