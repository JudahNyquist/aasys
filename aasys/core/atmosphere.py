"""Atmosphere model.

Exponential fit to the ISA lower troposphere. Accurate enough well past the
altitudes this simulation cares about, and it matters for two things:
projectile drag (which sets gun time-of-flight and therefore lead angle)
and target ballistics.
"""

from __future__ import annotations

import numpy as np

RHO0 = 1.225        # kg/m^3 at sea level
SCALE_HEIGHT = 8500.0
G = 9.80665
SOUND_SPEED_0 = 340.29


def density(altitude_m):
    return RHO0 * np.exp(-np.maximum(np.asarray(altitude_m, dtype=float), 0.0)
                         / SCALE_HEIGHT)


def gravity_vector() -> np.ndarray:
    """ENU gravity: straight down the +Z axis."""
    return np.array([0.0, 0.0, -G])
