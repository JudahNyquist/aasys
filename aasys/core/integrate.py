"""Fixed-step RK4 integration.

Physics runs at a fixed small step decoupled from both the sensor rates and
the render rate, so that reducing the frame rate cannot change the
trajectory. Determinism here is what makes seeded Monte Carlo runs
comparable.
"""

from __future__ import annotations

from typing import Callable

import numpy as np


def rk4_step(f: Callable[[float, np.ndarray], np.ndarray], t: float,
             y: np.ndarray, dt: float) -> np.ndarray:
    k1 = f(t, y)
    k2 = f(t + 0.5 * dt, y + 0.5 * dt * k1)
    k3 = f(t + 0.5 * dt, y + 0.5 * dt * k2)
    k4 = f(t + dt, y + dt * k3)
    return y + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
