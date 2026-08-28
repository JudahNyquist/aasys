"""Measurement models.

Each model knows how to predict a measurement from a state (`h`), how that
prediction varies with the state (`H`, the Jacobian), and how to difference
a real measurement against a prediction (`residual`).

Bundling the residual with the model is not incidental. Azimuth wraps at
+/-pi, so a naive `z - h(x)` across that seam produces an innovation of
almost 2*pi and destroys the filter. Every angular quantity must be
differenced through its own model.

State layout is [px, py, pz, vx, vy, vz, ...]; models write into the
columns they touch and leave any higher-order terms (acceleration, turn
rate) at zero, so one model serves CV, CA, and turn-rate states alike.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..core.vecmath import wrap_pi

EPS = 1e-9


class MeasurementModel(ABC):
    """Maps state -> measurement, with a Jacobian and a residual rule."""

    dim: int

    @abstractmethod
    def h(self, x: np.ndarray) -> np.ndarray:
        """Predicted measurement for state `x`."""

    @abstractmethod
    def H(self, x: np.ndarray) -> np.ndarray:
        """Jacobian d(h)/d(x), shape (dim, len(x))."""

    def residual(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        """Innovation z - h(x). Overridden where angles are involved."""
        return np.asarray(z, dtype=float) - self.h(x)


class CartesianPositionModel(MeasurementModel):
    """Direct position observation, e.g. a carved voxel-blob centroid.

    Linear in the state, so the EKF update degenerates to an exact linear
    Kalman update here with no approximation error.
    """

    dim = 3

    def h(self, x: np.ndarray) -> np.ndarray:
        return np.asarray(x, dtype=float)[:3].copy()

    def H(self, x: np.ndarray) -> np.ndarray:
        H = np.zeros((3, len(x)))
        H[:3, :3] = np.eye(3)
        return H


class SphericalModel(MeasurementModel):
    """Radar (range, azimuth, elevation) about a fixed sensor origin.

    Azimuth is measured in the XY plane from +X (East) counter-clockwise
    toward +Y (North); elevation rises from that plane toward +Z.
    """

    dim = 3

    def __init__(self, origin: np.ndarray) -> None:
        self.origin = np.asarray(origin, dtype=float)

    def _rel(self, x: np.ndarray) -> tuple[np.ndarray, float, float]:
        p = np.asarray(x, dtype=float)[:3] - self.origin
        rho = float(np.hypot(p[0], p[1]))
        r = float(np.linalg.norm(p))
        return p, max(rho, EPS), max(r, EPS)

    def h(self, x: np.ndarray) -> np.ndarray:
        p, rho, r = self._rel(x)
        return np.array([r, np.arctan2(p[1], p[0]), np.arctan2(p[2], rho)])

    def H(self, x: np.ndarray) -> np.ndarray:
        p, rho, r = self._rel(x)
        px, py, pz = p
        H = np.zeros((3, len(x)))
        # d(range)/dp
        H[0, :3] = p / r
        # d(azimuth)/dp -- independent of altitude
        H[1, 0] = -py / (rho * rho)
        H[1, 1] = px / (rho * rho)
        # d(elevation)/dp
        H[2, 0] = -px * pz / (r * r * rho)
        H[2, 1] = -py * pz / (r * r * rho)
        H[2, 2] = rho / (r * r)
        return H

    def residual(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        d = np.asarray(z, dtype=float) - self.h(x)
        d[1] = wrap_pi(d[1])
        d[2] = wrap_pi(d[2])
        return d


class SphericalDopplerModel(SphericalModel):
    """Radar returning (range, azimuth, elevation, range-rate).

    The Doppler channel is what makes radar worth fusing with optics: it
    observes velocity *directly*, which no number of cameras can do. It
    requires at least a 6-element state.
    """

    dim = 4

    def h(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        p, rho, r = self._rel(x)
        v = x[3:6]
        return np.array([
            r,
            np.arctan2(p[1], p[0]),
            np.arctan2(p[2], rho),
            float(np.dot(p, v)) / r,
        ])

    def H(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) < 6:
            raise ValueError("SphericalDopplerModel needs a state with velocity")
        p, rho, r = self._rel(x)
        v = x[3:6]
        pv = float(np.dot(p, v))

        H = np.zeros((4, len(x)))
        H[:3, :] = super().H(x)[:3, :]
        # d(rdot)/dp = v/r - (p.v) p / r^3
        H[3, :3] = v / r - pv * p / (r ** 3)
        # d(rdot)/dv = p/r  (unit line-of-sight)
        H[3, 3:6] = p / r
        return H

    def residual(self, z: np.ndarray, x: np.ndarray) -> np.ndarray:
        d = np.asarray(z, dtype=float) - self.h(x)
        d[1] = wrap_pi(d[1])
        d[2] = wrap_pi(d[2])
        return d
