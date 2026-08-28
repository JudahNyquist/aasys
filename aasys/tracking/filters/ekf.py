"""Extended Kalman filter.

One filter serves both sensors. Each measurement carries its own model, so
an optical blob centroid (linear in the state) and a radar spherical-Doppler
return (thoroughly nonlinear) go through the same `update` call and differ
only in the Jacobian and residual the model supplies. Fusing a new sensor
means writing a model, never touching this class.

The covariance update uses Joseph form. The textbook `(I - KH)P` is
algebraically equivalent but loses symmetry and positive-definiteness after
enough heterogeneous updates at wildly different accuracies -- exactly the
regime here, where a 0.4 m optical fix can follow a 24 m radar fix.
"""

from __future__ import annotations

import numpy as np

from ..models import MotionModel


class EKF:
    def __init__(self, x0: np.ndarray, P0: np.ndarray,
                 model: MotionModel) -> None:
        self.x = np.asarray(x0, dtype=float).copy()
        self.P = np.asarray(P0, dtype=float).copy()
        self.model = model
        self.last_nis: float | None = None

    @property
    def dim(self) -> int:
        return len(self.x)

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        F = self.model.F(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.model.Q(dt)
        self.P = 0.5 * (self.P + self.P.T)

    def innovation(self, meas):
        """Return (residual, innovation covariance, Jacobian)."""
        H = meas.model.H(self.x)
        nu = meas.model.residual(meas.z, self.x)
        S = H @ self.P @ H.T + meas.R
        return nu, 0.5 * (S + S.T), H

    def update(self, meas) -> float:
        """Fold in one measurement. Returns the normalised innovation squared."""
        nu, S, H = self.innovation(meas)
        try:
            Sinv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            return float("inf")

        K = self.P @ H.T @ Sinv
        self.x = self.x + K @ nu

        # Joseph form: stays symmetric and positive-definite even when
        # successive updates differ in accuracy by orders of magnitude.
        I_KH = np.eye(self.dim) - K @ H
        self.P = I_KH @ self.P @ I_KH.T + K @ meas.R @ K.T
        self.P = 0.5 * (self.P + self.P.T)

        self.last_nis = float(nu @ Sinv @ nu)
        return self.last_nis

    def likelihood(self, meas) -> float:
        """Gaussian likelihood of a measurement -- drives IMM mixing and the
        association cost."""
        nu, S, _ = self.innovation(meas)
        try:
            Sinv = np.linalg.inv(S)
            det = float(np.linalg.det(S))
        except np.linalg.LinAlgError:
            return 0.0
        if det <= 0:
            return 0.0
        d2 = float(nu @ Sinv @ nu)
        n = len(nu)
        return float(np.exp(-0.5 * d2) / np.sqrt((2 * np.pi) ** n * det))

    def copy(self) -> "EKF":
        return EKF(self.x.copy(), self.P.copy(), self.model)
