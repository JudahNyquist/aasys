"""Interacting Multiple Model estimator.

A single constant-velocity filter tuned tightly enough to be precise on a
straight leg will lag badly the moment the target turns; tuned loosely
enough to follow the turn, it is noisy the rest of the time. IMM sidesteps
the compromise by running a bank of models in parallel and weighting them
by how well each explains the incoming measurements.

The per-model probabilities are worth surfacing in the UI: watching the
turn model overtake the constant-velocity model a beat after a drone banks
is the clearest picture of what the estimator is actually doing.

Models in the bank may have different state dimensions (CV is 6, CA is 9).
Mixing happens in the largest state, with narrower models zero-padded and
their extra covariance rows left uncoupled.
"""

from __future__ import annotations

import numpy as np

from ..models import MotionModel
from .ekf import EKF


class IMM:
    def __init__(self, models: list[MotionModel], x0: np.ndarray,
                 P0: np.ndarray, transition: np.ndarray | None = None,
                 mu0: np.ndarray | None = None,
                 dwell_time_s: float = 8.0) -> None:
        self.models = models
        self.n = len(models)
        dims = {m.dim for m in models}
        if len(dims) != 1:
            raise ValueError(
                f"IMM models must share a state dimension, got {sorted(dims)}; "
                "use the common 9-state builders in tracking.models")
        self.dim = dims.pop()

        # Switching is defined as a *rate*, not a per-step probability.
        #
        # A fixed transition matrix silently assumes every prediction covers
        # the same interval. It does not: prediction runs at the physics rate
        # while measurements arrive far more slowly, so a fixed matrix gets
        # applied many times per measurement and walks the model
        # probabilities to their uniform stationary distribution, erasing the
        # evidence between updates. The symptom is an IMM whose probabilities
        # sit at exactly 1/n forever.
        #
        # Parameterising by mean dwell time and exponentiating over the
        # actual dt makes the chain consistent under subdivision:
        # P(a)P(b) = P(a+b), so how finely time is stepped stops mattering.
        self.dwell_time = float(dwell_time_s)
        self._fixed_transition = (None if transition is None
                                  else np.asarray(transition, dtype=float))
        self.transition = self._transition_for(0.05)
        self.mu = (np.full(self.n, 1.0 / self.n) if mu0 is None
                   else np.asarray(mu0, dtype=float).copy())

        x0 = self._pad_state(np.asarray(x0, dtype=float))
        P0 = self._pad_cov(np.asarray(P0, dtype=float))
        self.filters = [EKF(x0.copy(), P0.copy(), m) for m in models]
        self.last_nis: float | None = None

    def _transition_for(self, dt: float) -> np.ndarray:
        """Markov transition over an interval `dt`.

        Closed form for a symmetric generator with per-model escape rate
        1/dwell_time: exact, and far cheaper than a matrix exponential.
        """
        if self._fixed_transition is not None:
            return self._fixed_transition
        n = self.n
        if n == 1:
            return np.ones((1, 1))
        lam = 1.0 / max(self.dwell_time, 1e-6)
        decay = float(np.exp(-lam * n * max(dt, 0.0) / (n - 1)))
        P = np.full((n, n), (1.0 - decay) / n)
        np.fill_diagonal(P, 1.0 / n + (1.0 - 1.0 / n) * decay)
        return P

    # ------------------------------------------------------------- padding
    def _pad_state(self, x: np.ndarray) -> np.ndarray:
        if len(x) >= self.dim:
            return x[:self.dim].copy()
        out = np.zeros(self.dim)
        out[:len(x)] = x
        return out

    def _pad_cov(self, P: np.ndarray) -> np.ndarray:
        if P.shape[0] >= self.dim:
            return P[:self.dim, :self.dim].copy()
        out = np.eye(self.dim) * 1e3
        out[:P.shape[0], :P.shape[1]] = P
        return out

    def _sub(self, x, P, d):
        return x[:d].copy(), P[:d, :d].copy()

    def _grow(self, x, P, d):
        X = np.zeros(self.dim)
        X[:d] = x[:d]
        Pg = np.eye(self.dim) * 1e2
        Pg[:d, :d] = P[:d, :d]
        return X, Pg

    # ---------------------------------------------------------------- state
    def _stack(self) -> tuple[np.ndarray, np.ndarray]:
        """Bank states and covariances as (n, d) and (n, d, d) arrays.

        Every model in the bank shares one state dimension (enforced in
        `__init__`), so this is a plain stack rather than a padding step.
        """
        return (np.stack([f.x for f in self.filters]),
                np.stack([f.P for f in self.filters]))

    @property
    def x(self) -> np.ndarray:
        return self.mu @ np.stack([f.x for f in self.filters])

    @property
    def P(self) -> np.ndarray:
        X, P = self._stack()
        xbar = self.mu @ X
        d = X - xbar
        out = (np.einsum("i,ijk->jk", self.mu, P)
               + np.einsum("i,ij,ik->jk", self.mu, d, d))
        return 0.5 * (out + out.T)

    @property
    def position(self) -> np.ndarray:
        return self.x[:3]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[3:6]

    def model_probabilities(self) -> dict[str, float]:
        return {m.name: float(p) for m, p in zip(self.models, self.mu)}

    def dominant_model(self) -> str:
        return self.models[int(np.argmax(self.mu))].name

    # -------------------------------------------------------------- cycle
    def _mix(self) -> None:
        """Blend each model's estimate with its neighbours', weighted by the
        probability of having switched from them."""
        cbar = self.transition.T @ self.mu
        cbar = np.where(cbar > 1e-12, cbar, 1e-12)
        mix = (self.transition * self.mu[:, None]) / cbar[None, :]

        X, P = self._stack()
        # Xm[j] = sum_i mix[i,j] X[i] -- each model's estimate blended with
        # its neighbours'. The spread term uses the deviation of every source
        # model from the blend it is feeding, hence the (i, j, d) array.
        Xm = mix.T @ X
        d = X[:, None, :] - Xm[None, :, :]
        Pm = (np.einsum("ij,ikl->jkl", mix, P)
              + np.einsum("ij,ijk,ijl->jkl", mix, d, d))
        Pm = 0.5 * (Pm + np.swapaxes(Pm, 1, 2))

        for j, f in enumerate(self.filters):
            f.x, f.P = Xm[j].copy(), Pm[j].copy()

        self.mu = cbar

    def predict(self, dt: float) -> None:
        if dt <= 0:
            return
        self.transition = self._transition_for(dt)
        self._mix()
        for f in self.filters:
            f.predict(dt)

    def update(self, meas) -> float:
        likelihoods = np.array([max(f.likelihood(meas), 1e-300)
                                for f in self.filters])
        nis = [f.update(meas) for f in self.filters]

        self.mu = self.mu * likelihoods
        total = self.mu.sum()
        self.mu = (np.full(self.n, 1.0 / self.n) if total <= 0
                   else self.mu / total)

        self.last_nis = float(np.dot(self.mu, nis))
        return self.last_nis

    def likelihood(self, meas) -> float:
        return float(sum(w * f.likelihood(meas)
                         for w, f in zip(self.mu, self.filters)))

    def innovation(self, meas):
        """Innovation of the most probable model -- used for gating."""
        return self.filters[int(np.argmax(self.mu))].innovation(meas)
