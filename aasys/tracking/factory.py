"""Standard estimator constructions.

A new track is seeded from one measurement, which pins down position but
says nothing about velocity. The initial covariance therefore has to be
honest about that asymmetry: tight on position (from the measurement's own
R) and deliberately loose on velocity, sized to the fastest target worth
tracking. Seeding velocity confidently at zero is a classic way to make a
filter reject the very manoeuvre that would have confirmed the track.
"""

from __future__ import annotations

import numpy as np

from .filters import EKF, IMM
from .models import ca_model, ct_model, cv_model, cv_model_6


def _initial_cov(dim: int, R: np.ndarray, speed_sigma: float,
                 accel_sigma: float = 15.0) -> np.ndarray:
    P = np.eye(dim)
    # Position block seeded from the measurement, with a floor so an
    # optimistic R cannot make a brand-new track overconfident.
    pos_var = np.maximum(np.diag(R)[:3] if R.shape[0] >= 3 else np.ones(3),
                         1.0)
    P[:3, :3] = np.diag(pos_var * 4.0)
    P[3:6, 3:6] = np.eye(3) * speed_sigma ** 2
    if dim >= 9:
        P[6:9, 6:9] = np.eye(3) * accel_sigma ** 2
    return P


def cv_ekf_factory(q: float = 60.0, speed_sigma: float = 35.0):
    """Single constant-velocity EKF. The baseline."""
    model = cv_model_6(q)

    def make(position: np.ndarray, R: np.ndarray) -> EKF:
        x0 = np.zeros(6)
        x0[:3] = position
        return EKF(x0, _initial_cov(6, R, speed_sigma), model)

    return make


def imm_factory(q_cv: float = 60.0, q_ca: float = 200.0, q_ct: float = 8.0,
                turn_rates_deg_s=(-35.0, -15.0, 15.0, 35.0),
                speed_sigma: float = 35.0,
                dwell_time_s: float = 6.0):
    """IMM over constant-velocity, constant-acceleration, and turn models.

    The bank covers the three things a drone actually does: hold a heading,
    change speed, and bank. `dwell_time_s` is how long a target is expected
    to hold one mode before switching -- short values make the bank
    responsive but jumpy, long ones make it lag genuine manoeuvres.
    """
    models = [cv_model(q_cv), ca_model(q_ca)]
    models += [ct_model(w, q_ct) for w in turn_rates_deg_s]

    def make(position: np.ndarray, R: np.ndarray) -> IMM:
        dim = max(m.dim for m in models)
        x0 = np.zeros(dim)
        x0[:3] = position
        return IMM(models, x0, _initial_cov(dim, R, speed_sigma),
                   dwell_time_s=dwell_time_s)

    return make
