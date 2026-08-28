"""Motion models: state transition and process noise.

Every builder takes `dt` explicitly and rebuilds `Q` for it. That is not
tidiness -- it is required. Both sensors here deliver measurements at
irregular intervals (the radar as its beam schedule allows, the optical
chain whenever a target is inside the carving volume and unoccluded), so a
`Q` baked for a nominal step would be wrong nearly every update. The
discretisation below integrates the continuous noise over the actual
elapsed time, which is what keeps the filter consistent across gaps.
"""

from __future__ import annotations

import numpy as np

DIM = 3  # spatial dimensions


def _block(mat: np.ndarray) -> np.ndarray:
    """Expand a per-axis (k,k) template into a 3D block matrix, so a 2x2
    kinematic template becomes the 6x6 state matrix.

    This is `np.kron(mat, np.eye(DIM))` written out. The Kronecker product is
    the clearer statement of intent, but these matrices are rebuilt on every
    prediction of every model in every IMM bank -- tens of thousands of times
    a second -- and `kron` spends almost all of its time on generality these
    9x9-at-most matrices never use. Three strided assignments produce a
    bit-identical result about nine times faster; `test_block_matches_kron`
    pins the equivalence.
    """
    m = np.asarray(mat, dtype=float)
    k = m.shape[0]
    out = np.zeros((k * DIM, k * DIM))
    for d in range(DIM):
        out[d::DIM, d::DIM] = m
    return out


# --------------------------------------------------------------- constant velocity
def F_cv(dt: float) -> np.ndarray:
    return _block(np.array([[1.0, dt], [0.0, 1.0]]))


def Q_cv(dt: float, q: float) -> np.ndarray:
    """Continuous white-noise acceleration, exactly integrated over dt.

    `q` is the acceleration power spectral density in (m/s^2)^2/s -- set it
    to the square of the acceleration the target can plausibly pull.
    """
    return _block(np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0],
                            [dt ** 2 / 2.0, dt]])) * q


# ----------------------------------------------------------- constant acceleration
def F_ca(dt: float) -> np.ndarray:
    return _block(np.array([[1.0, dt, 0.5 * dt ** 2],
                            [0.0, 1.0, dt],
                            [0.0, 0.0, 1.0]]))


def Q_ca(dt: float, q: float) -> np.ndarray:
    """Continuous white-noise jerk. `q` is jerk PSD in (m/s^3)^2/s."""
    return _block(np.array([
        [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
        [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
        [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
    ])) * q


# ------------------------------------------------------------- coordinated turn
def F_ct(dt: float, omega: float) -> np.ndarray:
    """Horizontal coordinated turn at rate `omega`, vertical axis constant
    velocity.

    Aircraft and multirotors turn in the horizontal plane far more than they
    manoeuvre vertically, so a full 3D turn model would spend states on
    motion that rarely happens. Near omega=0 the closed form is singular, so
    it falls back to the constant-velocity limit.
    """
    F = np.eye(6)
    if abs(omega) < 1e-6:
        F[0, 3] = F[1, 4] = F[2, 5] = dt
        return F

    w = float(omega)
    s, c = np.sin(w * dt), np.cos(w * dt)
    # x, y coupled through the turn; z stays constant-velocity.
    F[0, 3] = s / w
    F[0, 4] = -(1.0 - c) / w
    F[1, 3] = (1.0 - c) / w
    F[1, 4] = s / w
    F[3, 3] = c
    F[3, 4] = -s
    F[4, 3] = s
    F[4, 4] = c
    F[2, 5] = dt
    return F


def Q_ct(dt: float, q: float) -> np.ndarray:
    """Turn models reuse the CV noise structure; the turn itself is carried
    deterministically by F, so what remains is un-modelled acceleration."""
    return Q_cv(dt, q)


# ------------------------------------------------------- common 9-state forms
# Every model in an IMM bank shares one state space, [pos, vel, acc].
#
# The alternative -- letting each model carry only the states it needs, so CV
# is 6-dim and CA 9-dim -- forces every mixing step to lift narrow estimates
# into the wide space and project them back, and gets the covariance of the
# invented dimensions wrong in subtle ways. Carrying acceleration as an
# unused nuisance state in the CV and turn models costs three floats and
# removes that whole class of bug: those models simply do not let
# acceleration influence the kinematics, while still passing an estimate of
# it through for whichever model does.


def F_cv9(dt: float) -> np.ndarray:
    return _block(np.array([[1.0, dt, 0.0],
                            [0.0, 1.0, 0.0],
                            [0.0, 0.0, 1.0]]))


def Q_cv9(dt: float, q: float, q_acc: float = 0.05) -> np.ndarray:
    Q = _block(np.array([[dt ** 3 / 3.0, dt ** 2 / 2.0, 0.0],
                         [dt ** 2 / 2.0, dt, 0.0],
                         [0.0, 0.0, 0.0]])) * q
    # Let the carried acceleration random-walk rather than going stale.
    Q += _block(np.diag([0.0, 0.0, dt])) * q_acc
    return Q


def F_ct9(dt: float, omega: float) -> np.ndarray:
    F = np.eye(9)
    F[:6, :6] = F_ct(dt, omega)
    return F


def Q_ct9(dt: float, q: float) -> np.ndarray:
    return Q_cv9(dt, q)


# ----------------------------------------------------------------- descriptors
class MotionModel:
    """Pairs a transition builder with its process noise."""

    def __init__(self, name: str, dim: int, F, Q, q: float) -> None:
        self.name = name
        self.dim = dim
        self._F = F
        self._Q = Q
        self.q = q

    def F(self, dt: float) -> np.ndarray:
        return self._F(dt)

    def Q(self, dt: float) -> np.ndarray:
        return self._Q(dt, self.q)

    def __repr__(self) -> str:
        return f"<MotionModel {self.name} dim={self.dim} q={self.q:g}>"


def cv_model(q: float = 4.0) -> MotionModel:
    return MotionModel("CV", 9, F_cv9, Q_cv9, q)


def ca_model(q: float = 20.0) -> MotionModel:
    return MotionModel("CA", 9, F_ca, Q_ca, q)


def ct_model(omega_deg_s: float, q: float = 4.0) -> MotionModel:
    w = np.radians(omega_deg_s)
    return MotionModel(f"CT{omega_deg_s:+.0f}", 9,
                       lambda dt: F_ct9(dt, w), Q_ct9, q)


def cv_model_6(q: float = 4.0) -> MotionModel:
    """Narrow constant-velocity model, for a standalone EKF with no bank."""
    return MotionModel("CV", 6, F_cv, Q_cv, q)
