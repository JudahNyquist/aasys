"""Vector and rotation helpers.

World frame is ENU: X=East, Y=North, Z=Up. All units SI.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def unit(v: np.ndarray, axis: int = -1) -> np.ndarray:
    """Normalize, leaving zero-length vectors as zeros rather than NaN."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return np.divide(v, n, out=np.zeros_like(v), where=n > EPS)


def norm(v: np.ndarray, axis: int = -1) -> np.ndarray:
    return np.linalg.norm(np.asarray(v, dtype=float), axis=axis)


def skew(v: np.ndarray) -> np.ndarray:
    """Skew-symmetric matrix such that skew(a) @ b == cross(a, b)."""
    x, y, z = v
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def wrap_pi(a: np.ndarray | float) -> np.ndarray | float:
    """Wrap angle(s) to (-pi, pi].

    Used for angular measurement residuals; without this an azimuth
    innovation across the +/-pi seam becomes ~2*pi and blows up the filter.
    """
    return (np.asarray(a) + np.pi) % (2.0 * np.pi) - np.pi


def look_at_rotation(eye: np.ndarray, target: np.ndarray,
                     up: np.ndarray | None = None) -> np.ndarray:
    """World->camera rotation for a camera at `eye` looking at `target`.

    Uses the computer-vision camera convention: +Z_cam forward (into the
    scene), +X_cam right, +Y_cam down. The returned matrix R maps a world
    vector into camera coordinates.
    """
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    if up is None:
        up = np.array([0.0, 0.0, 1.0])
    up = np.asarray(up, dtype=float)

    forward = unit(target - eye)
    if np.linalg.norm(forward) < EPS:
        raise ValueError("look_at_rotation: eye and target coincide")

    # Degenerate when looking straight along `up`; fall back to another axis.
    if abs(float(np.dot(forward, unit(up)))) > 1.0 - 1e-6:
        up = np.array([0.0, 1.0, 0.0])
        if abs(float(np.dot(forward, up))) > 1.0 - 1e-6:
            up = np.array([1.0, 0.0, 0.0])

    right = unit(np.cross(forward, up))
    down = np.cross(forward, right)
    # Rows are the camera axes expressed in world coordinates.
    return np.stack([right, down, forward], axis=0)


def enu_to_spherical(p: np.ndarray) -> tuple[float, float, float]:
    """(range, azimuth, elevation) for an ENU offset vector.

    Azimuth is measured in the XY plane from +X (East) counter-clockwise
    toward +Y (North); elevation rises from that plane toward +Z.
    """
    p = np.asarray(p, dtype=float)
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    rho = np.hypot(x, y)
    r = np.sqrt(rho * rho + z * z)
    return r, np.arctan2(y, x), np.arctan2(z, rho)


def spherical_to_enu(r: float, az: float, el: float) -> np.ndarray:
    ce = np.cos(el)
    return np.stack([r * ce * np.cos(az), r * ce * np.sin(az), r * np.sin(el)],
                    axis=-1)


def segment_distance(a: np.ndarray, b: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Distance from point `q` to each swept segment `a[i] -> b[i]`.

    Fast-moving effectors are sampled far too coarsely for their endpoints to
    stand in for their paths. A gun round covers 8.3 m per 120 Hz step
    against a ~1 m lethal radius, and a missile 1.7 m against a 5 m fuze, so
    testing only where each one *ended up* makes hit and detonation
    distances a property of the step size rather than of the geometry. The
    closest approach over the swept segment is the quantity both actually
    care about.

    `a` and `b` are (N, 3); `q` is a single (3,) point. Returns (N,).
    """
    a = np.atleast_2d(a)
    b = np.atleast_2d(b)
    seg = b - a
    den = np.einsum("ij,ij->i", seg, seg)
    t = np.einsum("ij,ij->i", q - a, seg) / np.where(den > 1e-12, den, 1.0)
    t = np.clip(np.where(den > 1e-12, t, 0.0), 0.0, 1.0)
    return np.linalg.norm(q - (a + t[:, None] * seg), axis=1)
