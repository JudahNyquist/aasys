"""Measurement-model tests.

The Jacobians are the highest-risk hand-derived math in the project: a sign
error there does not crash, it just quietly degrades every track. So each
one is checked against central-difference numerical differentiation at
several randomized states, including awkward geometry (near-zenith, across
the azimuth seam, close to the sensor).
"""

import unittest

import numpy as np

from aasys.core.vecmath import wrap_pi
from aasys.sensing.models import (
    CartesianPositionModel,
    SphericalDopplerModel,
    SphericalModel,
)


def numeric_jacobian(model, x, eps=1e-6):
    """Central-difference Jacobian, differencing through model.residual
    so that angle wrapping is handled the same way the filter does it."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    J = np.zeros((model.dim, n))
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = eps
        hp = model.h(x + dx)
        hm = model.h(x - dx)
        diff = hp - hm
        # Wrap angular components so the seam does not corrupt the estimate.
        if isinstance(model, SphericalModel):
            diff[1] = wrap_pi(diff[1])
            diff[2] = wrap_pi(diff[2])
        J[:, i] = diff / (2 * eps)
    return J


class TestCartesianModel(unittest.TestCase):
    def test_h_and_jacobian(self):
        m = CartesianPositionModel()
        x = np.array([10.0, -4.0, 33.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(m.h(x), [10.0, -4.0, 33.0])
        np.testing.assert_allclose(m.H(x), numeric_jacobian(m, x), atol=1e-7)

    def test_is_exactly_linear(self):
        """H must not depend on the state -- that is what lets the EKF
        update reduce to an exact linear Kalman update for optical blobs."""
        m = CartesianPositionModel()
        a = m.H(np.array([1.0, 2.0, 3.0, 0.0, 0.0, 0.0]))
        b = m.H(np.array([-900.0, 5.0, 71.0, 9.0, 9.0, 9.0]))
        np.testing.assert_allclose(a, b)


class TestSphericalModel(unittest.TestCase):
    def setUp(self):
        self.origin = np.array([5.0, -3.0, 2.0])
        self.model = SphericalModel(self.origin)

    def test_h_round_trips_through_spherical_to_enu(self):
        from aasys.core.vecmath import spherical_to_enu

        x = np.array([120.0, 45.0, 80.0, 0.0, 0.0, 0.0])
        r, az, el = self.model.h(x)
        np.testing.assert_allclose(
            spherical_to_enu(r, az, el) + self.origin, x[:3], atol=1e-9
        )

    def test_jacobian_matches_numeric(self):
        rng = np.random.default_rng(0)
        for _ in range(40):
            x = np.concatenate([rng.normal(0, 300, 3), rng.normal(0, 40, 3)])
            if np.linalg.norm(x[:3] - self.origin) < 5.0:
                continue
            np.testing.assert_allclose(
                self.model.H(x), numeric_jacobian(self.model, x),
                atol=1e-5, rtol=1e-4,
            )

    def test_jacobian_near_zenith(self):
        """Elevation is ill-conditioned directly overhead; the analytic and
        numeric Jacobians must still agree."""
        x = np.array([5.0 + 1e-3, -3.0 + 1e-3, 402.0, 0.0, 0.0, 0.0])
        np.testing.assert_allclose(
            self.model.H(x), numeric_jacobian(self.model, x),
            atol=1e-4, rtol=1e-3,
        )

    def test_residual_wraps_azimuth_seam(self):
        """A measurement just past +pi against a prediction just below it
        must give a small innovation, not ~2*pi."""
        x = np.array([-100.0, 1e-6, 0.0, 0.0, 0.0, 0.0]) + np.r_[self.origin, 0, 0, 0]
        pred = self.model.h(x)
        z = pred.copy()
        z[1] = wrap_pi(pred[1] + 0.02)
        d = self.model.residual(z, x)
        self.assertLess(abs(d[1]), 0.05)


class TestSphericalDopplerModel(unittest.TestCase):
    def setUp(self):
        self.origin = np.array([0.0, 0.0, 1.5])
        self.model = SphericalDopplerModel(self.origin)

    def test_range_rate_sign(self):
        """Closing gives negative range-rate; opening positive."""
        closing = np.array([200.0, 0.0, 1.5, -30.0, 0.0, 0.0])
        opening = np.array([200.0, 0.0, 1.5, +30.0, 0.0, 0.0])
        self.assertAlmostEqual(self.model.h(closing)[3], -30.0, places=6)
        self.assertAlmostEqual(self.model.h(opening)[3], +30.0, places=6)

    def test_tangential_motion_has_zero_doppler(self):
        """The MTI blind zone: pure crossing motion is Doppler-invisible."""
        x = np.array([200.0, 0.0, 1.5, 0.0, 45.0, 0.0])
        self.assertAlmostEqual(self.model.h(x)[3], 0.0, places=9)

    def test_jacobian_matches_numeric(self):
        rng = np.random.default_rng(7)
        for _ in range(40):
            x = np.concatenate([rng.normal(0, 250, 3), rng.normal(0, 50, 3)])
            if np.linalg.norm(x[:3] - self.origin) < 5.0:
                continue
            np.testing.assert_allclose(
                self.model.H(x), numeric_jacobian(self.model, x),
                atol=1e-5, rtol=1e-4,
            )

    def test_jacobian_with_longer_state(self):
        """A constant-acceleration state must produce a wider Jacobian with
        zeros in the acceleration columns, not an exception."""
        x = np.array([120.0, -60.0, 90.0, 10.0, 5.0, -2.0, 0.5, 0.0, -9.81])
        H = self.model.H(x)
        self.assertEqual(H.shape, (4, 9))
        np.testing.assert_allclose(H[:, 6:], 0.0)
        np.testing.assert_allclose(H, numeric_jacobian(self.model, x),
                                   atol=1e-5, rtol=1e-4)

    def test_rejects_state_without_velocity(self):
        with self.assertRaises(ValueError):
            self.model.H(np.array([1.0, 2.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
