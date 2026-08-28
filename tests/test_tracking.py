"""Tracking tests: motion models, filters, fusion, association, lifecycle."""

import unittest

import numpy as np

from aasys.sensing.base import Measurement
from aasys.sensing.models import (CartesianPositionModel, SphericalDopplerModel)
from aasys.tracking.association import associate
from aasys.tracking.factory import cv_ekf_factory, imm_factory
from aasys.tracking.filters import EKF, IMM
from aasys.tracking.manager import TrackManager
from aasys.tracking.models import (DIM, F_ca, F_ct9, F_cv, F_cv9, Q_ca, Q_cv,
                                   Q_cv9, _block, ca_model, ct_model, cv_model,
                                   cv_model_6)


def cart(t, p, sigma=0.5):
    return Measurement(t=t, z=np.asarray(p, float),
                       R=np.eye(3) * sigma ** 2,
                       model=CartesianPositionModel(), sensor_id="optical",
                       meta={"position": np.asarray(p, float)})


class TestMotionModels(unittest.TestCase):
    def test_Q_composes_over_dt(self):
        """Irregular sensor intervals mean Q is rebuilt every update; one
        step of 2dt must equal two steps of dt or the filter is inconsistent
        exactly when measurements are sparse."""
        dt, q = 0.37, 3.0
        for F, Q in ((F_cv, Q_cv), (F_ca, Q_ca), (F_cv9, Q_cv9)):
            F1, Q1 = F(dt), Q(dt, q)
            np.testing.assert_allclose(F1 @ Q1 @ F1.T + Q1, Q(2 * dt, q),
                                       atol=1e-9)

    def test_Q_positive_semidefinite(self):
        for Q in (Q_cv(0.3, 4.0), Q_ca(0.3, 4.0), Q_cv9(0.3, 4.0)):
            self.assertTrue(np.all(np.linalg.eigvalsh(Q) >= -1e-10))

    def test_turn_model_preserves_speed(self):
        x = np.zeros(9)
        x[3:6] = [0.0, 30.0, 0.0]
        out = F_ct9(1.0, np.radians(20.0)) @ x
        self.assertAlmostEqual(np.linalg.norm(out[3:5]), 30.0, places=9)

    def test_turn_reduces_to_cv_at_zero_rate(self):
        np.testing.assert_allclose(F_ct9(0.4, 1e-13), F_cv9(0.4), atol=1e-9)

    def test_block_matches_kron(self):
        """`_block` is a hand-rolled Kronecker product, taken because the
        real one dominated the profile. It has to stay exactly equal to the
        thing it replaced, not merely close -- a drift here would bias every
        prediction in the bank."""
        rng = np.random.default_rng(0)
        for k in (2, 3, 4):
            for _ in range(5):
                m = rng.normal(size=(k, k))
                np.testing.assert_array_equal(_block(m),
                                              np.kron(m, np.eye(DIM)))

    def test_imm_models_share_dimension(self):
        dims = {m.dim for m in (cv_model(), ca_model(), ct_model(10.0))}
        self.assertEqual(len(dims), 1)


class TestEKF(unittest.TestCase):
    def test_converges_on_noiseless_constant_velocity(self):
        f = EKF(np.zeros(6), np.eye(6) * 100.0, cv_model_6(q=1e-8))
        p0, v = np.array([10.0, 5.0, 60.0]), np.array([12.0, -3.0, 1.0])
        for k in range(150):
            f.predict(0.1)
            f.update(cart((k + 1) * 0.1, p0 + v * (k + 1) * 0.1, 0.1))
        np.testing.assert_allclose(f.velocity, v, atol=1e-3)

    def test_covariance_stays_symmetric_psd(self):
        """Joseph form must survive updates of wildly different accuracy,
        which is the normal case when radar and optical interleave."""
        f = EKF(np.zeros(6), np.eye(6) * 50.0, cv_model_6(q=5.0))
        rng = np.random.default_rng(0)
        for k in range(200):
            f.predict(0.05)
            sigma = 20.0 if k % 2 else 0.3
            f.update(cart(k * 0.05, rng.normal(0, sigma, 3), sigma))
            self.assertTrue(np.allclose(f.P, f.P.T))
            self.assertTrue(np.all(np.linalg.eigvalsh(f.P) > -1e-8))

    def test_sequential_update_matches_batch(self):
        """Fusing two independent sensors one at a time must equal stacking
        them into a single update."""
        from scipy.linalg import block_diag
        x0, P0 = np.array([10.0, 2.0, 50.0, 1.0, 0.0, 0.0]), np.eye(6) * 9.0
        m1 = cart(0.0, [10.4, 2.2, 50.3], 0.5)
        m2 = cart(0.0, [9.7, 1.8, 49.6], 1.5)

        seq = EKF(x0.copy(), P0.copy(), cv_model_6())
        seq.update(m1)
        seq.update(m2)

        H = np.zeros((6, 6))
        H[:3, :3] = np.eye(3)
        H[3:6, :3] = np.eye(3)
        z = np.concatenate([m1.z, m2.z])
        R = block_diag(m1.R, m2.R)
        S = H @ P0 @ H.T + R
        K = P0 @ H.T @ np.linalg.inv(S)
        x_batch = x0 + K @ (z - H @ x0)
        np.testing.assert_allclose(seq.x, x_batch, atol=1e-8)


class TestIMM(unittest.TestCase):
    def test_transition_is_time_consistent(self):
        """Regression: a fixed per-step transition applied at the physics
        rate walks model probabilities to uniform between measurements and
        erases all evidence. Rate-based transitions must compose."""
        f = IMM([cv_model(), ca_model(), ct_model(-20.0), ct_model(20.0)],
                np.zeros(9), np.eye(9), dwell_time_s=6.0)
        A = f._transition_for(0.5)
        np.testing.assert_allclose(A, f._transition_for(0.25) @ f._transition_for(0.25))
        np.testing.assert_allclose(
            A, np.linalg.matrix_power(f._transition_for(0.5 / 64), 64), atol=1e-9)
        np.testing.assert_allclose(A.sum(axis=1), 1.0)

    def test_identifies_a_turn(self):
        """A coordinated turn must raise the matching turn model above CV."""
        make = imm_factory()
        omega = np.radians(20.0)
        r, speed = 60.0, 20.0
        f = make(np.array([r, 0.0, 40.0]), np.eye(3) * 0.25)
        f.x[3:6] = [0.0, speed, 0.0]
        rng = np.random.default_rng(1)
        dt = 0.05
        for k in range(400):
            th = omega * k * dt
            p = np.array([r * np.cos(th), r * np.sin(th), 40.0])
            f.predict(dt)
            f.update(cart(k * dt, p + rng.normal(0, 0.2, 3), 0.2))
        probs = f.model_probabilities()
        self.assertEqual(f.dominant_model()[:2], "CT")
        self.assertGreater(probs[f.dominant_model()], probs["CV"])


class TestAssociation(unittest.TestCase):
    def test_recovers_known_assignment(self):
        """Greedy nearest-neighbour mis-assigns crossing tracks; the global
        solution must not."""
        make = cv_ekf_factory(q=1.0)

        class T:
            def __init__(self, p):
                self.filter = make(np.array(p, float), np.eye(3) * 0.25)

        tracks = [T([0.0, 0.0, 50.0]), T([10.0, 0.0, 50.0])]
        ms = [cart(0.0, [10.2, 0.1, 50.0]), cart(0.0, [0.1, -0.1, 50.0])]
        matches, ut, um = associate(tracks, ms)
        self.assertEqual(sorted(matches), [(0, 1), (1, 0)])
        self.assertEqual((ut, um), ([], []))

    def test_out_of_gate_measurement_is_left_unmatched(self):
        make = cv_ekf_factory(q=1.0)

        class T:
            def __init__(self, p):
                self.filter = make(np.array(p, float), np.eye(3) * 0.25)

        tracks = [T([0.0, 0.0, 50.0])]
        matches, ut, um = associate(tracks, [cart(0.0, [900.0, 900.0, 50.0])])
        self.assertEqual(matches, [])
        self.assertEqual(um, [0])


class TestTrackLifecycle(unittest.TestCase):
    def test_confirms_then_deletes_when_evidence_stops(self):
        mgr = TrackManager(cv_ekf_factory(q=2.0), confirm_hits=3,
                           tentative_timeout_s=1.0, max_coast_s=1.0)
        p = np.array([0.0, 0.0, 50.0])
        v = np.array([10.0, 0.0, 0.0])
        t = 0.0
        for k in range(8):
            t = k * 0.1
            mgr.step(t, [cart(t, p + v * t)])
        self.assertEqual(len(mgr.confirmed), 1)
        for k in range(40):
            t += 0.1
            mgr.step(t, [])
        self.assertEqual(len(mgr.tracks), 0)

    def test_no_double_prediction(self):
        """Regression: predicting from a stale timestamp advanced the state
        twice per frame, biasing position forward at speed."""
        mgr = TrackManager(cv_ekf_factory(q=1e-6), confirm_hits=2)
        p0, v = np.array([0.0, 0.0, 50.0]), np.array([50.0, 0.0, 0.0])
        for k in range(40):
            t = k * 0.05
            mgr.step(t, [cart(t, p0 + v * t, 0.05)])
        trk = mgr.tracks[0]
        truth = p0 + v * (39 * 0.05)
        self.assertLess(float(np.linalg.norm(trk.position - truth)), 1.0)

    def test_duplicate_measurement_does_not_spawn_second_track(self):
        """Two dwells seeing one target in the same frame is routine and
        must not create a duplicate track."""
        mgr = TrackManager(cv_ekf_factory(q=2.0), confirm_hits=2)
        p = np.array([0.0, 0.0, 50.0])
        for k in range(10):
            t = k * 0.1
            here = p + np.array([10.0, 0.0, 0.0]) * t
            mgr.step(t, [cart(t, here), cart(t, here + 0.05)])
        self.assertEqual(len(mgr.tracks), 1)


if __name__ == "__main__":
    unittest.main()
