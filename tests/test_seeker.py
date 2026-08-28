"""Seeker tests: lock geometry, the three loss mechanisms, and guidance use.

The seeker is a sensor -- it reads truth only inside its measurement channel
and hands guidance a noisy observation. These tests pin down what "sees" and
"dropped" mean and that a dropped seeker genuinely degrades the aim.
"""

import unittest

import numpy as np

from aasys.fire_control.effectors import Interceptor
from aasys.fire_control.seeker import SeekState, Seeker


class FakeTarget:
    def __init__(self, pos, destroyed=False):
        self.position = np.asarray(pos, dtype=float)
        self.destroyed = destroyed


def _angle(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.arccos(np.clip(a @ b / (np.linalg.norm(a)
                                            * np.linalg.norm(b)), -1.0, 1.0)))


class TestSeekerLock(unittest.TestCase):
    def setUp(self):
        self.s = Seeker(np.random.default_rng(0), range_m=1500.0, fov_deg=70.0,
                        p_det_near=1.0, p_det_far=1.0, slew_rate_deg_s=1e6)

    def test_lock_inside_cone_and_range(self):
        t = FakeTarget([800.0, 0.0, 40.0])
        out, ev = self.s.observe(0.0, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                                 [t], t.position)
        self.assertTrue(out.locked)
        self.assertIn("SEEKER LOCK", ev)
        out, ev = self.s.observe(1.0 / 120, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                                 [t], t.position)
        self.assertTrue(out.locked)
        self.assertIsNone(ev, "a steady lock logs nothing")

    def test_no_lock_outside_cone(self):
        t = FakeTarget([0.0, 800.0, 0.0])          # perpendicular to the rail
        out, _ = self.s.observe(0.0, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                                [t], t.position)
        self.assertIs(out.state, SeekState.SEEKING)

    def test_no_lock_out_of_range(self):
        t = FakeTarget([2600.0, 0.0, 0.0])
        out, _ = self.s.observe(0.0, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                                [t], t.position)
        self.assertIs(out.state, SeekState.SEEKING)


class TestSeekerLoss(unittest.TestCase):
    def test_glare_drop_then_relock(self):
        s = Seeker(np.random.default_rng(2), range_m=1500.0, fov_deg=70.0,
                   p_det_near=1.0, p_det_far=1.0, slew_rate_deg_s=1e6,
                   glare_strike_limit=8)
        t = FakeTarget([600.0, 0.0, 30.0])
        out, _ = s.observe(0.0, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                           [t], t.position)
        self.assertTrue(out.locked)
        # A tracking loop rides through a frame or two of bad contrast...
        s.p_det_near = s.p_det_far = 0.0          # glare now always fails
        for _ in range(4):
            out, ev = s.observe(1.0 / 120, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                                [t], t.position)
            self.assertTrue(out.locked)
            self.assertIsNone(ev)
        # ...but a sustained bank of smoke knocks it out.
        while out.locked:
            out, ev = s.observe(1.0 / 120, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                                [t], t.position)
        self.assertIn("SEEKER LOSS  glare", ev)
        s.p_det_near = s.p_det_far = 1.0
        out, ev = s.observe(1.0 / 120, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                            [t], t.position)
        self.assertTrue(out.locked)
        self.assertIn("SEEKER LOCK", ev)

    def test_slew_drop_on_fast_crossing(self):
        s = Seeker(np.random.default_rng(3), range_m=1500.0, fov_deg=70.0,
                   p_det_near=1.0, p_det_far=1.0, slew_rate_deg_s=55.0)
        t = FakeTarget([80.0, 0.0, 10.0])
        out, _ = s.observe(0.0, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                           [t], t.position)
        self.assertTrue(out.locked)
        t.position[1] += 180.0 / 120.0            # now crossing hard sideways
        out, ev = s.observe(1.0 / 120, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                            [t], t.position)
        self.assertIs(out.state, SeekState.DROPPED)
        self.assertIn("SEEKER LOSS  slew", ev)

    def test_off_boresight_drop(self):
        s = Seeker(np.random.default_rng(4), range_m=1500.0, fov_deg=70.0,
                   p_det_near=1.0, p_det_far=1.0, slew_rate_deg_s=1e6)
        t = FakeTarget([600.0, 0.0, 30.0])
        out, _ = s.observe(0.0, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                           [t], t.position)
        self.assertTrue(out.locked)
        t.position = np.array([0.0, 500.0, 20.0])  # slips out of the cone
        out, ev = s.observe(1.0 / 120, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                            [t], t.position)
        self.assertIs(out.state, SeekState.DROPPED)
        self.assertIn("SEEKER LOSS  off-boresight", ev)


class TestSeekerMeasurement(unittest.TestCase):
    def test_measurement_stays_inside_noise_band(self):
        s = Seeker(np.random.default_rng(5), range_m=1500.0, fov_deg=90.0,
                   p_det_near=1.0, p_det_far=1.0, slew_rate_deg_s=1e6,
                   ang_noise_mrad=1.0, range_noise_frac=0.04)
        t = FakeTarget([900.0, 0.0, 30.0])
        out, _ = s.observe(0.0, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                           [t], t.position)
        self.assertTrue(out.locked)
        true_u = np.array([900.0, 0.0, 30.0])
        true_u /= np.linalg.norm(true_u)
        self.assertLess(_angle(out.u_meas, true_u), 5 * out.sigma_ang)
        self.assertLess(abs(out.r_meas - 900.0) / 900.0, 5 * s.sigma_r_frac)


class TestSeekerGuidedFlight(unittest.TestCase):
    def test_locked_seeker_beats_biased_ground_track(self):
        # The guidance aim point is deliberately offset from the truth-line;
        # the seeker-armed chaser should still converge onto the true line of
        # sight while the seeker-less one flies the biased handoff.
        dt = 1.0 / 120.0
        bias = np.array([4.0, 25.0, 0.0])

        def fly(seeker, steps):
            truth = FakeTarget([900.0, 0.0, 30.0])
            m = Interceptor([0, 0, 0], [45.0, 0.0, 0.0], 77, uav=True,
                            seeker=seeker, boost_accel=30.0, boost_time=1.0,
                            max_lateral=140.0, max_time=60.0)
            t = 0.0
            for _ in range(steps):
                truth.position = truth.position + np.array([-30.0 * dt, 0, 0])
                aim = truth.position + bias
                vel = np.array([-30.0, 0.0, 0.0])
                if m.seeker is not None:
                    out, _ = m.seeker.observe(t, dt, m.position, m.velocity,
                                              [truth], truth.position)
                    m.guide(t, dt, aim, vel, seeker_out=out)
                else:
                    m.guide(t, dt, aim, vel)
                t += dt
            return m, truth

        with_lock, kept = fly(Seeker(np.random.default_rng(7), range_m=2000.0,
                                     fov_deg=70.0, p_det_near=1.0,
                                     p_det_far=1.0, slew_rate_deg_s=1e6), 600)
        no_lock, kept = fly(None, 600)

        def aim_err(m):
            rel = kept.position - m.position
            return _angle(m.velocity, rel)

        self.assertLess(aim_err(with_lock), aim_err(no_lock),
                        "seeker correction should hold the true LOS better")

    def test_deterministic_stream(self):
        runs = []
        for _ in range(2):
            s = Seeker(np.random.default_rng(9), range_m=1500.0, fov_deg=70.0,
                       p_det_near=0.9, p_det_far=0.4, slew_rate_deg_s=30.0)
            t = FakeTarget([600.0, 0.0, 30.0])
            events = []
            tt = 0.0
            for i in range(400):
                out, ev = s.observe(tt, 1.0 / 120, [0, 0, 0], [60, 0, 0],
                                    [t], t.position)
                if ev:
                    events.append((i, ev, out.state.value))
                tt += 1.0 / 120
            runs.append(events)
        self.assertEqual(runs[0], runs[1],
                         "the seeker stream must be reproducible from a seed")


if __name__ == "__main__":
    unittest.main()