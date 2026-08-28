"""Fire-control tests: intercept geometry, ballistics, guidance, doctrine."""

import unittest

import numpy as np

from aasys.fire_control.effectors import (Interceptor, Projectile,
                                          kill_probability,
                                          step_projectiles)
from aasys.fire_control.lethality import check_impacts
from aasys.fire_control.engagement import EngagementManager, evaluate_threats
from aasys.fire_control.guidance import (closing_speed, los_rate,
                                         proportional_navigation)
from aasys.fire_control.intercept import (intercept_time_constant_speed,
                                          projectile_time_of_flight,
                                          solve_gun, solve_intercept)
from aasys.tracking.track import Track, TrackState


class TestIntercept(unittest.TestCase):
    def test_head_on_matches_closed_form(self):
        t, p = solve_intercept([0, 0, 0], [1000.0, 0, 0], [-100.0, 0, 0], 300.0)
        self.assertAlmostEqual(t, 1000.0 / 400.0, places=6)
        np.testing.assert_allclose(p, [750.0, 0, 0], atol=1e-6)

    def test_impact_lies_on_interceptor_sphere(self):
        """Whatever the geometry, |impact - launcher| must equal speed*t."""
        speed = 250.0
        t, p = solve_intercept([0, 0, 0], [500.0, 500.0, 60.0],
                               [0.0, -80.0, 0.0], speed)
        self.assertAlmostEqual(float(np.linalg.norm(p)) / t, speed, places=6)

    def test_uncatchable_target_returns_none(self):
        self.assertIsNone(
            intercept_time_constant_speed(np.array([100.0, 0, 0]),
                                          np.array([500.0, 0, 0]), 100.0))
        self.assertEqual(
            solve_intercept([0, 0, 0], [100.0, 0, 0], [500.0, 0, 0], 100.0),
            (None, None))

    def test_diverging_acceleration_is_declined_not_returned(self):
        """The acceleration branch iterates t <- |p(t) - launcher| / speed,
        and p(t) carries a*t^2/2. For a large enough estimated acceleration
        that map diverges, and twelve passes reach magnitudes that overflow
        the launcher selection downstream. A runaway must be reported as
        out-of-envelope, the same as any other uncatchable geometry."""
        t, aim = solve_intercept([0, 0, 0], [800.0, 0, 100.0], [-40.0, 0, 0],
                                 180.0, target_accel=[0.0, 0.0, 4000.0])
        self.assertIsNone(t)
        self.assertIsNone(aim)

    def test_solution_is_always_finite_and_reachable(self):
        """Whatever it returns must be usable: no infinities, no aim points
        outside float range, and a time of flight the interceptor can fly."""
        rng = np.random.default_rng(0)
        for _ in range(300):
            pos = rng.normal(0, 900, 3)
            vel = rng.normal(0, 60, 3)
            acc = rng.normal(0, 400, 3)
            t, aim = solve_intercept(np.zeros(3), pos, vel, 185.0,
                                     target_accel=acc)
            if t is None:
                self.assertIsNone(aim)
                continue
            self.assertTrue(np.isfinite(t) and 0 < t <= 120.0, t)
            self.assertTrue(np.all(np.isfinite(aim)), aim)

    def test_accelerating_target_converges(self):
        t, p = solve_intercept([0, 0, 0], [800.0, 0, 100.0], [-40.0, 0, 0],
                               300.0, target_accel=[0.0, 0.0, -9.81])
        self.assertIsNotNone(t)
        self.assertAlmostEqual(float(np.linalg.norm(p)) / t, 300.0, places=2)


class TestBallistics(unittest.TestCase):
    def test_drag_lengthens_time_of_flight(self):
        for rng_m in (500.0, 1500.0):
            tof = projectile_time_of_flight(rng_m, 1000.0, 900.0)
            self.assertGreater(tof, rng_m / 1000.0,
                               "drag must make the round slower than vacuum")

    def test_time_of_flight_grows_with_range(self):
        a = projectile_time_of_flight(400.0, 1000.0, 900.0)
        b = projectile_time_of_flight(1200.0, 1000.0, 900.0)
        self.assertGreater(b, a)

    def test_gun_solution_is_superelevated(self):
        """The barrel must point above the aim point to offset gravity drop."""
        direction, tof, impact = solve_gun([0, 0, 3.0], [800.0, 0, 3.0],
                                           [0.0, 0, 0], 1000.0)
        self.assertIsNotNone(direction)
        self.assertGreater(direction[2], 0.0)

    def test_gun_leads_a_crossing_target(self):
        _, _, impact = solve_gun([0, 0, 3.0], [600.0, 0, 80.0],
                                 [0.0, 90.0, 0.0], 1000.0)
        self.assertGreater(impact[1], 30.0, "aim point must lead the target")


class TestRoundImpacts(unittest.TestCase):
    """A round crosses metres per physics step; its lethal radius is ~1 m.

    Scoring a hit from the round's sampled position alone means a round that
    flew straight through the target is recorded as a miss unless a sample
    happened to land close. That makes measured gun accuracy a property of
    the integrator's step size, which is precisely the coupling the
    simulation is built to avoid -- the gun is supposed to be the honest
    instrument for judging the tracker.
    """

    class _Target:
        """Minimal stand-in: `check_impacts` only reads these."""

        def __init__(self, position, radius=0.45):
            self.id = 1
            self.name = "probe"
            self.position = np.asarray(position, float)
            self.radius = radius
            self.destroyed = False

        def kill(self, t):
            self.destroyed = True

    def _fire_through(self, dt):
        """Fire a round straight at a target 46 m away and step at `dt`."""
        gun = np.array([0.0, 0.0, 0.0])
        aim = np.array([46.0, 0.0, 0.0])
        tgt = self._Target(aim)
        direction, _, _ = solve_gun(gun, aim, np.zeros(3), 1000.0)
        p = Projectile(gun, direction * 1000.0)
        stats, salvo = {"kills": 0}, {}
        events = []
        for i in range(200):
            step_projectiles([p], i * dt, dt)
            events += check_impacts(i * dt, [p], [tgt], stats, salvo)
            if tgt.destroyed or not p.alive:
                break
        return tgt.destroyed, salvo.get(p.salvo_id, float("inf"))

    def test_hits_a_target_it_flies_through(self):
        """At 120 Hz the round advances 8.3 m per step and straddles the
        target. It must still be scored as the hit it is."""
        hit, miss = self._fire_through(1.0 / 120.0)
        self.assertTrue(hit, f"round flew through the target and missed "
                             f"(closest sampled approach {miss:.2f} m)")

    def test_result_does_not_depend_on_the_step_size(self):
        """The whole point: halving the step must not change the outcome."""
        for dt in (1 / 60.0, 1 / 120.0, 1 / 480.0):
            with self.subTest(dt=dt):
                self.assertTrue(self._fire_through(dt)[0])

    def test_a_genuine_miss_is_still_a_miss(self):
        gun = np.array([0.0, 0.0, 0.0])
        tgt = self._Target([46.0, 8.0, 0.0])
        direction, _, _ = solve_gun(gun, np.array([46.0, 0.0, 0.0]),
                                    np.zeros(3), 1000.0)
        p = Projectile(gun, direction * 1000.0)
        stats, salvo = {"kills": 0}, {}
        for i in range(200):
            step_projectiles([p], i / 120.0, 1 / 120.0)
            check_impacts(i / 120.0, [p], [tgt], stats, salvo)
            if not p.alive:
                break
        self.assertFalse(tgt.destroyed)


class TestProximityFuze(unittest.TestCase):
    """The fuze is polled against every live target in turn.

    Its "range is opening" test -- the part that finds the actual closest
    point of approach rather than triggering on the way in -- compares the
    current range against the previous one. With a single stored range, the
    poll for target B overwrites the range to target A, so on the next step
    A's range is compared against B's.

    The geometry below is chosen to isolate that branch: the near target
    passes at 8 m, outside the 4 m fuze radius, so the plain range test can
    never fire and only the opening test can. That is why the bug survived --
    against a single target, or against anything passing inside the fuze
    radius, the shared range is harmless.
    """

    FUZE_R = 4.0
    MISS = 8.0          # outside FUZE_R, inside the 3x opening window

    def setUp(self):
        self.near = {"id": 1, "pos": np.array([self.MISS, 0.0, 50.0])}
        self.far = {"id": 2, "pos": np.array([400.0, 0.0, 50.0])}

    def _flypast(self, targets, key_by_target=True):
        """Coast an interceptor straight past `targets[0]`, polling the fuze
        against every target each step the way `lethality.check_fuzes` does.
        Returns the recorded miss distance, or None if it never fired."""
        m = Interceptor([0.0, -60.0, 50.0], [0.0, 200.0, 0.0], 1,
                        boost_accel=0.0, boost_time=0.0,
                        fuze_radius=self.FUZE_R)
        dt = 1.0 / 240.0
        for _ in range(400):
            m.position = m.position + m.velocity * dt
            for g in targets:
                if m.check_fuze(g["pos"], key=g["id"] if key_by_target else 0):
                    return m.miss_distance
        return None

    def test_finds_closest_approach_with_one_target(self):
        miss = self._flypast([self.near])
        self.assertIsNotNone(miss, "fuze never fired")
        self.assertAlmostEqual(miss, self.MISS, delta=0.5)

    def test_second_target_does_not_corrupt_the_first(self):
        """The regression: same result whether or not B is also polled."""
        miss = self._flypast([self.near, self.far])
        self.assertIsNotNone(miss, "fuze never fired with a second target")
        self.assertAlmostEqual(miss, self.MISS, delta=0.5)

    def test_sharing_one_range_across_targets_disarms_the_fuze(self):
        """Pins the failure mode, so reverting the per-target keying fails
        here. With one shared range, A's range is always compared against B's
        400 m and never looks like it is opening -- the interceptor flies
        straight past a target it should have killed."""
        self.assertIsNone(self._flypast([self.near, self.far],
                                        key_by_target=False))


class TestGuidance(unittest.TestCase):
    def test_collision_course_has_zero_los_rate(self):
        """Constant bearing with closing range is a collision: PN should
        command essentially nothing."""
        mp, mv = np.array([0.0, 0.0, 0.0]), np.array([200.0, 0.0, 0.0])
        tp, tv = np.array([1000.0, 0.0, 0.0]), np.array([-50.0, 0.0, 0.0])
        np.testing.assert_allclose(los_rate(tp - mp, tv - mv), 0.0, atol=1e-12)
        a = proportional_navigation(mp, mv, tp, tv)
        self.assertLess(float(np.linalg.norm(a)), 1e-9)

    def test_crossing_target_commands_lateral_acceleration(self):
        a = proportional_navigation(np.zeros(3), np.array([200.0, 0.0, 0.0]),
                                    np.array([800.0, 0.0, 0.0]),
                                    np.array([0.0, 60.0, 0.0]))
        self.assertGreater(float(np.linalg.norm(a)), 1.0)

    def test_command_respects_acceleration_limit(self):
        a = proportional_navigation(np.zeros(3), np.array([300.0, 0.0, 0.0]),
                                    np.array([200.0, 5.0, 0.0]),
                                    np.array([0.0, 400.0, 0.0]),
                                    max_accel=150.0)
        self.assertLessEqual(float(np.linalg.norm(a)), 150.0 + 1e-6)

    def test_closing_speed_sign(self):
        self.assertGreater(closing_speed(np.array([100.0, 0, 0]),
                                         np.array([-30.0, 0, 0])), 0)
        self.assertLess(closing_speed(np.array([100.0, 0, 0]),
                                      np.array([30.0, 0, 0])), 0)


class TestDoctrine(unittest.TestCase):
    def test_inbound_outranks_receding(self):
        class T:
            def __init__(self, i, p, v):
                self.id = i
                self.position = np.array(p, float)
                self.velocity = np.array(v, float)

        inbound = T(1, [300.0, 0, 60.0], [-40.0, 0, 0])
        leaving = T(2, [200.0, 0, 60.0], [40.0, 0, 0])
        ranked = evaluate_threats([leaving, inbound], [0, 0, 0])
        self.assertEqual(ranked[0].track_id, 1)
        self.assertTrue(ranked[0].inbound)
        self.assertFalse(ranked[1].inbound)

    def test_closest_approach_beats_raw_range(self):
        """A near crosser that misses is less urgent than a farther inbound."""
        class T:
            def __init__(self, i, p, v):
                self.id = i
                self.position = np.array(p, float)
                self.velocity = np.array(v, float)

        crosser = T(1, [120.0, 0, 60.0], [0.0, 60.0, 0.0])
        inbound = T(2, [420.0, 0, 40.0], [-45.0, 0, -4.0])
        ranked = evaluate_threats([crosser, inbound], [0, 0, 0], asset_radius=40.0)
        self.assertEqual(ranked[0].track_id, 2)

    def test_kill_probability_decays_with_miss(self):
        self.assertAlmostEqual(kill_probability(0.0, 5.0), 1.0)
        self.assertGreater(kill_probability(2.0, 5.0), kill_probability(6.0, 5.0))
        self.assertLess(kill_probability(20.0, 5.0), 0.01)

    def test_seeker_field_of_view(self):
        m = Interceptor([0, 0, 0], [200.0, 0, 0], 1, seeker_range=400.0,
                        seeker_fov_deg=30.0)
        self.assertTrue(m.seeker_sees(np.array([300.0, 0.0, 0.0])))
        self.assertFalse(m.seeker_sees(np.array([0.0, 300.0, 0.0])),
                         "target outside the seeker cone must not be seen")
        self.assertFalse(m.seeker_sees(np.array([900.0, 0.0, 0.0])),
                         "target beyond seeker range must not be seen")


def _confirmed_track(pos, vel, sigma=0.5):
    """A confirmed track with an age past the fire-control minimum."""

    class Filt:
        def __init__(self):
            self.x = np.array(list(pos) + list(vel) + [0.0, 0.0, 0.0])
            self.P = np.diag([sigma ** 2] * 3 + [5.0] * 6)

        @property
        def position(self):
            return self.x[:3]

        @property
        def velocity(self):
            return self.x[3:6]

    trk = Track(Filt(), 0.0, "radar")
    trk.t_updated = 5.0
    trk.state = TrackState.CONFIRMED
    return trk


class TestCounterDrone(unittest.TestCase):
    def _manager(self, **over):
        base = dict(asset_pos=[0.0, 0.0, 0.0], uav_hangar=2,
                    missile_sets=1, max_simultaneous=0,
                    rng=np.random.default_rng(0))
        base.update(over)
        return EngagementManager(**base)

    def test_uav_launches_within_envelope(self):
        eng = self._manager()
        trk = _confirmed_track([800.0, 0.0, 60.0], [-40.0, 0.0, 0.0])
        eng.decide(5.0, [trk])
        self.assertEqual(eng.stats["uavs"], 1)
        self.assertEqual(eng.stats["missiles"], 0,
                         "missiles off via max_simultaneous=0")
        self.assertEqual(len(eng.uavs), 1)
        self.assertTrue(eng.assigned.get(("uav", trk.id)))
        self.assertEqual(eng.records[-1].weapon, "uav")

    def test_uav_skips_target_already_under_missile(self):
        eng = self._manager(max_simultaneous=3)
        trk = _confirmed_track([800.0, 0.0, 60.0], [-40.0, 0.0, 0.0])
        eng.decide(5.0, [trk])
        self.assertEqual(eng.stats["missiles"], 1)
        self.assertEqual(eng.stats["uavs"], 0,
                         "a slow chaser must not stack on a missile's target")

    def test_uav_not_in_missile_mode(self):
        from aasys.fire_control.mounts import FireMode
        eng = self._manager()
        eng.fire_mode = FireMode.MISSILE
        trk = _confirmed_track([800.0, 0.0, 60.0], [-40.0, 0.0, 0.0])
        eng.decide(5.0, [trk])
        self.assertEqual(eng.stats["uavs"], 0)

    def test_uav_range_gate(self):
        eng = self._manager()
        trk = _confirmed_track([2200.0, 0.0, 60.0], [-40.0, 0.0, 0.0])
        eng.decide(5.0, [trk])
        self.assertEqual(eng.stats["uavs"], 0)

    def test_uav_sigma_gate(self):
        eng = self._manager()
        trk = _confirmed_track([800.0, 0.0, 60.0], [-40.0, 0.0, 0.0],
                               sigma=10.0)
        eng.decide(5.0, [trk])
        self.assertEqual(eng.stats["uavs"], 0)

    def test_airborne_cap_uses_hangar_pool(self):
        eng = self._manager(uav_hangar=2)
        tracks = [_confirmed_track([800.0, 0.0, 60.0], [-40.0, 0.0, 0.0]),
                  _confirmed_track([700.0, 0.0, 60.0], [-35.0, 0.0, 0.0]),
                  _confirmed_track([900.0, 0.0, 60.0], [-38.0, 0.0, 0.0])]
        eng.decide(5.0, tracks)
        self.assertEqual(eng.stats["uavs"], 2,
                         "hangar of 2 caps the number of chasers launched")


if __name__ == "__main__":
    unittest.main()
