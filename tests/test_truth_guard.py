"""The invariant the whole simulation rests on.

`README` opens with "nothing downstream ever sees ground truth" and
`lethality.py` names `aasys.core.truth_guard` as what enforces it. These
tests are that enforcement. They are worth more than any single tracking
test: a filter that peeks at truth passes every accuracy check ever written
while proving nothing at all.
"""

import unittest

import numpy as np

from aasys.core import truth_guard
from aasys.core.truth_guard import TruthLeak
from aasys.entities.target import Target
from aasys.scenarios import build
from aasys.sim import Simulation


def _leak(target):
    """Stand-in for a truth-blind module reading a target. Defined in this
    test module, which is on the allowlist, so it has to be run through a
    guard that removes that exemption to be a meaningful probe."""
    return target.position


class TestGuardItself(unittest.TestCase):
    """A guard that never fires is indistinguishable from no guard."""

    def test_detects_a_read_from_a_disallowed_module(self):
        tgt = Target([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        original = truth_guard._permitted
        try:
            # Pretend this test module is not entitled to truth.
            truth_guard._permitted = lambda m: False
            with truth_guard.enforce():
                with self.assertRaises(TruthLeak):
                    _leak(tgt)
        finally:
            truth_guard._permitted = original

    def test_names_the_attribute_and_the_reader(self):
        tgt = Target([1.0, 2.0, 3.0], [4.0, 0.0, 0.0])
        original = truth_guard._permitted
        try:
            truth_guard._permitted = lambda m: False
            with truth_guard.enforce():
                with self.assertRaises(TruthLeak) as ctx:
                    _leak(tgt)
        finally:
            truth_guard._permitted = original
        msg = str(ctx.exception)
        self.assertIn("Target.position", msg)
        self.assertIn("_leak", msg)

    def test_allows_the_sensors_and_lethality(self):
        for module in ("aasys.sensing.radar", "aasys.sensing.optical",
                       "aasys.fire_control.lethality",
                       "aasys.fire_control.seeker"):
            self.assertTrue(truth_guard._permitted(module), module)

    def test_denies_the_estimator_and_the_shooters(self):
        for module in ("aasys.tracking.manager", "aasys.tracking.filters.imm",
                       "aasys.fire_control.engagement",
                       "aasys.fire_control.intercept",
                       "aasys.fire_control.guidance",
                       "aasys.fire_control.effectors"):
            self.assertFalse(truth_guard._permitted(module), module)

    def test_restores_attribute_access_after_a_failure(self):
        tgt = Target([1.0, 2.0, 3.0], [0.0, 0.0, 0.0])
        before = Target.__getattribute__
        try:
            with truth_guard.enforce():
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertIs(Target.__getattribute__, before)
        np.testing.assert_array_equal(tgt.position, [1.0, 2.0, 3.0])


class TestScenariosStayBlind(unittest.TestCase):
    """The real assertion: a full engagement, with shooting, leaks nothing."""

    def _run(self, scenario, duration=6.0):
        sim = Simulation(build(scenario), seed=0)
        with truth_guard.enforce():
            sim.run(duration)
        return sim

    def test_swarm_engagement_reads_no_truth(self):
        sim = self._run("swarm")
        self.assertGreater(sim.engagement.stats["rounds"], 0,
                           "scenario must actually shoot for this to prove "
                           "anything about the fire-control path")

    def test_hovering_optical_only_track_reads_no_truth(self):
        self._run("hovering")

    def test_counterdrone_seeker_path_reads_no_truth(self):
        sim = self._run("counterdrone", duration=10.0)
        self.assertGreater(sim.engagement.stats["uavs"], 0,
                           "the seeker path must be exercised")

    def test_standoff_missile_path_reads_no_truth(self):
        # Standoff targets open at ~3 km and the first launch lands at
        # t=16.4 s, so a shorter run would assert over a battery that never
        # fired -- which is exactly the vacuous pass this suite exists to
        # avoid.
        sim = self._run("standoff", duration=18.0)
        self.assertGreater(sim.engagement.stats["missiles"], 0)
