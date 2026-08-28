"""Enforcement for the project's central invariant.

`lethality.py` states that nothing downstream of the sensors is allowed to
read ground truth, and that this module is what enforces it. Without an
actual check that is an honour system, and an honour system is exactly the
wrong guarantee for the one property the whole simulation exists to
demonstrate: a tracker that quietly peeks at truth produces beautiful
results that mean nothing.

The check is a runtime guard rather than a static one, because the thing
worth detecting is not a syntactic pattern -- it is a *read*, of a real
`Target`, from a module that is not entitled to it. Static analysis would
have to infer which names hold targets and would be wrong in both
directions. Swapping in a guarding `__getattribute__` for the duration of a
run answers the question exactly, at a cost that only a test ever pays.

Two kinds of truth access are legitimate, and they are separated by file so
that this allowlist can be short and specific:

* **Generating measurements.** Sensors must observe the real world; that is
  what makes them sensors. The noise, dropout and geometry they apply on the
  way out is the entire model.
* **Resolving outcomes.** Once a round is in the air, whether it passed
  within lethal radius of the real target is physics and no amount of
  filtering changes it.

Everything else -- association, filtering, threat evaluation, intercept
solutions, guidance -- must run on estimates alone.

Usage:

    with truth_guard.enforce():
        sim.run(30.0)

"""

from __future__ import annotations

import sys
from contextlib import contextmanager

from ..entities.target import Target

#: Attributes of a `Target` that reveal its true state.
TRUTH_ATTRS = frozenset({
    "position", "velocity", "accel", "speed", "state",
    "alive", "destroyed", "t_killed",
    "rcs", "radius", "mass", "area", "drag_coeff", "profile",
})

#: Modules permitted to read them, and why.
ALLOWED_MODULES = {
    # The target's own dynamics.
    "aasys.entities.target",
    # Measurement generation: sensors observe the world by definition.
    "aasys.sensing.radar",
    "aasys.sensing.optical",
    "aasys.sensing.silhouette",
    # An onboard seeker is a sensor too -- it observes truth and steers on
    # its own noisy measurement, never on the truth it observed.
    "aasys.fire_control.seeker",
    # Outcome resolution, after the shot is beyond recall.
    "aasys.fire_control.lethality",
    # The orchestrator advances the targets and scores against them. Scoring
    # is a report *about* the run, not an input to it.
    "aasys.sim",
}

#: Prefixes permitted wholesale -- the viewer draws the true world, and the
#: analysis tools and tests exist to score against it.
ALLOWED_PREFIXES = ("aasys.render.", "tools.", "tests.", "aasys.analysis.")


class TruthLeak(AssertionError):
    """A truth-blind module read a target's real state."""


def _permitted(module: str) -> bool:
    return (module in ALLOWED_MODULES
            or module.startswith(ALLOWED_PREFIXES)
            or module in ("__main__", "contextlib", "unittest.case"))


def _describe(frame) -> str:
    return (f"{frame.f_globals.get('__name__', '?')}"
            f":{frame.f_lineno} in {frame.f_code.co_name}()")


@contextmanager
def enforce(extra_allowed: frozenset[str] | set[str] = frozenset()):
    """Raise `TruthLeak` if a truth-blind module reads a `Target`'s state.

    Restores the original attribute access on exit, including on failure, so
    an enforced block cannot leave the class instrumented for later tests.
    """
    allowed = set(ALLOWED_MODULES) | set(extra_allowed)
    original = Target.__getattribute__

    def guarded(self, name):
        if name in TRUTH_ATTRS:
            caller = sys._getframe(1)
            module = caller.f_globals.get("__name__", "")
            if not (module in allowed or _permitted(module)):
                raise TruthLeak(
                    f"{_describe(caller)} read Target.{name}. Only sensors "
                    f"(measurement generation) and fire_control.lethality "
                    f"(outcome resolution) may see ground truth; everything "
                    f"downstream must run on track estimates.")
        return original(self, name)

    Target.__getattribute__ = guarded
    try:
        yield
    finally:
        Target.__getattribute__ = original
