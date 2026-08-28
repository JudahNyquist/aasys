"""Regression gate for tracker quality.

`bench_carve` guards carving throughput and `unittest` guards component
behaviour, but nothing guarded the thing the project is actually judged on:
how well the estimator tracks. Tuning changes -- process noise, gate
probabilities, revisit policy -- move those numbers without breaking a
single unit test, so a quiet degradation could ship unnoticed.

Each bound below was measured, not chosen, and is deliberately loose enough
to absorb seed-to-seed variation while still catching a real regression.
Tightening one because a change happened to improve it is fine; loosening
one to make a failing change pass defeats the point of having it.

A note on the NEES bounds, because two of them look wrong. NEES should
average the state dimension (6) for a filter whose covariance is honest.
`maneuvering` does. `orbiter` and `hovering` sit far below it, because the
process noise is sized for a drone that can break hard at any moment and
those two targets never do -- so the filter carries far more uncertainty
than it turns out to need. That is a deliberate safety margin rather than a
defect, but it is real, and pinning it here means a future retune has to be
an explicit decision instead of an accident.

Run:  python -m tools.regress
"""

from __future__ import annotations

import argparse
import sys

import numpy as np

from .batch_eval import run_trial

TRIALS = 4
DURATION = 25.0

#: scenario -> (rmse_max, nees_lo, nees_hi, held_min_pct)
BOUNDS = {
    "maneuvering": (1.60, 3.0, 12.0, 78.0),
    "orbiter": (0.80, 0.05, 12.0, 95.0),
    "hovering": (0.80, 0.50, 12.0, 95.0),
}


def check(scenario: str, trials: int, duration: float) -> list[str]:
    rmse_max, nees_lo, nees_hi, held_min = BOUNDS[scenario]
    rs = [run_trial(scenario, seed, duration) for seed in range(trials)]

    def mean(k):
        v = np.array([r[k] for r in rs], dtype=float)
        v = v[np.isfinite(v)]
        return float(np.mean(v)) if v.size else float("nan")

    rmse, nees, held = mean("rmse"), mean("nees"), mean("coverage") * 100.0
    fails = []
    if not (rmse <= rmse_max):
        fails.append(f"RMSE {rmse:.2f} m > {rmse_max:.2f} m")
    if not (nees_lo <= nees <= nees_hi):
        fails.append(f"NEES {nees:.2f} outside [{nees_lo}, {nees_hi}]")
    if not (held >= held_min):
        fails.append(f"held {held:.1f}% < {held_min:.1f}%")

    mark = "FAIL" if fails else "ok"
    print(f"  {scenario:<14s} RMSE={rmse:6.2f} m (<={rmse_max:.2f})   "
          f"NEES={nees:6.2f} ([{nees_lo}, {nees_hi}])   "
          f"held={held:5.1f}% (>={held_min:.0f})   {mark}")
    for f in fails:
        print(f"      {f}")
    return fails


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=TRIALS)
    ap.add_argument("--duration", type=float, default=DURATION)
    ap.add_argument("--scenario", choices=sorted(BOUNDS),
                    help="check one scenario instead of all")
    args = ap.parse_args()

    print("=" * 78)
    print(f"TRACKER REGRESSION GATE - {args.trials} seeds x "
          f"{args.duration:.0f} s per scenario")
    print("=" * 78)

    scenarios = [args.scenario] if args.scenario else sorted(BOUNDS)
    failed = [s for s in scenarios
              if check(s, args.trials, args.duration)]

    print()
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        print("A bound moving the wrong way is a result, not an obstacle --")
        print("find out why before changing the number.")
        return 1
    print("all bounds held")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
