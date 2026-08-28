"""Statistical validation across repeated runs.

A single pretty trajectory proves nothing. These are the measurements that
actually say whether the estimator is sound:

* **RMSE** -- how far off it is.
* **NEES** -- normalised estimation error squared, `(x-x̂)ᵀ P⁻¹ (x-x̂)`.
  Should average the state dimension. Well above means the filter is
  overconfident and its covariance is lying; well below means it is
  needlessly conservative. This is the number that catches a filter whose
  track *looks* smooth while its uncertainty is nonsense -- and the only
  honest way to calibrate the optical `r_scale`.
* **Track continuity** -- how much of the flight was actually held, and how
  many spurious tracks were spawned alongside.

Also runs the sensors in isolation, which is the only way to see what fusion
buys rather than assuming it buys something.

One trap is worth naming, because the obvious reading of this table is
wrong. Comparing a fused RMSE against an optical-only RMSE compares two
different populations: optical only sees a target inside its bubble, so its
error is averaged over the small, close, well-conditioned fraction of the
flight it can see at all, while the fused number is averaged over the whole
engagement including everything only radar can reach. Fused therefore looks
*worse* than optical while being strictly better informed. The conditioned
rows below split the fused run by which sensor last updated each track,
which is the comparison that actually means something.

Run:  .venv/bin/python -m tools.batch_eval --trials 8
"""

from __future__ import annotations

import argparse

import numpy as np

from aasys.scenarios import build
from aasys.sim import Simulation


def run_trial(scenario: str, seed: int, duration: float,
              radar: bool = True, optical: bool = True) -> dict:
    cfg = build(scenario)
    cfg["radar"] = radar
    cfg["optical"] = optical
    cfg["engage"] = False          # scoring the tracker, not the shooting
    sim = Simulation(cfg, seed=seed)

    err2, nees, nis = [], [], []
    by_sensor: dict[str, list] = {"radar": [], "optical": []}
    frames_with_track = 0
    n_frames = 0

    for _ in range(int(duration / sim.dt)):
        sim.step()
        n_frames += 1
        conf = sim.tracker.confirmed
        if conf:
            frames_with_track += 1
        for trk in conf:
            got = sim.truth_for_track(trk)
            if got is None:
                continue
            tgt, dist = got
            if dist > 25.0:
                continue           # not this target; do not score it
            e = np.concatenate([tgt.position - trk.position[:3],
                                tgt.velocity - trk.velocity[:3]])
            sq = float(e[:3] @ e[:3])
            err2.append(sq)
            # Which sensor is actually carrying this track right now. This is
            # what makes the fused row comparable to the single-sensor rows
            # instead of averaging over a different set of frames.
            if trk.last_sensor in by_sensor:
                by_sensor[trk.last_sensor].append(sq)
            P = trk.P[:6, :6]
            try:
                nees.append(float(e @ np.linalg.solve(P, e)))
            except np.linalg.LinAlgError:
                pass
            if trk.filter.last_nis is not None and np.isfinite(trk.filter.last_nis):
                nis.append(trk.filter.last_nis)

    err = np.sqrt(np.array(err2)) if err2 else np.empty(0)
    return {
        "rmse": float(np.sqrt(np.mean(err2))) if err2 else float("nan"),
        "median": float(np.median(err)) if err.size else float("nan"),
        "p95": float(np.percentile(err, 95)) if err.size else float("nan"),
        "nees": float(np.mean(nees)) if nees else float("nan"),
        "nis": float(np.mean(nis)) if nis else float("nan"),
        "coverage": frames_with_track / max(n_frames, 1),
        "initiated": sim.tracker.stats["initiated"],
        "confirmed": sim.tracker.stats["confirmed"],
        "samples": len(err2),
        "rmse_on_radar": _rmse(by_sensor["radar"]),
        "rmse_on_optical": _rmse(by_sensor["optical"]),
        "frac_on_optical": (len(by_sensor["optical"]) / len(err2)
                            if err2 else float("nan")),
    }


def _rmse(sq: list) -> float:
    return float(np.sqrt(np.mean(sq))) if sq else float("nan")


def summarize(name: str, results: list[dict]) -> None:
    def col(k):
        v = np.array([r[k] for r in results], dtype=float)
        return v[np.isfinite(v)]

    rmse, nees, nis, cov = col("rmse"), col("nees"), col("nis"), col("coverage")
    med, p95 = col("median"), col("p95")
    init = np.array([r["initiated"] for r in results], dtype=float)
    conf = np.array([r["confirmed"] for r in results], dtype=float)

    if rmse.size == 0:
        print(f"  {name:<14s}  no confirmed tracks in any trial")
        return

    # NEES is over a 6-state (position + velocity), so consistency means ~6.
    verdict = ("consistent" if 3.0 <= np.mean(nees) <= 12.0 else
               "OVERCONFIDENT" if np.mean(nees) > 12.0 else "conservative")
    print(f"  {name:<14s}  RMSE={np.mean(rmse):6.2f} m   "
          f"med={np.mean(med):5.2f}  p95={np.mean(p95):6.2f}   "
          f"NEES={np.mean(nees):6.2f} ({verdict})   "
          f"NIS={np.mean(nis):5.2f}   "
          f"held={np.mean(cov)*100:5.1f}%   "
          f"tracks {np.mean(conf):.1f}/{np.mean(init):.1f} conf/init")


def summarize_conditioned(results: list[dict]) -> None:
    """Split the fused run by which sensor last updated each track.

    Without this, the fused row is compared against single-sensor rows
    measured over entirely different frames, and fusion appears to lose to
    the sensor it is fusing.
    """
    def col(k):
        v = np.array([r[k] for r in results], dtype=float)
        return v[np.isfinite(v)]

    on_opt, on_rad, frac = (col("rmse_on_optical"), col("rmse_on_radar"),
                            col("frac_on_optical"))
    print()
    print("fused run, split by the sensor carrying the track:")
    if on_opt.size:
        print(f"    while optical is updating   RMSE={np.mean(on_opt):6.2f} m"
              f"   ({np.mean(frac)*100:.0f}% of scored frames)")
    else:
        print("    while optical is updating   never updated a confirmed track")
    if on_rad.size:
        print(f"    while radar is updating     RMSE={np.mean(on_rad):6.2f} m"
              f"   ({(1-np.mean(frac))*100:.0f}% of scored frames)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="maneuvering")
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--duration", type=float, default=30.0)
    args = ap.parse_args()

    print("=" * 78)
    print(f"MONTE CARLO — scenario={args.scenario} trials={args.trials} "
          f"duration={args.duration:.0f}s")
    print("=" * 78)
    print("\nsensor configuration comparison:")

    fused: list[dict] = []
    for name, radar, optical in (("radar only", True, False),
                                 ("optical only", False, True),
                                 ("fused", True, True)):
        results = [run_trial(args.scenario, seed, args.duration, radar, optical)
                   for seed in range(args.trials)]
        summarize(name, results)
        if radar and optical:
            fused = results

    if fused:
        summarize_conditioned(fused)

    print("\nNEES far above the state dimension means the covariance is")
    print("optimistic -- gates will be too tight and tracks will fragment.")
    print("Far below means the filter is throwing away accuracy it has.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
