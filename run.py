#!/usr/bin/env python3
"""aasys entry point.

    python3 run.py --list
    python3 run.py --scenario single --headless --duration 40
    python3 run.py --scenario swarm
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from aasys.scenarios import SCENARIOS, build
from aasys.sim import Simulation


def main() -> int:
    ap = argparse.ArgumentParser(description="voxel-carving optical + radar air defence sim")
    ap.add_argument("--scenario", default="single")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--duration", type=float, default=45.0)
    ap.add_argument("--headless", action="store_true",
                    help="run without a window and print a report")
    ap.add_argument("--no-engage", action="store_true",
                    help="track only; do not shoot")
    ap.add_argument("--record", metavar="PATH",
                    help="write a per-frame .npz + .csv of estimates vs truth")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        print("scenarios:")
        for k, fn in sorted(SCENARIOS.items()):
            doc = (fn.__doc__ or "").strip().split("\n")[0]
            print(f"  {k:14s} {doc}")
        return 0

    cfg = build(args.scenario)
    if args.no_engage:
        cfg["engage"] = False

    print(f"scenario={args.scenario} seed={args.seed}")
    print(f"  {cfg['grid'].describe()}")
    chk = cfg["grid"].resolution_check(0.9)
    print(f"  {chk['message']}")
    t0 = time.perf_counter()
    sim = Simulation(cfg, seed=args.seed)
    print(f"  build {time.perf_counter()-t0:.1f}s"
          + (f", LUT {sim.optical.lut.memory_mb():.0f} MB" if sim.optical else ""))
    if sim.radar:
        print(f"  {sim.radar.describe()}")
    if sim.optical is not None:
        print(f"  {sim.optical.lut.coverage_check(sim.optical.carver.min_cameras)['message']}")

    recorder = None
    if args.record:
        from aasys.analysis import Recorder
        recorder = Recorder()

    if args.headless:
        t0 = time.perf_counter()
        n = int(args.duration / sim.dt)
        for i in range(n):
            sim.step()
            if recorder is not None:
                recorder.capture(sim)
        wall = time.perf_counter() - t0
        print(f"\nsimulated {args.duration:.0f} s in {wall:.1f} s wall "
              f"({args.duration/wall:.2f}x realtime)\n")
        print(sim.report())
        if recorder is not None:
            print()
            print(recorder.summary())
            print(f"recorded to {recorder.save(args.record)}")
        if sim.events:
            print("\nengagements:")
            for e in sim.events:
                print(" ", e)
        return 0

    if recorder is not None:
        raise SystemExit("--record requires --headless")

    from aasys.render.app import run_window
    run_window(sim, duration=args.duration)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
