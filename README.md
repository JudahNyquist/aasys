# AASYS
This is a sandbox simulation Anti-Aircraft/Air defense engine.

```bash
python3 run.py --list                                  # scenarios
python3 run.py --scenario hovering                     # 3D window
python3 run.py --scenario swarm --headless --duration 60
python3 run.py --scenario swarm --headless --record run   # + .npz and .csv
python3 -m unittest discover -s tests                  # 84 tests
python3 -m tools.bench_carve                           # carving perf gate
python3 -m tools.batch_eval --trials 6                 # Monte Carlo validation
python3 -m tools.regress                               # tracker regression gate
python3 -m tools.plot_run run.npz --out run.png        # needs requirements-dev
```

here's all the scenarios
```
single (default): One drone flying straight at the asset — the baseline.

maneuvering: A weaving drone; the case a single CV model fails on and the IMM should catch.

hovering: A hoverer in the radar's zero-Doppler notch (optical-only) plus an inbound crosser

swarm: 5-drone saturation raid from all sides; stresses association, dwell budget, magazine.

standoff: 3 fixed-wing threats at ~3 km, engaged by missile rather than gun.

orbiter: Drone loitering in a sustained turn; engage=False (track only).

diving: Cruise then terminal dive; uses its own lower grid + shorter-mast rig.

counterdrone: 3 jinking quads inside 1 km, stopped by counter-drone chasers.

cv-baseline: Same as maneuvering but with a single CV EKF, for IMM comparison.
```
