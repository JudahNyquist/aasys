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

