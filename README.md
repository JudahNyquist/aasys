# aasys

A 3D air-defence simulation built around **object tracking**: flying targets,
two very different sensors, a fused estimator, and effectors that shoot at
what the estimator believes.

The rule the whole thing is built on: **nothing downstream ever sees ground
truth.** Sensor noise, camera geometry, calibration error and clock jitter
propagate into track quality, and track quality decides whether rounds hit.
Truth is used for exactly two things — generating sensor measurements, and
scoring afterwards. That rule is machine-checked, not merely intended: see
`aasys/core/truth_guard.py`.

## Running it

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

Viewer: drag orbit · scroll zoom · space pause · `1` voxels · `2` frusta ·
`3` ellipsoids · `4` detections · `5` trails · `F` follow · `R` reset.

## The two sensors

They fail in opposite ways, which is the reason for having both.

|                | Voxel carving (optical)   | Phased-array radar        |
| -------------- | ------------------------- | ------------------------- |
| Range          | bounded bubble (~100 m)   | km-scale                  |
| Accuracy       | **0.34 m** RMSE           | 2.82 m RMSE               |
| Measurement    | Cartesian — linear        | spherical r/az/el — nonlinear |
| Velocity       | inferred only             | **direct Doppler**        |
| Blind to       | targets under ~1.5 voxels | anything near zero Doppler |

**Optical.** Six cameras detect changed pixels; every changed pixel
back-projects into a voxel grid, and voxels seen as changed by enough
cameras are marked occupied. That carved visual hull is clustered into
detections.

**Radar.** An electronically steered beam interleaves a broad **search**
raster with narrow **track** dwells, and must ration a finite time budget
between them. Detection follows the radar equation (`SNR ∝ RCS/R⁴`), and
angle accuracy comes from monopulse — `beamwidth/(k·√SNR)`, so precision
*improves* as a target closes: cross-range error falls from 24 m at 2 km to
0.4 m at 500 m.

The **MTI clutter notch** is the interesting failure: returns near zero
Doppler are indistinguishable from ground clutter and get filtered out, so a
hovering drone is invisible to radar however strong its echo. Run
`--scenario hovering` to watch radar reject it a few hundred times while
voxel carving holds the track at 0.1 m error.

Fusion measured over 4 trials on an evading target:

```
radar only      RMSE 2.82 m   NEES 5.83   track held 84.4%
optical only    RMSE 0.34 m   NEES 3.88   track held 17.2%
fused           RMSE 0.71 m   NEES 5.48   track held 87.7%
```

Comparing those three RMSE numbers directly is a trap, and it is worth
naming because the obvious reading is backwards. They are averaged over
different populations: optical only sees a target inside its bubble, so its
error covers the small, close, well-conditioned fraction of the flight it
can see at all, while the fused number covers the whole engagement including
everything only radar can reach. Splitting the fused run by which sensor is
actually carrying the track is the comparison that means something:

```
fused, while optical is updating   RMSE 0.25 m   (16% of scored frames)
fused, while radar is updating     RMSE 0.76 m   (84% of scored frames)
```

Read that way, fusion beats each sensor on its own ground — 0.25 m against
optical's 0.34 m, 0.76 m against radar's 2.82 m — while holding the track
87.7% of the time against optical's 17.2%. `tools/batch_eval.py` prints both
tables.

## Three things worth knowing about voxel carving

**It is real-time only because the cameras never move.** Projecting 5M
voxels into 6 cameras every frame is hopeless in Python. But that mapping is
constant for the whole run, so it is computed once at startup. Per frame the
carve is then a handful of gathers over flat arrays — and because an
occupied voxel must be lit in at least one camera, an inverted pixel→voxel
index narrows candidates to the ~0.5% of the grid that could possibly
matter. Measured: **101 ms → 5.5 ms, an 18.4× speedup**, with the fast and
slow paths verified to produce identical output.

**Voxel size is not a free parameter.** A voxel is marked occupied when its
centre projects inside the silhouette in every camera. If the voxel's
projected footprint is larger than the target's silhouette that only happens
by chance, and the target carves to nothing. Both shrink as 1/range, so the
requirement is distance- and camera-independent:

```
voxel_size ≤ target_diameter / 1.5
```

Measured: ratio 1.07 → 0 voxels, 1.20 → 2, 1.80 → 7. `VoxelGrid.resolution_check()`
enforces it; below the threshold the sensor is simply blind.

**A grid can also be the wrong shape.** Too coarse and the sensor is blind;
too tall and most of it is invisible. Cameras on 6–26 m masts aimed at
mid-altitude cannot see the ground beneath themselves, and the carver
refuses any voxel fewer than `min_cameras` can see — so the bottom of a
naively-sized bubble holds voxels that can never be occupied whatever flies
through them. That failure looks exactly like a sensor with nothing to see.
`CarvingLUT.coverage_check()` measures the usable band and prints it at
startup:

```
rig covers 56% of the volume (3+ cameras), usable band z = 18-85 m
```

Sizing the default bubble to that band removed 16% of the grid without
changing a single carved voxel.

## Estimation

Measurements are self-describing — each carries its own model, Jacobian and
residual rule — so one EKF ingests both Cartesian blob centroids and
nonlinear spherical-Doppler returns without special-casing. Adding a sensor
means writing a model, not touching the filter.

- **IMM** over `{CV, CA, CT±15°/s, CT±35°/s}`. On an orbiting target it puts
  most of its mass on the matching turn model and correctly rejects the
  opposite one.
- **GNN association** (Hungarian) over negative log-likelihood, so crossing
  tracks do not get swapped the way greedy nearest-neighbour swaps them.
- **Track lifecycle** tentative → confirmed → coasting → deleted. This is
  also the phantom filter: optical ghosts flicker and never accumulate the
  hits confirmation requires.
- Process noise was **fitted, not guessed** — swept against NEES until the
  claimed covariance matched real error (see `tools/batch_eval.py`).

The measurement covariance for optical detections is derived from the
spatial second moment of each carved blob, so when camera geometry
constrains a direction poorly the hull is genuinely elongated along it and
the filter distrusts that axis on its own.

**The array is scheduled by track quality, not at a fixed rate.** A confirmed
track's revisit interval is driven by two things the tracker already
computes: its position uncertainty, which grows through every coast, and the
IMM's manoeuvre probability, which says whether the prediction will go stale
before the next look. A drone flying straight and level is revisited every
0.5 s, freeing beam time for search; one breaking hard or drifting out of
covariance gets 0.1 s. Making the schedule adaptive rather than fixed cut
radar-only RMSE from 6.04 m to 2.82 m and stopped the tracker fragmenting a
single target into three successive tracks — the engagement is now one
continuous track, with search frame time essentially unchanged.

## Effectors

**Gun** — unguided. Miss distance is the track error at the trigger plus
dispersion, and nothing downstream can recover it, which makes it the honest
instrument for judging the tracker. Needs a real ballistic solution: drag
adds 21% to time of flight at 500 m and 188% at 2.5 km, and time of flight
is what sets the lead.

That honesty depends on resolving hits against the path a round *flew*, not
the points it was sampled at. A round leaves the muzzle at 1000 m/s while
physics steps at 120 Hz, so it advances 8.3 m per step against a ~1 m lethal
radius: testing sampled positions alone scores a round that passed straight
through the target as a miss, and gun accuracy becomes a property of the
integrator's step size rather than of the fire-control solution. Impacts and
proximity fuzes both resolve against the swept segment
(`core.vecmath.segment_distance`), and `tests/test_fire_control.py` pins the
outcome to be independent of the step size.

**Interceptor** — proportional navigation, `a = N·λ̇ × V`, driving
line-of-sight rotation to zero. Constant bearing with closing range *is* a
collision, so nulling that rotation is the intercept condition. Corrects in
flight, so it tolerates a far looser launch solution.

Both the missile and the counter-drone carry a terminal seeker; the
difference between them is the cone, not the presence of one. It buys what
theory says it should and no more — against close-in manoeuvring drones,
where the ground track is poor, it cuts mean miss distance measurably;
against well-tracked standoff targets it changes nothing.

Doctrine is layered, and fire discipline is where filter quality becomes
policy: the gun refuses to fire on a track with σ > 2 m, the missile
tolerates 20 m. A target that is not closing is discounted — unless it is
*loitering*, because a drone parked over the defended asset is not
approaching only for the reason that it has already arrived.

## Layout

```
aasys/
  core/          RK4, atmosphere, ENU frames, seeded RNG, truth guard
  entities/      targets + flight profiles
  sensing/       camera, rig, silhouette, radar, optical, measurement models
  carving/       grid, projection LUT, voting, clustering
  tracking/      motion models, EKF/IMM, gating, GNN, track lifecycle
  fire_control/  intercept, guidance, effectors, seeker, engagement
  analysis/      per-frame recording of estimate vs truth
  render/        pyglet 3D viewer
tests/           84 tests
tools/           bench_carve, batch_eval, regress, plot_run
```

The simulation core is pure numpy and headless — physics, carving, radar and
tracking are all testable and Monte-Carlo-able with no window open. Physics
runs at a fixed 120 Hz step independent of sensor and frame rates, and every
run is reproducible from `--seed`. `swarm` runs at about 2.5× realtime.

## Checking it

Four gates, each guarding something the others do not:

- `unittest` — component behaviour, including the Jacobians checked against
  numerical differentiation.
- `tools/bench_carve` — carving throughput, and that the sparse and dense
  paths agree exactly.
- `tools/regress` — tracker RMSE, NEES and track continuity against recorded
  bounds. Tuning changes move these without breaking a single unit test.
- `aasys/core/truth_guard` — the invariant at the top of this file. It swaps
  in a guarding attribute accessor for the duration of a run and raises if
  any module outside the sensors and `lethality` reads a target's real
  state. `tests/test_truth_guard.py` runs it over four full engagements, and
  also checks that the guard itself fires when it should, because a guard
  that never triggers is indistinguishable from no guard.

For diagnosis rather than gating, `--record` writes a per-frame `.npz` and
CSV pairing every estimate with truth, and `tools/plot_run` draws error,
NEES against its consistency band, and IMM manoeuvre probability on a shared
time axis. An RMSE tells you the tracker got worse; those tell you it lost
lock for 1.4 s during a turn while the bank sat on the wrong model.

## Requirements

Python 3.11+, numpy, scipy, pyglet (all prebuilt wheels — no compiler
needed). `requirements-dev.txt` adds matplotlib, for `tools/plot_run` only.
