"""Phase 2 gate: voxel carving correctness and throughput.

Blocking criteria:
  1. A single sphere, noiseless, must reconstruct with its centroid inside
     one voxel of truth.
  2. The sparse and dense paths must agree exactly.
  3. Voting must sustain >= 30 Hz.

Run:  .venv/bin/python -m tools.bench_carve
"""

from __future__ import annotations

import time

import numpy as np

from aasys.carving.grid import VoxelGrid
from aasys.carving.lut import CarvingLUT
from aasys.carving.vote import Carver
from aasys.sensing.rig import CameraRig
from aasys.sensing.silhouette import rasterize

TARGET_HZ = 30.0


def timed(fn, reps=20):
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps


def main() -> int:
    print("=" * 68)
    print("PHASE 2 GATE — voxel carving")
    print("=" * 68)

    grid = VoxelGrid(lo=[-200.0, -200.0, 0.0], hi=[200.0, 200.0, 200.0], size=2.0)
    rig = CameraRig.ring_staggered(n=6, radius=260.0, fov_deg=70.0)

    t0 = time.perf_counter()
    lut = CarvingLUT(grid, rig)
    print(f"\n{grid.describe()}")
    print(f"cameras            {len(rig)}")
    print(f"LUT build          {time.perf_counter()-t0:.2f} s")

    cov = lut.coverage_stats()
    print(f"mean cams/voxel    {cov['mean_cameras_per_voxel']:.2f}")
    print(f"seen by all        {cov['frac_seen_by_all']*100:.1f}%")

    carver = Carver(lut, vote_threshold=len(rig), min_cameras=3)

    # ---- correctness: one sphere -------------------------------------
    truth = np.array([30.0, -20.0, 70.0])
    radius = 4.0
    masks = [rasterize(cam, truth[None, :], np.array([radius])) for cam in rig]
    print(f"\nsilhouette pixels  {[int(m.sum()) for m in masks]}"
          f"  ({sum(int(m.sum()) for m in masks)} total)")

    t0 = time.perf_counter()
    res = carver.carve(masks)
    print(f"inverted-index build (first call, amortised): "
          f"{time.perf_counter()-t0:.2f} s")
    print(f"LUT+index memory   {lut.memory_mb():.1f} MB")

    if len(res) == 0:
        print("\nFAIL: nothing carved")
        return 1

    centroid = res.centers().mean(axis=0)
    err = float(np.linalg.norm(centroid - truth))
    print(f"\npath               {res.path}")
    print(f"candidates         {res.n_candidates}  "
          f"({res.n_candidates/grid.count*100:.4f}% of grid)")
    print(f"occupied voxels    {len(res)}")
    print(f"centroid error     {err:.3f} m  (voxel = {grid.size} m)")
    print(f"hull volume        {len(res)*grid.voxel_volume:.1f} m^3 vs true "
          f"sphere {4/3*np.pi*radius**3:.1f} m^3")

    ok_centroid = err <= grid.size
    print(f"  -> centroid within one voxel: {'PASS' if ok_centroid else 'FAIL'}")

    # ---- sparse vs dense equivalence ---------------------------------
    flats = [np.ascontiguousarray(m).ravel() for m in masks]
    sparse = np.sort(carver._carve_sparse(masks, flats).indices)
    dense = np.sort(carver._carve_dense(flats).indices)
    ok_same = np.array_equal(sparse, dense)
    print(f"\nsparse {sparse.size} vs dense {dense.size} voxels")
    print(f"  -> paths agree exactly: {'PASS' if ok_same else 'FAIL'}")

    # ---- throughput ---------------------------------------------------
    dt_sparse = timed(lambda: carver._carve_sparse(masks, flats))
    dt_dense = timed(lambda: carver._carve_dense(flats), reps=5)
    print(f"\nsparse carve       {dt_sparse*1000:7.2f} ms  "
          f"= {1/dt_sparse:7.1f} Hz")
    print(f"dense carve        {dt_dense*1000:7.2f} ms  "
          f"= {1/dt_dense:7.1f} Hz")
    print(f"speedup            {dt_dense/dt_sparse:.1f}x")

    dt_full = timed(lambda: carver.carve(masks))
    hz = 1.0 / dt_full
    print(f"end-to-end carve   {dt_full*1000:7.2f} ms  = {hz:7.1f} Hz")
    ok_speed = hz >= TARGET_HZ
    print(f"  -> >= {TARGET_HZ:.0f} Hz: {'PASS' if ok_speed else 'FAIL'}")

    # ---- noise stress: does the sparse path degrade gracefully? -------
    print("\nnoise stress (sparse -> dense handoff):")
    rng = np.random.default_rng(0)
    for p_false in (0.0, 1e-5, 1e-4, 1e-3, 1e-2):
        noisy = [m | (rng.random(m.shape) < p_false) for m in masks]
        r = carver.carve(noisy)
        d = timed(lambda: carver.carve(noisy), reps=5)
        print(f"  p_false={p_false:<8g} lit={sum(int(m.sum()) for m in noisy):>7} "
              f"path={r.path:<7} cand={r.n_candidates:>8} "
              f"occ={len(r):>6} {d*1000:7.2f} ms")

    print("\n" + "=" * 68)
    verdict = ok_centroid and ok_speed and ok_same
    print("GATE:", "PASS" if verdict else "FAIL")
    print("=" * 68)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
