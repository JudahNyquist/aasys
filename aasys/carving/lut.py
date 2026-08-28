"""Precomputed voxel -> pixel lookup tables.

This is what makes voxel carving real-time in pure Python.

Voting naively means projecting every voxel into every camera every frame:
4M voxels x 6 cameras of trigonometry per frame is hopeless. But the
cameras never move, so that mapping is *constant for the entire run*.
Computing it once at startup reduces per-frame voting to a handful of
fancy-index gathers over flat arrays -- no trigonometry, no matrix
multiplies, and no Python-level loop over voxels.

The tables are built from the *assumed* camera calibration. Feeding a
perturbed rig here while generating imagery from the true rig is exactly
how calibration error behaves in the field.
"""

from __future__ import annotations

import numpy as np

from ..sensing.rig import CameraRig
from .grid import VoxelGrid


class CarvingLUT:
    """Static projection tables for a fixed grid and a fixed camera rig."""

    def __init__(self, grid: VoxelGrid, rig: CameraRig,
                 chunk: int = 500_000) -> None:
        self.grid = grid
        self.rig = rig
        n_vox = grid.count
        n_cam = len(rig)

        # pix: flat pixel index per voxel, per camera. Entries for voxels a
        # camera cannot see are clamped to 0 -- the gather still reads them,
        # so they must point somewhere in range; `vis` then masks them out.
        self.pix = np.zeros((n_cam, n_vox), dtype=np.int32)
        self.vis = np.zeros((n_cam, n_vox), dtype=bool)

        for start in range(0, n_vox, chunk):
            stop = min(start + chunk, n_vox)
            centers = grid.centers_chunk(start, stop)
            for c, cam in enumerate(rig):
                uv, _, valid = cam.project(centers)
                u = np.clip(uv[:, 0].astype(np.int32), 0, cam.width - 1)
                v = np.clip(uv[:, 1].astype(np.int32), 0, cam.height - 1)
                flat = v.astype(np.int32) * cam.width + u
                self.pix[c, start:stop] = np.where(valid, flat, 0)
                self.vis[c, start:stop] = valid

        # How many cameras can physically see each voxel. Static, so the
        # per-voxel vote threshold below is free.
        self.n_visible = self.vis.sum(axis=0).astype(np.uint8)
        self._thresh_cache: dict[int, np.ndarray] = {}
        self._inverted_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def n_cameras(self) -> int:
        return self.pix.shape[0]

    def threshold(self, k: int) -> np.ndarray:
        """Per-voxel vote threshold, capped by how many cameras can see it.

        Thresholding against a flat `k` is a real bug, not a nicety: voxels
        near the edge of the volume fall outside some cameras' fields of
        view, so they can never reach `k` votes and the margins of the
        working volume carve into permanent dead zones. A camera that
        physically cannot see a voxel must not count as voting against it.
        """
        if k not in self._thresh_cache:
            self._thresh_cache[k] = np.minimum(
                np.uint8(k), self.n_visible).astype(np.uint8)
        return self._thresh_cache[k]

    def inverted(self, c: int) -> tuple[np.ndarray, np.ndarray]:
        """Pixel -> voxel index, the reverse of `pix`. Built lazily per camera.

        Silhouettes light on the order of 0.1% of pixels, so scanning all
        four million voxels per camera to find the few thousand that matter
        is almost entirely wasted work. This index inverts the mapping:
        given the lit pixels, it hands back exactly the voxels projecting
        into them, making the carve cost scale with scene content instead
        of grid size.

        Returns `(voxels_sorted_by_pixel, starts)` where the voxels for
        pixel p occupy the slice `[starts[p]:starts[p+1]]`.
        """
        if c in self._inverted_cache:
            return self._inverted_cache[c]

        cam = self.rig[c]
        n_pixels = cam.width * cam.height
        visible = np.flatnonzero(self.vis[c]).astype(np.int32)
        p = self.pix[c][visible]
        order = np.argsort(p, kind="stable")
        voxels = visible[order]
        starts = np.searchsorted(p[order], np.arange(n_pixels + 1)).astype(np.int64)

        self._inverted_cache[c] = (voxels, starts)
        return self._inverted_cache[c]

    def candidates_from_mask(self, c: int, mask: np.ndarray,
                             lit: np.ndarray | None = None) -> np.ndarray:
        """Voxel indices projecting into any lit pixel of `mask` in camera `c`.

        `lit` is the flat indices of the lit pixels. The caller usually has
        already computed it -- deciding between the sparse and dense paths
        needs the same scan -- so passing it in avoids walking two million
        booleans a second time per camera per frame.
        """
        voxels, starts = self.inverted(c)
        if lit is None:
            lit = np.flatnonzero(np.ascontiguousarray(mask).ravel())
        if lit.size == 0:
            return np.empty(0, dtype=np.int32)

        lo = starts[lit]
        hi = starts[lit + 1]
        counts = hi - lo
        total = int(counts.sum())
        if total == 0:
            return np.empty(0, dtype=np.int32)

        # Ragged range expansion: concatenate [lo_i, hi_i) for every lit pixel
        # without a Python loop.
        ends = np.cumsum(counts)
        out_start = ends - counts
        flat = (np.arange(total, dtype=np.int64)
                - np.repeat(out_start, counts)
                + np.repeat(lo, counts))
        return voxels[flat]

    def memory_mb(self) -> float:
        base = self.pix.nbytes + self.vis.nbytes + self.n_visible.nbytes
        inv = sum(v.nbytes + s.nbytes for v, s in self._inverted_cache.values())
        return (base + inv) / 1e6

    def coverage_check(self, min_cameras: int = 3) -> dict:
        """Is this grid actually visible to this rig?

        The companion to `VoxelGrid.resolution_check`. That rule catches a
        grid too coarse to carve its targets; this one catches a grid whose
        volume the cameras cannot see, which fails in exactly the same way --
        the sensor reports nothing -- but for the opposite reason and with no
        symptom at all until someone notices the blob count is zero.

        A voxel that fewer than `min_cameras` cameras can see is refused by
        the carver by construction, so it can never be occupied no matter
        what flies through it. Volume made entirely of those voxels is paid
        for in memory and lookup-table build time and can return nothing.

        Reports the altitude band that is actually usable, since coverage is
        lost at the floor and the ceiling rather than uniformly: cameras on
        short masts aimed at mid-altitude simply cannot see the ground under
        themselves.
        """
        usable = self.n_visible.reshape(self.grid.shape) >= min_cameras
        per_layer = usable.reshape(-1, usable.shape[2]).mean(axis=0)
        live = np.flatnonzero(per_layer > 0.0)
        zs = self.grid.lo[2] + (np.arange(usable.shape[2]) + 0.5) * self.grid.size
        frac = float(usable.mean())

        if live.size == 0:
            return {"frac_usable": 0.0, "z_lo": None, "z_hi": None, "ok": False,
                    "message": (f"carving volume is invisible to this rig: no "
                                f"voxel is seen by {min_cameras} cameras")}

        z_lo, z_hi = float(zs[live[0]]), float(zs[live[-1]])
        ok = frac >= 0.5
        msg = (f"rig covers {frac*100:.0f}% of the volume "
               f"({min_cameras}+ cameras), usable band "
               f"z = {z_lo:.0f}-{z_hi:.0f} m")
        if z_lo > self.grid.lo[2] + self.grid.size:
            msg += (f"; nothing below {z_lo:.0f} m can carve "
                    f"(grid floor is {self.grid.lo[2]:.0f} m)")
        return {"frac_usable": frac, "z_lo": z_lo, "z_hi": z_hi, "ok": ok,
                "message": msg}

    def coverage_stats(self) -> dict:
        """Diagnostics on how well the rig covers the volume."""
        counts = np.bincount(self.n_visible, minlength=self.n_cameras + 1)
        total = self.grid.count
        return {
            "voxels": total,
            "mean_cameras_per_voxel": float(self.n_visible.mean()),
            "frac_seen_by_all": float(counts[self.n_cameras] / total),
            "frac_seen_by_none": float(counts[0] / total),
            "histogram": counts.tolist(),
        }
