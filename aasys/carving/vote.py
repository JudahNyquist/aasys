"""Voxel voting.

Two paths compute exactly the same answer.

**Sparse** (default). An occupied voxel must be lit in at least one camera,
so the union of the cameras' lit-voxel sets is a valid superset of the
result. Silhouettes light ~0.1% of pixels, so that superset is a few
thousand voxels rather than four million, and the carve costs what the
*scene* contains rather than what the *grid* holds.

**Dense**. The straightforward full-array scan. Used automatically when the
change masks are so full that the sparse candidate set stops being sparse
-- heavy pixel noise, or a target close enough to fill the frame.

Keeping both matters: the sparse path's advantage is a property of the
imagery, not a guarantee, and a carve that silently degrades under noise
would be worse than one that simply switches strategy.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .grid import VoxelGrid
from .lut import CarvingLUT


@dataclass
class CarveResult:
    indices: np.ndarray      # flat indices of occupied voxels
    votes: np.ndarray        # vote count for each occupied voxel
    grid: VoxelGrid
    path: str                # "sparse" or "dense"
    n_candidates: int = 0

    def __len__(self) -> int:
        return int(self.indices.size)

    def volume(self) -> np.ndarray:
        """Dense boolean occupancy shaped like the grid (for labelling)."""
        vol = np.zeros(self.grid.count, dtype=bool)
        vol[self.indices] = True
        return vol.reshape(self.grid.shape)

    def centers(self) -> np.ndarray:
        """World-space centres of the occupied voxels."""
        if self.indices.size == 0:
            return np.empty((0, 3))
        nx, ny, nz = self.grid.shape
        idx = self.indices.astype(np.int64)
        iz = idx % nz
        iy = (idx // nz) % ny
        ix = idx // (nz * ny)
        return self.grid.lo + (np.stack([ix, iy, iz], axis=1) + 0.5) * self.grid.size


class Carver:
    def __init__(self, lut: CarvingLUT, vote_threshold: int = 6,
                 min_cameras: int = 3,
                 max_lit_pixels: int = 40_000) -> None:
        """`vote_threshold` trades phantoms against holes: raising it toward
        the camera count suppresses ghost volumes but starts eating real
        hull wherever a silhouette pixel was missed. `min_cameras` refuses
        to trust any voxel too few cameras can see, keeping the
        poorly-constrained fringe of the volume out of the results.
        `max_lit_pixels` is where the sparse path hands off to the dense one.
        """
        self.lut = lut
        self.vote_threshold = int(vote_threshold)
        self.min_cameras = int(min_cameras)
        self.max_lit_pixels = int(max_lit_pixels)
        self._votes = np.zeros(lut.grid.count, dtype=np.uint8)

    # ---------------------------------------------------------------- sparse
    def _carve_sparse(self, masks, flats, lit=None) -> CarveResult:
        cand = np.concatenate([
            self.lut.candidates_from_mask(c, masks[c],
                                          None if lit is None else lit[c])
            for c in range(self.lut.n_cameras)
        ])
        if cand.size == 0:
            return CarveResult(np.empty(0, np.int64), np.empty(0, np.uint8),
                               self.lut.grid, "sparse", 0)
        cand = np.unique(cand)
        n_cand = int(cand.size)

        votes = np.zeros(n_cand, dtype=np.uint8)
        for c in range(self.lut.n_cameras):
            votes += self.lut.vis[c][cand] & flats[c][self.lut.pix[c][cand]]

        nvis = self.lut.n_visible[cand]
        thr = np.minimum(np.uint8(self.vote_threshold), nvis)
        keep = (votes >= thr) & (nvis >= self.min_cameras)
        return CarveResult(cand[keep], votes[keep], self.lut.grid,
                           "sparse", n_cand)

    # ----------------------------------------------------------------- dense
    def _carve_dense(self, flats) -> CarveResult:
        votes = self._votes
        votes[:] = 0
        for c in range(self.lut.n_cameras):
            votes += self.lut.vis[c] & flats[c][self.lut.pix[c]]

        occ = ((votes >= self.lut.threshold(self.vote_threshold))
               & (self.lut.n_visible >= self.min_cameras))
        idx = np.flatnonzero(occ)
        return CarveResult(idx, votes[idx], self.lut.grid, "dense",
                           self.lut.grid.count)

    # ------------------------------------------------------------------ api
    def carve(self, masks) -> CarveResult:
        if len(masks) != self.lut.n_cameras:
            raise ValueError(
                f"expected {self.lut.n_cameras} masks, got {len(masks)}")

        flats = [np.ascontiguousarray(m).ravel() for m in masks]

        # Locating the lit pixels is the only scan of the full masks either
        # path needs, so do it once and reuse it. Deciding the path from
        # `f.sum()` instead promotes 12M booleans to int64 per frame and cost
        # more than four times the carve it was selecting.
        lit = [np.flatnonzero(f) for f in flats]
        total_lit = int(sum(l.size for l in lit))
        if total_lit <= self.max_lit_pixels:
            return self._carve_sparse(masks, flats, lit)
        return self._carve_dense(flats)

    # Kept for the dense reference implementation used in equivalence tests.
    def occupancy_flat(self, masks) -> np.ndarray:
        res = self.carve(masks)
        out = np.zeros(self.lut.grid.count, dtype=bool)
        out[res.indices] = True
        return out

    def occupancy(self, masks) -> np.ndarray:
        return self.occupancy_flat(masks).reshape(self.lut.grid.shape)
