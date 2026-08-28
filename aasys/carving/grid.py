"""Voxel grid geometry.

Cost is cubic in extent over voxel size, which is the constraint that
shapes the whole design:

    400 x 400 x 200 m at 2.0 m  ->   4.0 M voxels   (real-time)
    400 x 400 x 200 m at 1.0 m  ->  32.0 M voxels   (offline only)
    10 x 10 x 5 km   at 1.0 m   ->   5e11 voxels    (impossible)

Hence a coarse global grid for detection plus local refinement only where
something was actually found.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class VoxelGrid:
    lo: np.ndarray        # lower corner (3,)
    hi: np.ndarray        # upper corner (3,)
    size: float           # voxel edge length (m)

    def __post_init__(self):
        self.lo = np.asarray(self.lo, dtype=float)
        self.hi = np.asarray(self.hi, dtype=float)
        extent = self.hi - self.lo
        if np.any(extent <= 0):
            raise ValueError("VoxelGrid: hi must exceed lo on every axis")
        self.shape = tuple(int(np.ceil(e / self.size)) for e in extent)

    @property
    def count(self) -> int:
        return int(np.prod(self.shape))

    @property
    def voxel_volume(self) -> float:
        return self.size ** 3

    def centers_chunk(self, start: int, stop: int) -> np.ndarray:
        """World-space centres of flat voxel indices [start, stop).

        Chunked deliberately: materialising all centres for a 4M-voxel grid
        costs ~96 MB in float64, and the lookup-table build needs them only
        transiently.
        """
        nx, ny, nz = self.shape
        idx = np.arange(start, stop, dtype=np.int64)
        iz = idx % nz
        iy = (idx // nz) % ny
        ix = idx // (nz * ny)
        return np.stack([
            self.lo[0] + (ix + 0.5) * self.size,
            self.lo[1] + (iy + 0.5) * self.size,
            self.lo[2] + (iz + 0.5) * self.size,
        ], axis=1)

    def index_to_world(self, ijk: np.ndarray) -> np.ndarray:
        return self.lo + (np.asarray(ijk, dtype=float) + 0.5) * self.size

    def world_to_index(self, P: np.ndarray) -> np.ndarray:
        return np.floor((np.asarray(P, dtype=float) - self.lo) / self.size).astype(int)

    def contains(self, P: np.ndarray) -> np.ndarray:
        P = np.atleast_2d(np.asarray(P, dtype=float))
        return np.all((P >= self.lo) & (P < self.hi), axis=1)

    def resolution_check(self, target_diameter: float,
                         min_ratio: float = 1.5) -> dict:
        """Is this grid fine enough to carve a target of this size?

        Shape-from-silhouette marks a voxel occupied when its centre
        projects inside the silhouette in every camera that sees it. If a
        voxel's projected footprint is larger than the target's silhouette,
        that can only happen by chance alignment, and the target carves to
        nothing however many cameras are watching.

        Both the target and the voxel subtend angles proportional to 1/range,
        so their ratio is independent of distance and camera placement --
        the requirement collapses to a pure sizing rule:

            voxel_size <= target_diameter / 1.5

        Measured behaviour: ratio 1.07 carves 0 voxels, 1.20 carves 2,
        1.80 carves 7. Below ~1.5 the sensor is simply blind to the target.
        """
        ratio = float(target_diameter) / self.size
        return {
            "ratio": ratio,
            "ok": ratio >= min_ratio,
            "max_voxel_size": float(target_diameter) / min_ratio,
            "message": (
                f"target {target_diameter:.2f} m across vs {self.size:.2f} m "
                f"voxels -> ratio {ratio:.2f} "
                + ("(ok)" if ratio >= min_ratio else
                   f"(TOO COARSE: need voxel <= "
                   f"{float(target_diameter)/min_ratio:.2f} m)")),
        }

    def describe(self) -> str:
        nx, ny, nz = self.shape
        return (f"VoxelGrid {nx}x{ny}x{nz} = {self.count/1e6:.2f}M voxels "
                f"@ {self.size} m")
