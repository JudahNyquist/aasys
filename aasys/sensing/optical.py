"""Optical sensor: cameras -> change masks -> voxel carving -> measurements.

Presents the whole carving chain behind the same `Sensor` interface the
radar uses, so the tracker cannot tell them apart beyond the measurement
model each supplies.

Two rigs are held deliberately. Imagery is generated from the *true*
camera poses, while carving projects using the *assumed* (calibrated)
poses. When those differ the hull shrinks and phantom volumes multiply,
which is exactly how calibration error behaves in the field -- and it
cannot be reproduced at all by a simulation that uses one rig for both.
"""

from __future__ import annotations

import numpy as np

from ..carving.cluster import blobs_to_measurements, extract_blobs
from ..carving.grid import VoxelGrid
from ..carving.lut import CarvingLUT
from ..carving.vote import Carver
from .base import Measurement, Sensor
from .rig import CameraRig
from .silhouette import apply_noise, rasterize


class OpticalCarvingSensor(Sensor):
    def __init__(self, rig: CameraRig, grid: VoxelGrid,
                 sensor_id: str = "optical",
                 rate_hz: float = 20.0,
                 vote_threshold: int | None = None,
                 min_cameras: int = 3,
                 p_false: float = 0.0,
                 p_miss: float = 0.0,
                 sync_jitter_s: float = 0.0,
                 calib_pos_sigma: float = 0.0,
                 calib_rot_sigma_deg: float = 0.0,
                 r_scale: float = 0.25,
                 min_voxels: int = 2,
                 rng=None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.true_rig = rig
        self.assumed_rig = (rig if (calib_pos_sigma <= 0 and calib_rot_sigma_deg <= 0)
                            else rig.perturbed(self.rng, calib_pos_sigma,
                                               calib_rot_sigma_deg))
        self.grid = grid
        self.sensor_id = sensor_id
        self.period = 1.0 / float(rate_hz)
        self.p_false = float(p_false)
        self.p_miss = float(p_miss)
        self.sync_jitter = float(sync_jitter_s)
        self.r_scale = float(r_scale)
        self.min_voxels = int(min_voxels)

        self.lut = CarvingLUT(grid, self.assumed_rig)
        self.carver = Carver(
            self.lut,
            vote_threshold=len(rig) if vote_threshold is None else vote_threshold,
            min_cameras=min_cameras)

        self._t_next = 0.0
        self.last_masks: list[np.ndarray] = []
        self.last_result = None
        self.last_blobs: list = []
        self.stats = {"frames": 0, "blobs": 0, "voxels": 0}

    @property
    def n_cameras(self) -> int:
        return len(self.true_rig)

    def in_volume(self, position: np.ndarray) -> bool:
        return bool(self.grid.contains(np.atleast_2d(position))[0])

    def _masks(self, t: float, targets: list) -> list[np.ndarray]:
        """Render each camera's change mask.

        Sync jitter is applied per camera by offsetting each target along its
        own velocity: unsynchronised shutters place a fast mover in a
        slightly different spot in every view, and the silhouette cones then
        fail to intersect where the target actually is. The effect scales
        with speed, which is why it bites hardest on precisely the targets
        that matter most.
        """
        masks = []
        for cam in self.true_rig:
            dt = (self.rng.normal(0.0, self.sync_jitter)
                  if self.sync_jitter > 0 else 0.0)
            pos, rad = [], []
            for tgt in targets:
                if not tgt.alive:
                    continue
                pos.append(tgt.position + tgt.velocity * dt)
                rad.append(tgt.radius)
            m = rasterize(cam, np.array(pos) if pos else np.empty((0, 3)),
                          np.array(rad) if rad else np.empty(0))
            if self.p_false > 0 or self.p_miss > 0:
                m = apply_noise(m, self.rng, self.p_false, self.p_miss)
            masks.append(m)
        return masks

    def sense(self, t: float, targets: list) -> list[Measurement]:
        if t < self._t_next:
            return []
        self._t_next = t + self.period

        masks = self._masks(t, targets)
        result = self.carver.carve(masks)
        blobs = extract_blobs(result, min_voxels=self.min_voxels)

        self.last_masks = masks
        self.last_result = result
        self.last_blobs = blobs
        self.stats["frames"] += 1
        self.stats["blobs"] += len(blobs)
        self.stats["voxels"] += len(result)

        return blobs_to_measurements(blobs, t, self.sensor_id,
                                     scale=self.r_scale)
