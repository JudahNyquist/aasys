"""Camera rig placement.

Geometry is the single biggest lever on carving quality. Cameras clustered
along one bearing leave the depth direction poorly constrained, so the
carved hull smears along the line of sight; spreading them in azimuth *and*
elevation is what buys an isotropic reconstruction. `tools/placement_study.py`
sweeps these presets to quantify the effect.
"""

from __future__ import annotations

import numpy as np

from .camera import Camera


class CameraRig:
    def __init__(self, cameras: list[Camera]) -> None:
        self.cameras = cameras

    def __len__(self) -> int:
        return len(self.cameras)

    def __iter__(self):
        return iter(self.cameras)

    def __getitem__(self, i) -> Camera:
        return self.cameras[i]

    @classmethod
    def ring(cls, n: int = 6, radius: float = 220.0, height: float = 12.0,
             aim: np.ndarray | None = None, fov_deg: float = 60.0,
             width: int = 640, height_px: int = 480,
             start_deg: float = 0.0) -> "CameraRig":
        """`n` cameras evenly spaced on a circle, all canted inward and up.

        The default aim point sits above the ring centre so the cameras look
        into the airspace being defended rather than at the ground.
        """
        if n < 4:
            raise ValueError("voxel carving needs at least 4 cameras")
        aim = np.array([0.0, 0.0, 60.0]) if aim is None else np.asarray(aim, float)
        cams = []
        for i in range(n):
            a = np.radians(start_deg) + 2.0 * np.pi * i / n
            eye = np.array([radius * np.cos(a), radius * np.sin(a), height])
            cams.append(Camera.from_look_at(
                eye, aim, fov_deg, width, height_px, name=f"cam{i}"))
        return cls(cams)

    @classmethod
    def ring_staggered(cls, n: int = 6, radius: float = 220.0,
                       heights: tuple[float, ...] = (8.0, 30.0),
                       **kw) -> "CameraRig":
        """Ring with alternating mast heights.

        Elevation diversity is what constrains the vertical axis; a perfectly
        coplanar ring reconstructs altitude far more weakly than range.
        """
        rig = cls.ring(n=n, radius=radius, **kw)
        aim = kw.get("aim")
        aim = np.array([0.0, 0.0, 60.0]) if aim is None else np.asarray(aim, float)
        out = []
        for i, cam in enumerate(rig.cameras):
            eye = cam.C.copy()
            eye[2] = heights[i % len(heights)]
            out.append(Camera.from_look_at(
                eye, aim, kw.get("fov_deg", 60.0), cam.width, cam.height,
                name=cam.name))
        return cls(out)

    def perturbed(self, rng, pos_sigma: float = 0.0,
                  rot_sigma_deg: float = 0.0) -> "CameraRig":
        return CameraRig([c.perturbed(rng, pos_sigma, rot_sigma_deg)
                          for c in self.cameras])
