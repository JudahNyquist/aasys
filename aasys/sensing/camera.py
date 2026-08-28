"""Pinhole camera model.

Convention: +Z_cam points into the scene, +X_cam right, +Y_cam down, which
is the standard computer-vision arrangement and keeps the intrinsics matrix
in its familiar form.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.vecmath import look_at_rotation


@dataclass
class Camera:
    K: np.ndarray          # 3x3 intrinsics
    R: np.ndarray          # 3x3 rotation, world -> camera
    C: np.ndarray          # camera centre in world coordinates
    width: int
    height: int
    name: str = ""

    @classmethod
    def from_look_at(cls, eye, target, fov_deg: float, width: int, height: int,
                     up=None, name: str = "") -> "Camera":
        eye = np.asarray(eye, dtype=float)
        # Horizontal FOV fixes fx; square pixels then fix fy.
        f = 0.5 * width / np.tan(0.5 * np.radians(fov_deg))
        K = np.array([[f, 0.0, width / 2.0],
                      [0.0, f, height / 2.0],
                      [0.0, 0.0, 1.0]])
        return cls(K=K, R=look_at_rotation(eye, target, up), C=eye,
                   width=int(width), height=int(height), name=name)

    @property
    def fx(self) -> float:
        return float(self.K[0, 0])

    @property
    def forward(self) -> np.ndarray:
        """Viewing direction in world coordinates."""
        return self.R[2].copy()

    def to_camera(self, P: np.ndarray) -> np.ndarray:
        """World points (N,3) -> camera coordinates (N,3)."""
        P = np.asarray(P, dtype=float)
        return (P - self.C) @ self.R.T

    def project(self, P: np.ndarray):
        """World points (N,3) -> (uv (N,2) float, depth (N,), valid (N,) bool).

        `valid` requires the point to be strictly in front of the camera and
        inside the image. Points behind the camera must be culled rather
        than projected: the perspective divide by a negative depth folds
        them back into the image as phantom detections.
        """
        Pc = self.to_camera(P)
        z = Pc[:, 2]
        in_front = z > 1e-6
        zsafe = np.where(in_front, z, 1.0)

        uv = np.empty((Pc.shape[0], 2))
        uv[:, 0] = self.K[0, 0] * Pc[:, 0] / zsafe + self.K[0, 2]
        uv[:, 1] = self.K[1, 1] * Pc[:, 1] / zsafe + self.K[1, 2]

        valid = (
            in_front
            & (uv[:, 0] >= 0) & (uv[:, 0] < self.width)
            & (uv[:, 1] >= 0) & (uv[:, 1] < self.height)
        )
        return uv, z, valid

    def projected_radius_px(self, depth: np.ndarray, radius_m: np.ndarray):
        """Approximate on-image radius of a sphere of `radius_m` at `depth`.

        Exact only for a sphere centred on the optical axis, but targets are
        small relative to their range here, so the error is sub-pixel.
        """
        depth = np.maximum(np.asarray(depth, dtype=float), 1e-6)
        return self.fx * np.asarray(radius_m, dtype=float) / depth

    def perturbed(self, rng, pos_sigma: float = 0.0,
                  rot_sigma_deg: float = 0.0) -> "Camera":
        """Return a copy with calibration error applied.

        The simulation uses the true camera to generate imagery and the
        perturbed one to carve, which is exactly how real calibration error
        manifests: hulls shrink and phantom volumes multiply.
        """
        C = self.C + rng.normal(0.0, pos_sigma, 3) if pos_sigma > 0 else self.C.copy()
        R = self.R
        if rot_sigma_deg > 0:
            w = rng.normal(0.0, np.radians(rot_sigma_deg), 3)
            theta = float(np.linalg.norm(w))
            if theta > 1e-12:
                k = w / theta
                Kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
                dR = (np.eye(3) + np.sin(theta) * Kx
                      + (1 - np.cos(theta)) * (Kx @ Kx))
                R = dR @ R
        return Camera(K=self.K.copy(), R=R, C=C, width=self.width,
                      height=self.height, name=self.name + "~")
