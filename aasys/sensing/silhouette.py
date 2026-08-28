"""Change-mask generation.

The background is static and targets are the only movers, so a real
background-subtraction stage would return exactly the union of the target
silhouettes plus sensor noise. Rasterizing projected discs reproduces that
directly and costs a tiny fraction of rendering six full images per frame.

Targets rasterize into one shared mask per camera, so mutual occlusion
merges silhouettes for free -- which is precisely the condition that breeds
phantom volumes downstream.
"""

from __future__ import annotations

import numpy as np

from .camera import Camera


def rasterize(camera: Camera, positions: np.ndarray, radii: np.ndarray,
              min_radius_px: float = 0.4,
              out: np.ndarray | None = None) -> np.ndarray:
    """Union of projected target discs as a boolean (H, W) mask."""
    mask = np.zeros((camera.height, camera.width), dtype=bool) if out is None else out
    if len(positions) == 0:
        return mask

    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    radii = np.atleast_1d(np.asarray(radii, dtype=float))
    uv, depth, _ = camera.project(positions)
    rad_px = camera.projected_radius_px(depth, radii)

    for i in range(len(positions)):
        # Cull behind the camera; never rely on `valid` here, since a target
        # whose centre is off-image can still have its disc overlap the frame.
        if depth[i] <= 1e-6 or rad_px[i] < min_radius_px:
            continue

        u, v, r = uv[i, 0], uv[i, 1], rad_px[i]
        u0, u1 = int(np.floor(u - r)), int(np.ceil(u + r)) + 1
        v0, v1 = int(np.floor(v - r)), int(np.ceil(v + r)) + 1
        u0, u1 = max(u0, 0), min(u1, camera.width)
        v0, v1 = max(v0, 0), min(v1, camera.height)
        if u0 >= u1 or v0 >= v1:
            continue

        # Sub-pixel targets still trip a detector; mark the single pixel.
        if r < 0.75:
            mask[int(v), int(u)] = True
            continue

        vv = np.arange(v0, v1)[:, None] + 0.5
        uu = np.arange(u0, u1)[None, :] + 0.5
        mask[v0:v1, u0:u1] |= ((uu - u) ** 2 + (vv - v) ** 2) <= r * r

    return mask


def apply_noise(mask: np.ndarray, rng, p_false: float = 0.0,
                p_miss: float = 0.0) -> np.ndarray:
    """Sensor imperfection on a change mask.

    `p_false` is the per-pixel false-change rate (sensor noise, moving
    foliage, shadows) which seeds phantom votes. `p_miss` is the per-pixel
    probability a genuine silhouette pixel goes undetected because the
    target does not contrast against its background, which punches holes in
    the hull and can split one blob into several.
    """
    out = mask
    if p_miss > 0:
        out = out & (rng.random(mask.shape) >= p_miss)
    if p_false > 0:
        out = out | (rng.random(mask.shape) < p_false)
    return out
