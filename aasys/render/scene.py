"""Primitive building for the 3D view.

Everything is drawn as points and lines with per-vertex colour -- a
deliberate choice, since the interesting content here is geometric (voxel
occupancy, uncertainty ellipsoids, camera frusta, trajectory trails) and a
wireframe shows all of it at once with nothing hidden behind a surface.

Geometry is built with numpy and emitted in bulk. The obvious
implementation -- a Python loop appending three floats per vertex -- costs
about 17 ms per frame at realistic scene complexity, which alone caps the
window near 20 fps and makes the whole simulation run at a third of real
time. Building the same geometry as vectorised arrays is roughly 80x
faster, and is what lets the physics, carving and tracking actually run
live rather than in slow motion.

Upload is the other half. Counter-intuitively pyglet ingests a plain
Python list faster than a numpy array, because it converts element by
element through ctypes either way and skips a conversion layer for lists.
So: build with numpy, hand over lists.
"""

from __future__ import annotations

import numpy as np

VERT_SRC = """#version 330 core
in vec3 position;
in vec4 colors;
out vec4 vertex_color;
uniform WindowBlock { mat4 projection; mat4 view; } window;
uniform float point_size;
void main() {
    gl_Position = window.projection * window.view * vec4(position, 1.0);
    gl_PointSize = point_size;
    vertex_color = colors;
}
"""

FRAG_SRC = """#version 330 core
in vec4 vertex_color;
out vec4 final_color;
void main() { final_color = vertex_color; }
"""

# Twelve edges of a unit cube centred on the origin, as 24 endpoints.
_CUBE_EDGES = np.array([
    [-.5, -.5, -.5], [+.5, -.5, -.5], [+.5, -.5, -.5], [+.5, +.5, -.5],
    [+.5, +.5, -.5], [-.5, +.5, -.5], [-.5, +.5, -.5], [-.5, -.5, -.5],
    [-.5, -.5, +.5], [+.5, -.5, +.5], [+.5, -.5, +.5], [+.5, +.5, +.5],
    [+.5, +.5, +.5], [-.5, +.5, +.5], [-.5, +.5, +.5], [-.5, -.5, +.5],
    [-.5, -.5, -.5], [-.5, -.5, +.5], [+.5, -.5, -.5], [+.5, -.5, +.5],
    [+.5, +.5, -.5], [+.5, +.5, +.5], [-.5, +.5, -.5], [-.5, +.5, +.5],
], dtype=np.float32)

_CROSS_DIRS = np.array([
    [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, -1], [0, 0, 1],
], dtype=np.float32)

EMPTY = np.empty((0, 3), dtype=np.float32)


class GeometryBuffer:
    """Accumulates line and point geometry, then emits it once."""

    def __init__(self) -> None:
        self._lv: list[np.ndarray] = []
        self._lc: list[np.ndarray] = []
        self._pv: list[np.ndarray] = []
        self._pc: list[np.ndarray] = []

    def add_lines(self, verts: np.ndarray, color) -> None:
        if verts is None or len(verts) == 0:
            return
        verts = np.asarray(verts, dtype=np.float32).reshape(-1, 3)
        self._lv.append(verts)
        self._lc.append(np.tile(np.asarray(color, dtype=np.float32),
                                (len(verts), 1)))

    def add_lines_colored(self, verts: np.ndarray, colors: np.ndarray) -> None:
        if verts is None or len(verts) == 0:
            return
        self._lv.append(np.asarray(verts, dtype=np.float32).reshape(-1, 3))
        self._lc.append(np.asarray(colors, dtype=np.float32).reshape(-1, 4))

    def add_points(self, pts: np.ndarray, color) -> None:
        if pts is None or len(pts) == 0:
            return
        pts = np.asarray(pts, dtype=np.float32).reshape(-1, 3)
        self._pv.append(pts)
        self._pc.append(np.tile(np.asarray(color, dtype=np.float32),
                                (len(pts), 1)))

    def merge(self, other: "GeometryBuffer") -> None:
        self._lv += other._lv
        self._lc += other._lc
        self._pv += other._pv
        self._pc += other._pc

    def build(self):
        """Return (line_pos, line_col, point_pos, point_col) as flat lists."""
        def cat(chunks, width):
            if not chunks:
                return []
            return np.concatenate(chunks).ravel().tolist()
        return (cat(self._lv, 3), cat(self._lc, 4),
                cat(self._pv, 3), cat(self._pc, 4))

    @property
    def n_line_verts(self) -> int:
        return sum(len(a) for a in self._lv)


# --------------------------------------------------------------- primitives
def ground_grid(half: float = 400.0, step: float = 50.0, z: float = 0.0):
    """Reference grid on the ground plane. Static -- build once."""
    n = int(half / step)
    c = np.arange(-n, n + 1, dtype=np.float32) * step
    m = len(c)
    v = np.empty((m * 4, 3), dtype=np.float32)
    v[0::4] = np.stack([np.full(m, -half), c, np.full(m, z)], axis=1)
    v[1::4] = np.stack([np.full(m, +half), c, np.full(m, z)], axis=1)
    v[2::4] = np.stack([c, np.full(m, -half), np.full(m, z)], axis=1)
    v[3::4] = np.stack([c, np.full(m, +half), np.full(m, z)], axis=1)

    col = np.tile(np.array([0.30, 0.38, 0.44, 1.0], dtype=np.float32), (m * 4, 1))
    axis = np.abs(c) < 1e-6
    idx = np.flatnonzero(axis)
    for i in idx:                       # at most one row and one column
        col[i * 4:(i + 1) * 4] = [0.45, 0.62, 0.72, 1.0]
    return v, col


def box_wire(lo, hi) -> np.ndarray:
    lo = np.asarray(lo, dtype=np.float32)
    hi = np.asarray(hi, dtype=np.float32)
    return _CUBE_EDGES * (hi - lo) + (lo + hi) * 0.5


def cubes_wire(centers: np.ndarray, size: float) -> np.ndarray:
    """Wireframe cubes for many centres at once.

    Voxel occupancy can be hundreds of cubes; looping in Python over twelve
    edges each is exactly the cost this module exists to avoid.
    """
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    if len(centers) == 0:
        return EMPTY
    return (centers[:, None, :] + _CUBE_EDGES[None, :, :] * size).reshape(-1, 3)


def crosses(centers: np.ndarray, size: float) -> np.ndarray:
    """Three-axis tick markers for many positions at once."""
    centers = np.asarray(centers, dtype=np.float32).reshape(-1, 3)
    if len(centers) == 0:
        return EMPTY
    return (centers[:, None, :] + _CROSS_DIRS[None, :, :] * size).reshape(-1, 3)


def polyline(points, max_points: int = 400, stride: int = 1) -> np.ndarray:
    """Line strip as GL_LINES pairs.

    `stride` decimates long trails: at render scale a trail sampled every
    third point is indistinguishable from one sampled every point, and costs
    a third of the vertices.
    """
    if points is None or len(points) < 2:
        return EMPTY
    arr = np.asarray(points[-max_points:], dtype=np.float32)
    if stride > 1 and len(arr) > 2 * stride:
        arr = np.concatenate([arr[::stride], arr[-1:]])
    if len(arr) < 2:
        return EMPTY
    out = np.empty((2 * (len(arr) - 1), 3), dtype=np.float32)
    out[0::2] = arr[:-1]
    out[1::2] = arr[1:]
    return out


def covariance_ellipsoid(mean, cov, sigma: float = 3.0, segments: int = 24):
    """Wireframe 3-sigma ellipsoid of a position covariance.

    The most informative overlay in the view: it swells while a track coasts
    unobserved, snaps tight the instant a measurement lands, and stretches
    along whichever axis the current sensor geometry constrains least.
    """
    try:
        evals, evecs = np.linalg.eigh(np.asarray(cov, dtype=float))
    except np.linalg.LinAlgError:
        return EMPTY
    axes = sigma * np.sqrt(np.maximum(evals, 1e-9))
    th = np.linspace(0, 2 * np.pi, segments, endpoint=False)
    cs, sn = np.cos(th), np.sin(th)

    rings = []
    for a, b in ((0, 1), (1, 2), (0, 2)):
        local = np.zeros((segments, 3))
        local[:, a] = axes[a] * cs
        local[:, b] = axes[b] * sn
        pts = np.asarray(mean, dtype=float) + local @ evecs.T
        seg = np.empty((2 * segments, 3), dtype=np.float32)
        seg[0::2] = pts
        seg[1::2] = np.roll(pts, -1, axis=0)
        rings.append(seg)
    return np.concatenate(rings)


def camera_frustum(cam, depth: float = 55.0) -> np.ndarray:
    """A camera's viewing pyramid, so coverage gaps are visible. Static."""
    w, h = cam.width, cam.height
    Kinv = np.linalg.inv(cam.K)
    rays = []
    for u, v in ((0, 0), (w, 0), (w, h), (0, h)):
        d = Kinv @ np.array([u, v, 1.0])
        rays.append(cam.C + (cam.R.T @ (d / np.linalg.norm(d))) * depth)
    rays = np.array(rays)

    out = np.empty((16, 3), dtype=np.float32)
    out[0::2][:4] = cam.C
    out[1::2][:4] = rays
    out[8::2] = rays
    out[9::2] = np.roll(rays, -1, axis=0)
    return out


def rays_from(origin, directions, length: float) -> np.ndarray:
    """Line segments radiating from a point -- radar beams."""
    directions = np.asarray(directions, dtype=np.float32).reshape(-1, 3)
    if len(directions) == 0:
        return EMPTY
    origin = np.asarray(origin, dtype=np.float32)
    out = np.empty((2 * len(directions), 3), dtype=np.float32)
    out[0::2] = origin
    out[1::2] = origin + directions * length
    return out


def beam_directions(az, el) -> np.ndarray:
    az = np.atleast_1d(np.asarray(az, dtype=np.float32))
    el = np.atleast_1d(np.asarray(el, dtype=np.float32))
    return np.stack([np.cos(el) * np.cos(az),
                     np.cos(el) * np.sin(az),
                     np.sin(el)], axis=1)
