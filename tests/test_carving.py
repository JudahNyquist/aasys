"""Voxel carving tests, including regressions for two real bugs."""

import unittest

import numpy as np

from aasys.carving.cluster import extract_blobs
from aasys.carving.grid import VoxelGrid
from aasys.carving.lut import CarvingLUT
from aasys.carving.vote import Carver
from aasys.sensing.rig import CameraRig
from aasys.sensing.silhouette import rasterize


def small_setup(voxel=0.5, n=6):
    grid = VoxelGrid(lo=[-20, -20, 10], hi=[20, 20, 50], size=voxel)
    rig = CameraRig.ring_staggered(n=n, radius=60.0, fov_deg=50.0,
                                   width=960, height_px=720,
                                   aim=np.array([0.0, 0.0, 30.0]),
                                   heights=(4.0, 16.0))
    lut = CarvingLUT(grid, rig)
    return grid, rig, lut


class TestLUT(unittest.TestCase):
    def test_lut_matches_direct_projection(self):
        grid, rig, lut = small_setup()
        rng = np.random.default_rng(0)
        idx = rng.integers(0, grid.count, 200)
        centers = grid.centers_chunk(0, grid.count)[idx]
        for c, cam in enumerate(rig):
            uv, _, valid = cam.project(centers)
            for k, i in enumerate(idx):
                self.assertEqual(bool(lut.vis[c, i]), bool(valid[k]))
                if valid[k]:
                    expect = int(uv[k, 1]) * cam.width + int(uv[k, 0])
                    self.assertEqual(int(lut.pix[c, i]), expect)

    def test_threshold_capped_by_visibility(self):
        """Regression: thresholding against a flat k makes voxels outside
        some camera's FOV permanently unreachable, carving dead zones into
        the edges of the working volume."""
        grid, rig, lut = small_setup()
        thr = lut.threshold(len(rig))
        self.assertTrue(np.all(thr <= lut.n_visible))
        partial = lut.n_visible < len(rig)
        if partial.any():
            self.assertTrue(np.all(thr[partial] == lut.n_visible[partial]),
                            "voxels seen by only some cameras must still be "
                            "reachable")

    def test_invalid_indices_are_in_range(self):
        """Gathers read every entry, including ones later masked out, so no
        index may point outside the image."""
        grid, rig, lut = small_setup()
        for c, cam in enumerate(rig):
            self.assertTrue(np.all(lut.pix[c] >= 0))
            self.assertTrue(np.all(lut.pix[c] < cam.width * cam.height))


class TestCarving(unittest.TestCase):
    def test_sphere_centroid_within_one_voxel(self):
        grid, rig, lut = small_setup()
        carver = Carver(lut, vote_threshold=len(rig), min_cameras=3)
        truth = np.array([4.0, -3.0, 30.0])
        masks = [rasterize(cam, truth[None, :], np.array([1.2])) for cam in rig]
        res = carver.carve(masks)
        self.assertGreater(len(res), 0)
        err = np.linalg.norm(res.centers().mean(axis=0) - truth)
        self.assertLessEqual(err, grid.size)

    def test_sparse_and_dense_agree(self):
        """The fast path is an optimisation, not a different answer."""
        grid, rig, lut = small_setup()
        carver = Carver(lut, vote_threshold=len(rig), min_cameras=3)
        rng = np.random.default_rng(3)
        pos = np.array([[2.0, 5.0, 28.0], [-8.0, -6.0, 34.0]])
        masks = [rasterize(cam, pos, np.array([1.0, 1.0])) for cam in rig]
        masks = [m | (rng.random(m.shape) < 2e-5) for m in masks]
        flats = [np.ascontiguousarray(m).ravel() for m in masks]
        a = np.sort(carver._carve_sparse(masks, flats).indices)
        b = np.sort(carver._carve_dense(flats).indices)
        np.testing.assert_array_equal(a, b)

    def test_two_targets_give_two_blobs(self):
        grid, rig, lut = small_setup()
        carver = Carver(lut, vote_threshold=len(rig), min_cameras=3)
        pos = np.array([[12.0, 12.0, 30.0], [-12.0, -12.0, 30.0]])
        masks = [rasterize(cam, pos, np.array([1.0, 1.0])) for cam in rig]
        blobs = extract_blobs(carver.carve(masks), min_voxels=1)
        self.assertEqual(len(blobs), 2)

    def test_empty_scene_carves_nothing(self):
        grid, rig, lut = small_setup()
        carver = Carver(lut, vote_threshold=len(rig), min_cameras=3)
        masks = [np.zeros((c.height, c.width), dtype=bool) for c in rig]
        self.assertEqual(len(carver.carve(masks)), 0)


class TestResolutionRule(unittest.TestCase):
    def test_rule_flags_too_coarse_grid(self):
        """Carving is blind to targets smaller than a voxel's footprint;
        the check must say so rather than silently returning nothing."""
        coarse = VoxelGrid(lo=[-20, -20, 10], hi=[20, 20, 50], size=2.0)
        self.assertFalse(coarse.resolution_check(0.9)["ok"])
        fine = VoxelGrid(lo=[-20, -20, 10], hi=[20, 20, 50], size=0.5)
        self.assertTrue(fine.resolution_check(0.9)["ok"])

    def test_too_coarse_grid_really_carves_nothing(self):
        grid = VoxelGrid(lo=[-20, -20, 10], hi=[20, 20, 50], size=2.0)
        rig = CameraRig.ring_staggered(n=6, radius=60.0, fov_deg=50.0,
                                       width=960, height_px=720,
                                       aim=np.array([0.0, 0.0, 30.0]))
        carver = Carver(CarvingLUT(grid, rig), vote_threshold=6, min_cameras=3)
        truth = np.array([[3.0, 3.0, 30.0]])
        masks = [rasterize(cam, truth, np.array([0.45])) for cam in rig]
        self.assertEqual(len(carver.carve(masks)), 0,
                         "grid coarser than the rule should carve nothing, "
                         "which is what the rule exists to predict")


if __name__ == "__main__":
    unittest.main()
