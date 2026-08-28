"""Turn a carved volume into measurements.

Clustering runs on the *sparse* occupied set rather than by labelling the
full dense volume: occupancy is a few hundred voxels out of four million,
so a neighbour graph over those points costs microseconds where
`ndimage.label` over the whole grid would dominate the frame.

The measurement covariance is the interesting part. It is derived from the
spatial second moment of each blob, so when the camera geometry constrains
one direction poorly the hull is genuinely elongated along it and the
filter learns to distrust that axis on its own -- a GDOP-like effect
falling out of the reconstruction instead of being hand-tuned per scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from ..sensing.base import Measurement
from ..sensing.models import CartesianPositionModel
from .vote import CarveResult

# 26-connectivity: diagonal neighbours sit sqrt(3) voxel-widths apart.
_CONNECT_RADIUS = 1.75


@dataclass
class Blob:
    centroid: np.ndarray      # (3,) world position
    cov: np.ndarray           # (3,3) spatial covariance of the voxel cloud
    n_voxels: int
    volume: float             # m^3
    extent: np.ndarray        # (3,) principal-axis standard deviations

    @property
    def equivalent_radius(self) -> float:
        return float((3.0 * self.volume / (4.0 * np.pi)) ** (1.0 / 3.0))


def extract_blobs(res: CarveResult, min_voxels: int = 2) -> list[Blob]:
    """Group occupied voxels into connected components."""
    centers = res.centers()
    n = centers.shape[0]
    if n == 0:
        return []

    if n == 1:
        labels, n_comp = np.zeros(1, dtype=int), 1
    else:
        tree = cKDTree(centers)
        pairs = tree.query_pairs(r=_CONNECT_RADIUS * res.grid.size,
                                 output_type="ndarray")
        if pairs.size == 0:
            labels, n_comp = np.arange(n), n
        else:
            g = coo_matrix(
                (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                shape=(n, n))
            n_comp, labels = connected_components(g, directed=False)

    voxel_vol = res.grid.voxel_volume
    blobs: list[Blob] = []
    for k in range(n_comp):
        pts = centers[labels == k]
        if len(pts) < min_voxels:
            continue
        centroid = pts.mean(axis=0)
        if len(pts) >= 2:
            cov = np.cov(pts.T)
            cov = np.atleast_2d(cov)
        else:
            cov = np.zeros((3, 3))
        # Quantisation floor: a single voxel still has finite extent.
        cov = cov + np.eye(3) * (res.grid.size ** 2 / 12.0)
        evals = np.linalg.eigvalsh(cov)
        blobs.append(Blob(
            centroid=centroid,
            cov=cov,
            n_voxels=len(pts),
            volume=len(pts) * voxel_vol,
            extent=np.sqrt(np.maximum(evals, 0.0)),
        ))

    blobs.sort(key=lambda b: -b.n_voxels)
    return blobs


def blobs_to_measurements(blobs: list[Blob], t: float,
                          sensor_id: str = "optical",
                          scale: float = 0.25,
                          floor_m: float = 0.35) -> list[Measurement]:
    """Convert blobs into Cartesian position measurements.

    `R` is the blob's spatial covariance scaled down: the second moment
    describes how large the hull *is*, while what the filter needs is how
    uncertain its *centroid* is, which is smaller. The voxels are not
    independent samples -- they form one solid body -- so the textbook
    standard-error-of-the-mean division by N is far too optimistic and
    would make the filter wildly overconfident.

    `scale` is therefore an empirical shrink factor, calibrated by driving
    NIS/NEES to consistency in `tools/batch_eval.py` rather than guessed.
    `floor_m` keeps R positive-definite for single-voxel blobs.
    """
    out = []
    for b in blobs:
        R = scale * b.cov + np.eye(3) * floor_m ** 2
        out.append(Measurement(
            t=t,
            z=b.centroid.copy(),
            R=R,
            model=CartesianPositionModel(),
            sensor_id=sensor_id,
            meta={"position": b.centroid.copy(),
                  "n_voxels": b.n_voxels,
                  "volume": b.volume,
                  "extent": b.extent},
        ))
    return out
