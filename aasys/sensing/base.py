"""Common sensor interface.

A `Measurement` is self-describing: it carries its own measurement model,
so a single filter ingests Cartesian voxel-blob centroids and nonlinear
spherical-Doppler radar returns without any special-casing at the fusion
point. Adding a new sensor means adding a model, not editing the filter.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np

from .models import MeasurementModel


@dataclass
class Measurement:
    t: float                      # timestamp (s) -- fusion orders by this
    z: np.ndarray                 # measurement vector
    R: np.ndarray                 # measurement covariance
    model: MeasurementModel       # how z relates to the state
    sensor_id: str = ""
    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return int(self.z.shape[0])

    def position_hint(self) -> np.ndarray | None:
        """Cartesian position implied by this measurement, if cheap.

        Used only to seed brand-new tracks; the filter itself always works
        through the model rather than this shortcut.
        """
        return self.meta.get("position")


class Sensor(ABC):
    """Produces measurements from ground truth, never exposing it."""

    sensor_id: str

    @abstractmethod
    def sense(self, t: float, targets: list) -> list[Measurement]:
        """Observe `targets` at time `t` and return measurements."""
