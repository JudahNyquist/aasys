"""Measurement gating.

Gating asks whether a measurement is close enough to a track's prediction
to plausibly belong to it, in units of the filter's own uncertainty rather
than metres. That normalisation is what lets one gate serve both sensors: a
24 m radar fix at 2 km and a 0.4 m optical fix at 80 m are each "one sigma"
when judged against their own innovation covariance.

The squared Mahalanobis distance of a correct association is chi-squared
distributed with as many degrees of freedom as the measurement, so the
threshold comes straight from that distribution.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import chi2

_CACHE: dict[tuple[int, float], float] = {}


def gate_threshold(dim: int, prob: float = 0.997) -> float:
    """Chi-squared value containing `prob` of correct associations."""
    key = (int(dim), float(prob))
    if key not in _CACHE:
        _CACHE[key] = float(chi2.ppf(prob, int(dim)))
    return _CACHE[key]


def mahalanobis2(nu: np.ndarray, S: np.ndarray) -> float:
    try:
        return float(nu @ np.linalg.solve(S, nu))
    except np.linalg.LinAlgError:
        return float("inf")


def in_gate(nu: np.ndarray, S: np.ndarray, prob: float = 0.997) -> bool:
    return mahalanobis2(nu, S) <= gate_threshold(len(nu), prob)
