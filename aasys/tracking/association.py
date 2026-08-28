"""Data association.

With several targets, clutter, and phantom volumes all producing
measurements at once, the question "which measurement belongs to which
track" has to be answered before any filter update. Answering it greedily
(nearest first) is fast and wrong in the case that matters: two tracks
crossing, where the greedy choice locks the first track onto the other's
measurement and drags both off course.

Global nearest neighbour instead minimises total assignment cost across all
pairs simultaneously, via the Hungarian algorithm. The cost is negative
log-likelihood rather than raw distance, so a measurement is judged against
each track's own uncertainty -- which is what makes assignments comparable
between a coarse radar fix and a precise optical one.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

from .gating import gate_threshold, mahalanobis2

# Cost assigned to a forbidden (out-of-gate) pairing.
BIG = 1e6


def build_cost_matrix(tracks, measurements, gate_prob: float = 0.997):
    """Negative log-likelihood cost, with out-of-gate pairs blocked."""
    n_t, n_m = len(tracks), len(measurements)
    cost = np.full((n_t, n_m), BIG)
    gated = np.zeros((n_t, n_m), dtype=bool)

    for i, trk in enumerate(tracks):
        for j, meas in enumerate(measurements):
            nu, S, _ = trk.filter.innovation(meas)
            d2 = mahalanobis2(nu, S)
            if d2 > gate_threshold(len(nu), gate_prob):
                continue
            try:
                det = float(np.linalg.det(S))
            except np.linalg.LinAlgError:
                continue
            if det <= 0:
                continue
            # -2 log N(nu; 0, S), dropping constants common to all pairs.
            cost[i, j] = d2 + np.log(det)
            gated[i, j] = True

    return cost, gated


def associate(tracks, measurements, gate_prob: float = 0.997):
    """Return (matches, unmatched_tracks, unmatched_measurements)."""
    if not tracks or not measurements:
        return [], list(range(len(tracks))), list(range(len(measurements)))

    cost, gated = build_cost_matrix(tracks, measurements, gate_prob)
    rows, cols = linear_sum_assignment(cost)

    matches = []
    used_t, used_m = set(), set()
    for i, j in zip(rows, cols):
        # The solver fills a rectangular matrix and will happily return
        # blocked pairs; drop anything that was never in gate.
        if not gated[i, j]:
            continue
        matches.append((int(i), int(j)))
        used_t.add(int(i))
        used_m.add(int(j))

    unmatched_t = [i for i in range(len(tracks)) if i not in used_t]
    unmatched_m = [j for j in range(len(measurements)) if j not in used_m]
    return matches, unmatched_t, unmatched_m
