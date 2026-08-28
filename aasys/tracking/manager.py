"""Track management: the loop that turns measurements into tracks.

Measurements arrive from sensors that neither share a clock rate nor agree
on what they are measuring. The manager processes them strictly in
timestamp order, advancing every track's prediction to each measurement's
own time before associating. Skipping that and treating a batch as
simultaneous is a quiet source of bias whenever sensors interleave, which
here they always do.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..sensing.base import Measurement
from .association import associate
from .gating import gate_threshold, mahalanobis2
from .track import Track, TrackState


class TrackManager:
    def __init__(self, filter_factory, confirm_hits: int = 3,
                 confirm_window: int = 5, max_coast_s: float = 2.5,
                 max_consecutive_misses: int = 8, gate_prob: float = 0.997,
                 max_position_sigma: float = 120.0,
                 init_speed_sigma: float = 35.0,
                 tentative_timeout_s: float = 2.5,
                 suppress_gate_prob: float = 0.99999) -> None:
        """`filter_factory(position, R)` builds the estimator for a new track.

        `max_position_sigma` deletes tracks whose uncertainty has grown past
        any use during a coast, which is the honest way to give up rather
        than carrying a track that is really just a prediction.
        """
        self.filter_factory = filter_factory
        self.confirm_hits = confirm_hits
        self.confirm_window = confirm_window
        self.max_coast_s = max_coast_s
        self.max_consecutive_misses = max_consecutive_misses
        self.gate_prob = gate_prob
        self.max_position_sigma = max_position_sigma
        self.init_speed_sigma = init_speed_sigma
        self.tentative_timeout_s = tentative_timeout_s
        self.suppress_gate_prob = suppress_gate_prob

        self.tracks: list[Track] = []
        self.deleted: list[Track] = []
        self._t = 0.0
        self.stats = defaultdict(int)

    @property
    def confirmed(self) -> list[Track]:
        return [t for t in self.tracks if t.confirmed]

    # ------------------------------------------------------------ lifecycle
    def _initiate(self, meas: Measurement) -> Track | None:
        pos = meas.position_hint()
        if pos is None:
            return None
        filt = self.filter_factory(np.asarray(pos, dtype=float), meas.R)
        trk = Track(filt, meas.t, meas.sensor_id)
        self.tracks.append(trk)
        self.stats["initiated"] += 1
        return trk

    def _suppressed(self, meas: Measurement) -> bool:
        """Should this leftover measurement be denied a new track?

        Global nearest neighbour assigns at most one measurement per track,
        so a second look at an already-tracked target in the same frame --
        routine when a search dwell and a track dwell overlap -- falls out
        unmatched and would otherwise spawn a duplicate. Anything sitting
        well inside an existing track's uncertainty is treated as a
        redundant view of it rather than as a new object.
        """
        for trk in self.tracks:
            nu, S, _ = trk.filter.innovation(meas)
            if mahalanobis2(nu, S) <= gate_threshold(len(nu),
                                                     self.suppress_gate_prob):
                return True
        return False

    def _promote_and_prune(self, t: float) -> None:
        alive = []
        for trk in self.tracks:
            if (trk.state is TrackState.TENTATIVE
                    and sum(trk.window) >= self.confirm_hits):
                trk.state = TrackState.CONFIRMED
                self.stats["confirmed"] += 1

            stale = trk.age_since_update(t)
            if trk.state is TrackState.CONFIRMED and stale > 1e-9:
                trk.state = TrackState.COASTING
            elif trk.state is TrackState.COASTING and stale <= 1e-9:
                trk.state = TrackState.CONFIRMED

            drop = False
            if trk.state is TrackState.TENTATIVE:
                # An unconfirmed track gets a short leash; this is where
                # optical phantoms and radar clutter die.
                #
                # The leash is measured in *time*, not in missed
                # associations. A miss is only meaningful if the sensor
                # actually looked: with a scanning array most frames simply
                # never illuminate this patch of sky, and counting those as
                # evidence against the track would delete every real target
                # before its second look.
                if trk.age_since_update(t) > self.tentative_timeout_s:
                    drop = True
            else:
                if stale > self.max_coast_s:
                    drop = True
                if trk.consecutive_misses >= self.max_consecutive_misses:
                    drop = True
            if trk.position_sigma > self.max_position_sigma:
                drop = True

            if drop:
                trk.state = TrackState.DELETED
                self.deleted.append(trk)
                self.stats["deleted"] += 1
            else:
                alive.append(trk)
        self.tracks = alive

    # ----------------------------------------------------------------- step
    def step(self, t: float, measurements: list[Measurement]) -> None:
        """Advance to time `t`, folding in any measurements at or before it."""
        groups: dict[float, list[Measurement]] = defaultdict(list)
        for m in measurements:
            # Quantise to a microsecond so a genuine scan lands in one group.
            groups[round(m.t, 6)].append(m)

        for tm in sorted(groups):
            self._process_group(tm, groups[tm])

        # Bring every track forward to the frame time, then age the roster.
        for trk in self.tracks:
            trk.predict_to(t)
        self._promote_and_prune(t)
        self._t = t
        for trk in self.tracks:
            trk.record(t)

    def _process_group(self, t: float, measurements: list[Measurement]) -> None:
        for trk in self.tracks:
            trk.predict_to(t)

        matches, unmatched_t, unmatched_m = associate(
            self.tracks, measurements, self.gate_prob)

        for i, j in matches:
            self.tracks[i].register_hit(measurements[j], self.confirm_window)
            self.stats["hits"] += 1

        for i in unmatched_t:
            self.tracks[i].register_miss(self.confirm_window)
            self.stats["misses"] += 1

        for j in unmatched_m:
            if self._suppressed(measurements[j]):
                self.stats["suppressed"] += 1
                continue
            self._initiate(measurements[j])

    # ---------------------------------------------------------------- report
    def summary(self) -> str:
        by_state = defaultdict(int)
        for t in self.tracks:
            by_state[t.state.value] += 1
        parts = [f"{k}={v}" for k, v in sorted(by_state.items())]
        return (f"tracks[{', '.join(parts) or 'none'}] "
                f"init={self.stats['initiated']} conf={self.stats['confirmed']} "
                f"del={self.stats['deleted']}")
