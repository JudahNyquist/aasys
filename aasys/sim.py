"""Simulation orchestrator.

Wires the pieces into one loop and keeps the rates separate, which matters
more than it looks: physics integrates at a small fixed step so trajectories
never depend on the frame rate, sensors report at their own cadences, and
the renderer just reads whatever the latest state happens to be. Anything
that couples those together makes results depend on how fast the machine
drew the last frame.
"""

from __future__ import annotations

import numpy as np

from .core.rng import RngHub
from .fire_control.engagement import EngagementManager
from .sensing.optical import OpticalCarvingSensor
from .sensing.radar import PhasedArrayRadar
from .tracking.manager import TrackManager
from .tracking.track import TrackState


class Simulation:
    def __init__(self, scenario, seed: int = 0) -> None:
        self.scenario = scenario
        self.seed = seed
        self.rng = RngHub(seed)
        self.t = 0.0
        self.dt = scenario.get("dt", 1.0 / 120.0)

        self.targets = scenario["targets"]
        self.asset_pos = np.asarray(scenario.get("asset", [0.0, 0.0, 0.0]),
                                    dtype=float)

        self.radar: PhasedArrayRadar | None = None
        if scenario.get("radar", True):
            self.radar = PhasedArrayRadar(
                rng=self.rng.stream("radar"),
                **scenario.get("radar_params", {}))

        self.optical: OpticalCarvingSensor | None = None
        if scenario.get("optical", True):
            self.optical = OpticalCarvingSensor(
                scenario["rig"], scenario["grid"],
                rng=self.rng.stream("optical"),
                **scenario.get("optical_params", {}))

        self.tracker = TrackManager(scenario["filter_factory"],
                                    **scenario.get("tracker_params", {}))
        self.engagement = EngagementManager(
            asset_pos=self.asset_pos, rng=self.rng.stream("engage"),
            **scenario.get("engagement_params", {}))

        self.events: list[str] = []
        self.measurements: list = []      # most recent frame, for rendering
        self.threats: list = []
        self.engage_enabled = scenario.get("engage", True)
        self._history = {"t": [], "n_tracks": [], "n_conf": []}

    # ----------------------------------------------------------------- step
    def step(self, dt: float | None = None) -> None:
        dt = self.dt if dt is None else dt
        t = self.t

        for g in self.targets:
            g.step(t, dt)

        meas = []
        if self.radar is not None:
            meas += self.radar.sense(t, self.targets)
        if self.optical is not None:
            meas += self.optical.sense(t, self.targets)
        self.measurements = meas

        self.tracker.step(t, meas)

        # Close the AESA loop. Without this cue the array would only ever find
        # a target by chance during its search raster.
        if self.radar is not None:
            live = set()
            for trk in self.tracker.tracks:
                live.add(trk.id)
                priority, revisit = self._revisit_for(trk)
                self.radar.request_track(trk.id, trk.position,
                                         priority=priority, revisit_s=revisit)
            for tid in [k for k in self.radar.track_requests if k not in live]:
                self.radar.drop_track(tid)

        if self.engage_enabled:
            self.threats = self.engagement.decide(t, self.tracker.tracks)
            ev = self.engagement.step_effectors(t, dt, self.tracker.tracks,
                                                self.targets)
            if ev:
                self.events.extend(ev)

        self._history["t"].append(t)
        self._history["n_tracks"].append(len(self.tracker.tracks))
        self._history["n_conf"].append(len(self.tracker.confirmed))
        self.t += dt

    # Revisit policy ------------------------------------------------------
    # How tight a confirmed track has to stay to be worth shooting on. The
    # gun refuses a track above 2 m and the missile above 20 m, so budgeting
    # against something in between keeps the array working to keep tracks
    # inside the envelope that matters.
    SIGMA_BUDGET_M = 8.0
    REVISIT_RELAXED_S = 0.5
    REVISIT_URGENT_S = 0.10

    def _revisit_for(self, trk) -> tuple[float, float]:
        """How hard should the array work to keep this track?

        A fixed revisit for every confirmed track spends the same beam time
        on a drone flying straight and level as on one breaking hard, which
        gets it wrong in both directions: it wastes dwells that search needs,
        and it still loses the manoeuvring target. Two signals the tracker
        already computes say which case this is.

        `position_sigma` is how bad the estimate has got -- it grows through
        every coast, so a track the array is losing asks to be looked at
        sooner. The IMM's model probabilities say whether the target is
        holding a heading: mass away from the constant-velocity model means
        it is turning or accelerating, so the prediction will stale before
        the next look regardless of how good the estimate is right now.

        Taking the worse of the two, rather than blending, means either
        reason alone is enough to earn a faster revisit.
        """
        if trk.state is TrackState.TENTATIVE:
            # An unconfirmed track has a short leash and no useful covariance
            # yet; confirm it or drop it quickly.
            return 3.0, 0.12

        load = min(trk.position_sigma / self.SIGMA_BUDGET_M, 1.0)
        manoeuvre = 1.0 - float(trk.model_probabilities().get("CV", 1.0))
        urgency = float(np.clip(max(load, manoeuvre), 0.0, 1.0))

        revisit = (self.REVISIT_RELAXED_S
                   + (self.REVISIT_URGENT_S - self.REVISIT_RELAXED_S) * urgency)
        return 1.0 + 2.0 * urgency, revisit

    def run(self, duration: float, on_step=None) -> None:
        n = int(duration / self.dt)
        for _ in range(n):
            self.step()
            if on_step is not None:
                on_step(self)

    # -------------------------------------------------------------- reporting
    def truth_for_track(self, trk):
        """Nearest true target to a track -- for scoring only, never for
        anything the system itself is allowed to see."""
        live = [g for g in self.targets if g.alive]
        if not live:
            return None
        d = [float(np.linalg.norm(g.position - trk.position)) for g in live]
        i = int(np.argmin(d))
        return live[i], d[i]

    def report(self) -> str:
        lines = [f"t = {self.t:.1f} s", self.tracker.summary()]
        if self.radar is not None:
            s = self.radar.stats
            lines.append(
                f"radar: det={s['detections']} search={s['search_dwells']} "
                f"track={s['track_dwells']} mti_rej={s['mti_rejected']} "
                f"fa={s['false_alarms']} frame={self.radar.search_frame_s:.2f}s")
        if self.optical is not None:
            s = self.optical.stats
            f = max(s["frames"], 1)
            lines.append(f"optical: frames={s['frames']} "
                         f"blobs/frame={s['blobs']/f:.2f} "
                         f"voxels/frame={s['voxels']/f:.1f}")
        e = self.engagement.stats
        lines.append(f"engagement: bursts={e['bursts']} rounds={e['rounds']} "
                     f"missiles={e['missiles']} uavs={e['uavs']} "
                     f"kills={e['kills']} seeker_losses={e['seeker_losses']} "
                     f"declined(uncertain)={e['declined_uncertain']}")
        alive = sum(1 for g in self.targets if not g.destroyed)
        lines.append(f"targets: {alive}/{len(self.targets)} still flying")
        return "\n".join(lines)
