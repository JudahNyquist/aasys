"""Per-frame recording of what the estimator believed and what was true.

A headless run currently prints a dozen summary numbers and throws away
everything that produced them. That is enough to notice a regression and
useless for diagnosing one: RMSE says the tracker is worse, not that it lost
lock for 1.4 s during a turn while the IMM sat on the wrong model.

What is captured is deliberately the *paired* series -- estimate alongside
truth, at the same instant, with the error and its claimed covariance -- so
that consistency (NEES) can be recomputed offline rather than trusted from a
single scalar at the end.

Truth access here is legitimate for the same reason it is in `lethality`:
this is scoring, it runs after the fact, and nothing it produces is visible
to the tracker or to fire control.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

#: A track further than this from any live target is scored as unmatched
#: rather than being blamed on the nearest one.
MATCH_GATE_M = 25.0

TRACK_FIELDS = (
    "t", "track_id", "state", "sensor",
    "x", "y", "z", "vx", "vy", "vz",
    "sigma", "nis", "err", "nees", "target_id", "p_manoeuvre", "wreck",
)
TRUTH_FIELDS = ("t", "target_id", "x", "y", "z", "vx", "vy", "vz", "destroyed")

_STATES = ("tentative", "confirmed", "coasting", "deleted")


class Recorder:
    """Captures one row per (frame, track) and per (frame, target).

    Long format rather than one array per track: track counts change every
    frame as tracks are born, coast and die, and padding a rectangular array
    to the maximum would store mostly holes and lose exactly the lifecycle
    detail worth keeping.
    """

    def __init__(self, match_gate_m: float = MATCH_GATE_M) -> None:
        self.match_gate = float(match_gate_m)
        self.tracks: list[tuple] = []
        self.truth: list[tuple] = []
        self.model_names: tuple[str, ...] = ()
        self.model_probs: list[tuple] = []      # (t, track_id, *probabilities)
        self.events: list[str] = []

    # ------------------------------------------------------------- capture
    def capture(self, sim) -> None:
        """Record one frame. Safe to pass directly as `Simulation.run`'s
        `on_step` callback."""
        t = sim.t
        live = [g for g in sim.targets if g.alive]

        for g in sim.targets:
            self.truth.append((t, g.id, *g.position, *g.velocity,
                               float(g.destroyed)))

        if live:
            truth_pos = np.array([g.position for g in live])

        for trk in sim.tracker.tracks:
            tid, err, nees, wreck = -1, np.nan, np.nan, 0.0
            if live:
                d = np.linalg.norm(truth_pos - trk.position, axis=1)
                i = int(np.argmin(d))
                if d[i] <= self.match_gate:
                    tid = live[i].id
                    # A destroyed target is unpowered wreckage tumbling
                    # ballistically. The tracker is still following
                    # something real, but scoring that alongside a live
                    # manoeuvring drone conflates two different claims.
                    wreck = float(live[i].destroyed)
                    e = np.concatenate([live[i].position - trk.position,
                                        live[i].velocity - trk.velocity])
                    err = float(np.linalg.norm(e[:3]))
                    nees = _nees(e, trk.P[:6, :6])

            probs = trk.model_probabilities()
            if not self.model_names:
                self.model_names = tuple(probs)
            p_man = 1.0 - float(probs.get("CV", 1.0))

            self.tracks.append((
                t, trk.id, _STATES.index(trk.state.value), trk.last_sensor,
                *trk.position, *trk.velocity,
                trk.position_sigma,
                _finite(trk.filter.last_nis), err, nees, tid, p_man,
                wreck))
            self.model_probs.append(
                (t, trk.id, *(probs.get(n, np.nan) for n in self.model_names)))

        if len(sim.events) > len(self.events):
            self.events = list(sim.events)

    # -------------------------------------------------------------- output
    def save(self, path) -> Path:
        """Write `<path>.npz` plus `<path>.tracks.csv` and `.truth.csv`.

        `.npz` because numpy is already a dependency and it round-trips
        exactly; CSV alongside it because the point of recording a run is
        often to look at it in something that is not Python.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stem = path.with_suffix("")

        tracks = np.array([r[:3] + r[4:] for r in self.tracks], dtype=float) \
            if self.tracks else np.empty((0, len(TRACK_FIELDS) - 1))
        sensors = np.array([r[3] for r in self.tracks], dtype=object)

        np.savez_compressed(
            stem.with_suffix(".npz"),
            tracks=tracks,
            track_fields=np.array([f for f in TRACK_FIELDS if f != "sensor"]),
            sensors=sensors.astype(str),
            truth=np.array(self.truth, dtype=float) if self.truth
            else np.empty((0, len(TRUTH_FIELDS))),
            truth_fields=np.array(TRUTH_FIELDS),
            model_names=np.array(self.model_names),
            model_probs=(np.array(self.model_probs, dtype=float)
                         if self.model_probs else np.empty((0, 2))),
            events=np.array(self.events, dtype=str),
        )

        _write_csv(stem.with_suffix(".tracks.csv"), TRACK_FIELDS, self.tracks)
        _write_csv(stem.with_suffix(".truth.csv"), TRUTH_FIELDS, self.truth)
        return stem.with_suffix(".npz")

    # ------------------------------------------------------------- summary
    def summary(self) -> str:
        """Confirmed-track accuracy, live targets and wreckage reported apart.

        Once a target is shot down it becomes an unpowered, tumbling wreck
        that no manoeuvre model describes, so the filter's error and its NEES
        both blow up -- correctly. Averaging those frames in with live
        tracking makes a battery that works look like a tracker that does
        not, and hides a real regression behind the noise of its own
        successes.
        """
        rows = [r for r in self.tracks
                if _STATES[int(r[2])] in ("confirmed", "coasting")]
        if not rows:
            return "no confirmed-track frames"

        out = [f"recorded {len(self.tracks)} track-frames over "
               f"{len(set(r[1] for r in self.tracks))} tracks"]
        for label, want_wreck in (("live target", 0.0), ("wreckage", 1.0)):
            sub = [r for r in rows if r[16] == want_wreck]
            line = _score_line(label, sub)
            if line:
                out.append(line)
        return "\n".join(out)


def _score_line(label: str, rows) -> str:
    err = np.array([r[12] for r in rows], dtype=float)
    err = err[np.isfinite(err)]
    if err.size == 0:
        return ""
    nees = np.array([r[13] for r in rows], dtype=float)
    nees = nees[np.isfinite(nees)]
    return (f"  {label:<12s} n={err.size:5d}  "
            f"RMSE={np.sqrt(np.mean(err ** 2)):6.2f} m  "
            f"median={np.median(err):5.2f} m  "
            f"p95={np.percentile(err, 95):6.2f} m  "
            f"NEES={np.mean(nees) if nees.size else float('nan'):6.2f}")


def _finite(v) -> float:
    return float(v) if v is not None and np.isfinite(v) else np.nan


def _nees(e: np.ndarray, P: np.ndarray) -> float:
    try:
        return float(e @ np.linalg.solve(P, e))
    except np.linalg.LinAlgError:
        return np.nan


def _write_csv(path: Path, fields, rows) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(fields)
        w.writerows(rows)
