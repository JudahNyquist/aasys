"""Phased-array (AESA) radar.

A rotating dish gives one detection per revolution -- fine for knowing
something is out there, far too slow and coarse to shoot with. Real systems
solve this with an electronically steered beam that interleaves two jobs:

  * **search** -- a raster over the surveillance volume, finding new targets
  * **track**  -- dedicated dwells revisiting known targets at a much higher
                  rate, with the revisit interval shortened for targets that
                  are manoeuvring or threatening

Because the array has one beam and a finite time budget, those two jobs
compete. That scheduling pressure is modelled here rather than assumed
away: adding targets genuinely steals time from search, and the operator
sees search frame time stretch as the raid grows.

Three physical effects dominate performance against small drones:

  * **Radar equation** -- SNR falls as R^-4, and a quadcopter's RCS is
    ~0.01-0.05 m^2, so detection range against one is far shorter than the
    same radar's range against an aircraft.
  * **Monopulse** -- angle accuracy is *not* the beamwidth. Comparing sum
    and difference channels gives roughly beamwidth/(k*sqrt(2*SNR)), so
    accuracy improves as the target closes.
  * **MTI clutter notch** -- returns near zero Doppler are indistinguishable
    from ground clutter and get filtered out. A hovering or tangentially
    crossing drone is therefore invisible to radar no matter how strong its
    echo, which is precisely the gap the optical channel exists to fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..core.vecmath import enu_to_spherical, spherical_to_enu, wrap_pi
from .base import Measurement, Sensor
from .models import SphericalDopplerModel, SphericalModel

# SNR (linear) at which a target sits exactly at its 50% detection range.
SNR_AT_R50 = 20.0          # ~13 dB
MONOPULSE_K = 1.6


@dataclass
class TrackRequest:
    """Fire control asking the radar to keep looking at something."""
    track_id: int
    position: np.ndarray
    priority: float = 1.0
    revisit_s: float = 0.4
    next_due: float = 0.0


@dataclass
class Dwell:
    t: float
    az: float
    el: float
    kind: str                  # "search" | "track"
    width: float = 0.0         # beamwidth used for this look (rad)
    track_id: int | None = None


class PhasedArrayRadar(Sensor):
    def __init__(self, position=(0.0, 0.0, 3.0), sensor_id: str = "radar",
                 beamwidth_deg: float = 2.5,
                 search_beamwidth_deg: float = 7.0,
                 search_max_elevation_deg: float = 30.0,
                 dwell_s: float = 5e-3,
                 duty: float = 0.85,
                 r50_ref_m: float = 1200.0,
                 rcs_ref: float = 0.05,
                 max_range_m: float = 4000.0,
                 range_res_m: float = 15.0,
                 doppler_res_ms: float = 1.2,
                 min_elevation_deg: float = 2.0,
                 max_elevation_deg: float = 75.0,
                 mti_notch_ms: float = 2.0,
                 clutter_rate_hz: float = 0.4,
                 use_doppler: bool = True,
                 rng=None) -> None:
        self.position = np.asarray(position, dtype=float)
        self.sensor_id = sensor_id
        self.beamwidth = np.radians(beamwidth_deg)
        # Search deliberately runs a broadened beam. Covering the volume with
        # the narrow track beam would take tens of seconds per frame, which
        # is how a real array would be configured only if it never expected
        # to find anything.
        self.search_beamwidth = np.radians(search_beamwidth_deg)
        self.search_max_el = np.radians(search_max_elevation_deg)
        self.dwell_s = float(dwell_s)
        self.duty = float(duty)
        self.r50_ref = float(r50_ref_m)
        self.rcs_ref = float(rcs_ref)
        self.max_range = float(max_range_m)
        self.range_res = float(range_res_m)
        self.doppler_res = float(doppler_res_ms)
        self.min_el = np.radians(min_elevation_deg)
        self.max_el = np.radians(max_elevation_deg)
        self.mti_notch = float(mti_notch_ms)
        self.clutter_rate = float(clutter_rate_hz)
        self.use_doppler = use_doppler
        self.rng = rng if rng is not None else np.random.default_rng(0)

        self._model_cls = SphericalDopplerModel if use_doppler else SphericalModel
        self.model = self._model_cls(self.position)

        self.track_requests: dict[int, TrackRequest] = {}
        self._t_last: float | None = None
        self._dwell_credit = 0.0
        self.last_dwells: list[Dwell] = []

        # Search raster: beam positions spaced one beamwidth apart.
        self._search_grid = self._build_search_grid()
        self._search_i = 0
        self._search_frame_start = 0.0
        self.search_frame_s = float("nan")
        self.stats = {"search_dwells": 0, "track_dwells": 0, "detections": 0,
                      "mti_rejected": 0, "false_alarms": 0}

    # -------------------------------------------------------------- geometry
    def _build_search_grid(self) -> np.ndarray:
        bw = self.search_beamwidth
        els = np.arange(self.min_el + bw / 2, self.search_max_el, bw)
        beams = []
        for el in els:
            # Rows near zenith need fewer beams to cover 360 degrees.
            n_az = max(4, int(np.ceil(2 * np.pi * np.cos(el) / bw)))
            for az in np.linspace(-np.pi, np.pi, n_az, endpoint=False):
                beams.append((az, el))
        return np.array(beams)

    @property
    def n_search_beams(self) -> int:
        return len(self._search_grid)

    def request_track(self, track_id: int, position, priority: float = 1.0,
                      revisit_s: float = 0.4) -> None:
        """Ask for recurring dwells on a known target.

        Fire control raises priority and shortens `revisit_s` for
        manoeuvring or high-threat tracks; the scheduler then spends more of
        the array's time on them and less on searching.
        """
        req = self.track_requests.get(track_id)
        if req is None:
            self.track_requests[track_id] = TrackRequest(
                track_id, np.asarray(position, dtype=float), priority,
                revisit_s, 0.0)
        else:
            req.position = np.asarray(position, dtype=float)
            req.priority = priority
            req.revisit_s = revisit_s

    def drop_track(self, track_id: int) -> None:
        self.track_requests.pop(track_id, None)

    # ------------------------------------------------------------- detection
    def _snr(self, rng_m: float, rcs: float) -> float:
        """Linear SNR from the radar equation, normalised so that a target
        of `rcs_ref` at `r50_ref` sits exactly at the detection threshold."""
        r50 = self.r50_ref * (max(rcs, 1e-6) / self.rcs_ref) ** 0.25
        if rng_m < 1e-6:
            return 1e9
        return SNR_AT_R50 * (r50 / rng_m) ** 4

    def _p_detect(self, snr: float) -> float:
        """Smooth detection curve, 0.5 exactly at threshold SNR."""
        if snr <= 0:
            return 0.0
        return float(0.5 ** (SNR_AT_R50 / snr))

    def _sigmas(self, snr: float, width: float | None = None
                ) -> tuple[float, float, float]:
        """Measurement sigmas for (range, angle, range-rate) at this SNR.

        All three sharpen as sqrt(SNR), which is why a closing target is
        tracked far more precisely than a distant one.
        """
        s = max(snr, 1e-3)
        root = np.sqrt(2.0 * s)
        sigma_ang = (width or self.beamwidth) / (MONOPULSE_K * root)
        sigma_rng = self.range_res / root
        sigma_dop = self.doppler_res / root
        # Floors: calibration and clock stability bound accuracy regardless.
        return (max(sigma_rng, 0.5),
                max(sigma_ang, np.radians(0.01)),
                max(sigma_dop, 0.05))

    def _observe(self, t: float, target, dwell: Dwell) -> Measurement | None:
        rel = target.position - self.position
        r, az, el = enu_to_spherical(rel)
        if r > self.max_range or el < self.min_el or el > self.max_el:
            return None

        # Inside the beam?
        width = dwell.width or self.beamwidth
        d_az = wrap_pi(az - dwell.az) * np.cos(el)
        d_el = wrap_pi(el - dwell.el)
        if np.hypot(d_az, d_el) > width / 2:
            return None

        # MTI: reject anything sitting in the zero-Doppler clutter notch.
        rdot = float(np.dot(rel, target.velocity) / max(r, 1e-9))
        if abs(rdot) < self.mti_notch:
            self.stats["mti_rejected"] += 1
            return None

        snr = self._snr(r, target.rcs)
        if width > self.beamwidth:
            # Spreading the same energy over a wider beam costs SNR as the
            # square of the broadening -- search sees less far than track.
            snr *= (self.beamwidth / width) ** 2
        if self.rng.random() > self._p_detect(snr):
            return None

        s_r, s_a, s_d = self._sigmas(snr, width)
        # Monopulse measures off-boresight angle; converting that to azimuth
        # divides by cos(elevation), so azimuth error grows toward zenith.
        # R has to state the same sigma that is injected here, or the filter
        # is handed a covariance it was never entitled to believe.
        s_az = s_a / max(np.cos(el), 0.1)
        z_r = r + self.rng.normal(0.0, s_r)
        z_az = wrap_pi(az + self.rng.normal(0.0, s_az))
        z_el = el + self.rng.normal(0.0, s_a)

        if self.use_doppler:
            z = np.array([z_r, z_az, z_el, rdot + self.rng.normal(0.0, s_d)])
            R = np.diag([s_r ** 2, s_az ** 2, s_a ** 2, s_d ** 2])
        else:
            z = np.array([z_r, z_az, z_el])
            R = np.diag([s_r ** 2, s_az ** 2, s_a ** 2])

        self.stats["detections"] += 1
        return Measurement(
            t=t, z=z, R=R, model=self.model, sensor_id=self.sensor_id,
            meta={"position": self.position + spherical_to_enu(z_r, z_az, z_el),
                  "snr_db": float(10 * np.log10(max(snr, 1e-9))),
                  "kind": dwell.kind, "truth_id": target.id},
        )

    def _false_alarm(self, t: float, dwell: Dwell) -> Measurement:
        """A clutter return, reported exactly the way a real one would be.

        The covariance has to be built the same way as in `_observe`, from
        the same beamwidth and with the same cos(elevation) term on azimuth.
        Quoting clutter a tighter azimuth than a genuine detection would let
        a false alarm win an association contest it should lose -- the
        tracker compares Mahalanobis distances, so an optimistic R is
        indistinguishable from a better measurement.
        """
        width = dwell.width or self.beamwidth
        r = float(self.rng.uniform(50.0, self.max_range))
        az = wrap_pi(dwell.az + self.rng.normal(0, width / 3))
        el = float(np.clip(dwell.el + self.rng.normal(0, width / 3),
                           self.min_el, self.max_el))
        s_r, s_a, s_d = self._sigmas(SNR_AT_R50, width)
        s_az = s_a / max(np.cos(el), 0.1)
        rdot = float(self.rng.normal(0.0, 12.0))
        if self.use_doppler:
            z = np.array([r, az, el, rdot])
            R = np.diag([s_r ** 2, s_az ** 2, s_a ** 2, s_d ** 2])
        else:
            z = np.array([r, az, el])
            R = np.diag([s_r ** 2, s_az ** 2, s_a ** 2])
        self.stats["false_alarms"] += 1
        return Measurement(
            t=t, z=z, R=R, model=self.model, sensor_id=self.sensor_id,
            meta={"position": self.position + spherical_to_enu(r, az, el),
                  "kind": "clutter", "truth_id": None},
        )

    # ------------------------------------------------------------ scheduling
    def _schedule(self, t: float, dt: float) -> list[Dwell]:
        """Allocate the array's time between track revisits and search.

        Track dwells that are due win; whatever is left over continues the
        search raster. This is where a growing raid visibly starves search.
        """
        self._dwell_credit += dt * self.duty / self.dwell_s
        n = int(self._dwell_credit)
        if n <= 0:
            return []
        self._dwell_credit -= n

        dwells: list[Dwell] = []
        due = [rq for rq in self.track_requests.values() if rq.next_due <= t]
        due.sort(key=lambda rq: (-rq.priority, rq.next_due))

        for rq in due:
            if len(dwells) >= n:
                break
            rel = rq.position - self.position
            _, az, el = enu_to_spherical(rel)
            dwells.append(Dwell(t, float(az), float(el), "track",
                                self.beamwidth, rq.track_id))
            rq.next_due = t + rq.revisit_s
            self.stats["track_dwells"] += 1

        while len(dwells) < n:
            az, el = self._search_grid[self._search_i]
            dwells.append(Dwell(t, float(az), float(el), "search",
                                self.search_beamwidth))
            self._search_i += 1
            self.stats["search_dwells"] += 1
            if self._search_i >= len(self._search_grid):
                self._search_i = 0
                self.search_frame_s = t - self._search_frame_start
                self._search_frame_start = t
        return dwells

    # ------------------------------------------------------------------- api
    def sense(self, t: float, targets: list) -> list[Measurement]:
        dt = 0.0 if self._t_last is None else max(t - self._t_last, 0.0)
        self._t_last = t

        dwells = self._schedule(t, dt)
        self.last_dwells = dwells
        if not dwells:
            return []

        out: list[Measurement] = []
        for dwell in dwells:
            for tgt in targets:
                if not tgt.alive:
                    continue
                m = self._observe(t, tgt, dwell)
                if m is not None:
                    out.append(m)

        expected_fa = self.clutter_rate * dt
        for _ in range(self.rng.poisson(expected_fa) if expected_fa > 0 else 0):
            out.append(self._false_alarm(t, dwells[self.rng.integers(len(dwells))]))
        return out

    def describe(self) -> str:
        return (f"AESA track {np.degrees(self.beamwidth):.1f}deg / "
                f"search {np.degrees(self.search_beamwidth):.1f}deg, "
                f"{self.n_search_beams} search positions, "
                f"{1/self.dwell_s:.0f} dwells/s, "
                f"R50={self.r50_ref:.0f}m @ {self.rcs_ref} m^2")
