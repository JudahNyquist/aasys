"""Threat evaluation and engagement doctrine.

Decides what to shoot, with what, and when. Every input is a track estimate,
so a mis-ranked threat or a premature shot traces back to the tracker, not
to the shooter.

The battery is layered and independent: a ring of automated CRAM machine-guns
for close-in work, two six-tube tracking-missile launchers for standoff reach,
and a small hangar of counter-drone UAVs that chase raiders down themselves.
Doctrine fires the missiles first -- an interceptor corrects in flight, so it
is the primary weapon -- while the guns engage the same threats at the same
time and keep covering anything the missiles cannot reach. A counter-drone is
launched only where the primary weapons are not already covering a target,
because it is slow to arrive and expensive to spend. A target under a salvo
and a burst at once is therefore normal, not a conflict to be resolved.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np

from ..core.vecmath import unit
from . import lethality
from .effectors import Interceptor, Projectile, step_projectiles
from .mounts import FireMode, GunMount
from .intercept import solve_gun, solve_intercept
from .seeker import Seeker

_salvo_ids = itertools.count(1)

# Cells within one launcher are spaced along its rail.
_CELL_SPACING = 0.9
_LAUNCHER_RING_R = 4.0


#: Below this speed a track is station-keeping rather than travelling.
LOITER_SPEED_MS = 5.0


@dataclass
class Threat:
    track_id: int
    track: object
    cpa_distance: float          # closest approach to the defended asset
    time_to_cpa: float
    range_now: float
    closing: float
    score: float
    speed: float = 0.0

    @property
    def inbound(self) -> bool:
        return self.closing > 0.0

    @property
    def loitering(self) -> bool:
        """Station-keeping rather than leaving.

        `closing <= 0` lumps two opposite situations together: a target
        flying away, which is becoming someone else's problem, and one
        hovering over the defended asset, which is not approaching only
        because it has already arrived. Treating the second as harmless is
        how a drone can sit above the site indefinitely while the battery
        watches it.
        """
        return self.speed < LOITER_SPEED_MS

    @property
    def engageable(self) -> bool:
        """Worth committing a weapon to: closing, or parked overhead."""
        return self.inbound or self.loitering


def evaluate_threats(tracks, asset_pos, asset_radius: float = 40.0
                     ) -> list[Threat]:
    """Rank tracks by how dangerous they are to the defended asset.

    Range alone is a poor ranking: a fast crosser at 300 m may never
    threaten the asset while a slow inbound at 600 m certainly will. Closest
    point of approach and the time to reach it capture that difference.
    """
    asset_pos = np.asarray(asset_pos, dtype=float)
    out: list[Threat] = []

    for trk in tracks:
        rel = trk.position - asset_pos
        vel = trk.velocity
        rng = float(np.linalg.norm(rel))
        speed = float(np.linalg.norm(vel))

        if speed < 1e-6:
            t_cpa, cpa = 0.0, rng
        else:
            t_cpa = float(-(rel @ vel) / (speed ** 2))
            t_cpa = max(t_cpa, 0.0)
            cpa = float(np.linalg.norm(rel + vel * t_cpa))

        closing = float(-(rel @ vel) / rng) if rng > 1e-6 else 0.0

        # Score = how close it will get, times how soon, times whether it is
        # actually coming.
        #
        # Two details matter. Proximity decays smoothly rather than cutting
        # off at the asset radius: a target that will pass just outside the
        # perimeter is not suddenly harmless. And a receding target has its
        # closest approach *behind* it, so t_cpa clamps to zero and its
        # urgency term goes to maximum -- exactly backwards. Without a firm
        # penalty a departing target outranks a genuine inbound.
        urgency = 1.0 / (1.0 + max(t_cpa, 0.0))
        proximity = float(np.exp(-cpa / max(asset_radius, 1e-6)))
        score = 100.0 * proximity * urgency
        if closing <= 0 and speed >= LOITER_SPEED_MS:
            score *= 0.05          # leaving: not a threat, whatever its range

        out.append(Threat(trk.id, trk, cpa, t_cpa, rng, closing, score, speed))

    out.sort(key=lambda th: -th.score)
    return out


@dataclass
class EngagementRecord:
    t: float
    track_id: int
    weapon: str
    salvo_id: int
    rounds: int = 0
    result: str = "pending"
    miss_distance: float | None = None


class EngagementManager:
    def __init__(self, gun_positions=((-14.0, 0.0, 3.0), (14.0, 0.0, 3.0)),
                 missile_pos=(0.0, 0.0, 2.0),
                 asset_pos=(0.0, 0.0, 0.0),
                 fire_mode=FireMode.AUTO,
                 gun_max_range: float = 1800.0,
                 gun_min_range: float = 40.0,
                 gun_muzzle: float = 1000.0,
                 gun_rate_rpm: float = 900.0,
                 gun_belt: int = 900,
                 gun_dispersion_mrad: float = 1.7,
                 gun_max_tof: float = 2.2,
                 missile_sets: int = 2,
                 missiles_per_set: int = 6,
                 missile_max_range: float = 3500.0,
                 missile_min_range: float = 300.0,
                 missile_speed_est: float = 185.0,
                 missile_reload_s: float = 4.0,
                 missile_seeker_range: float = 900.0,
                 missile_seeker_fov_deg: float = 30.0,
                 uav_hangar: int = 2,
                 uav_pad=(0.0, 0.0, 3.0),
                 uav_max_range: float = 1500.0,
                 uav_min_range: float = 80.0,
                 uav_speed_est: float = 60.0,
                 uav_max_sigma: float = 8.0,
                 uav_reload_s: float = 8.0,
                 uav_seeker_range: float = 1500.0,
                 uav_seeker_fov_deg: float = 70.0,
                 max_simultaneous: int = 3,
                 min_track_age: float = 0.6,
                 gun_max_sigma: float = 2.0,
                 missile_max_sigma: float = 20.0,
                 rng=None) -> None:
        self.rng = rng if rng is not None else np.random.default_rng(0)
        self.gun_positions = [np.asarray(p, dtype=float) for p in gun_positions]
        self.missile_pos = np.asarray(missile_pos, dtype=float)
        self.asset_pos = np.asarray(asset_pos, dtype=float)
        self.gun_max_range = gun_max_range
        self.gun_min_range = gun_min_range
        self.gun_muzzle = gun_muzzle
        self.gun_max_tof = gun_max_tof
        self.fire_mode = (fire_mode if isinstance(fire_mode, FireMode)
                          else FireMode.parse(fire_mode))
        self.missile_max_range = missile_max_range
        self.missile_min_range = missile_min_range
        # Average speed over the whole flight, not peak. The missile leaves
        # the rail slowly, boosts hard, then bleeds energy to drag; using the
        # peak here makes every intercept solution optimistic and the
        # engagement manager re-fire at a target whose missile is still on
        # its way.
        self.missile_speed_est = missile_speed_est
        self.missile_reload_s = float(missile_reload_s)
        # Terminal seeker. Shorter-ranged and narrower than the chaser's,
        # because a missile closes fast and nearly head-on: it needs the last
        # few hundred metres resolved, not a wide search cone. Before lock it
        # flies the ground track it was launched on; after lock it steers on
        # its own measured line of sight, and the handoff is where the miss
        # distance collapses.
        self.missile_seeker_range = float(missile_seeker_range)
        self.missile_seeker_fov_deg = float(missile_seeker_fov_deg)
        self.max_simultaneous = max_simultaneous
        self.min_track_age = min_track_age

        # --- counter-drone hangar: a slow, long-endurance chaser that has to
        # hunt a manoeuvring target. It is expendable like a missile but much
        # cheaper to think about; because it is slow, it only commits when
        # the primary weapons are not already covering, and only when the
        # geometry is actually catchable at its modest speed.
        self.uav_pad = np.asarray(uav_pad, dtype=float)
        self.uav_max_range = float(uav_max_range)
        self.uav_min_range = float(uav_min_range)
        self.uav_speed_est = float(uav_speed_est)
        self.uav_max_sigma = float(uav_max_sigma)
        self.uav_reload_s = float(uav_reload_s)
        self.uav_seeker_range = float(uav_seeker_range)
        self.uav_seeker_fov_deg = float(uav_seeker_fov_deg)
        # Fire discipline, and the sharpest coupling from filter quality to
        # outcome. The thresholds differ by weapon for a physical reason: an
        # unguided round is committed at the trigger, so its miss distance is
        # the track error at that instant and nothing downstream can fix it.
        # A missile keeps correcting all the way in, so it can be launched on
        # a far looser solution and still arrive.
        self.gun_max_sigma = gun_max_sigma
        self.missile_max_sigma = missile_max_sigma

        # --- CRAM battery: one cyclic mount per position, each with its own
        # thermal state and ammo belt so no turret can monopolise the fight.
        self.guns = [GunMount(p, muzzle_speed=gun_muzzle,
                              rate_rpm=gun_rate_rpm, belt=gun_belt,
                              dispersion_mrad=gun_dispersion_mrad,
                              rng=self.rng)
                     for p in self.gun_positions]
        self.gun_tracks: list[int | None] = [None] * len(self.guns)
        # Each continuous stream out of a mount is one burst; a fresh burst
        # on the same target after a lull is a new salvo for scoring.
        self._gun_burst: list[int | None] = [None] * len(self.guns)
        self.gun_aims: list[tuple] = []   # (gun_pos, aim_dir, impact) for render

        # --- missile launchers: `missile_sets` rails of `missiles_per_set`
        # tubes each. Every tube reloads independently after a miss/ride-out.
        self.cells: list[dict] = []
        n_sets = max(int(missile_sets), 1)
        rails = [k for k in range(n_sets)]
        for s in rails:
            ang = 2.0 * np.pi * s / n_sets + 0.6
            og = (self.missile_pos
                  + _LAUNCHER_RING_R * np.array([np.cos(ang), np.sin(ang), 0.0]))
            tang = np.array([-np.sin(ang), np.cos(ang), 0.0])
            n_cells = max(int(missiles_per_set), 1)
            for c in range(n_cells):
                off = (c - (n_cells - 1) / 2.0) * _CELL_SPACING
                self.cells.append({"position": og + tang * off,
                                   "loaded": True, "ready_at": 0.0})
        self.cell_positions = np.array([c["position"] for c in self.cells])

        # Each hangar slot reloads independently, like a launcher tube.
        self.uav_cells = [{"loaded": True, "ready_at": 0.0}
                          for _ in range(max(int(uav_hangar), 1))]
        self.uavs: list[Interceptor] = []

        self.projectiles: list[Projectile] = []
        self.missiles: list[Interceptor] = []
        self.records: list[EngagementRecord] = []
        # (weapon, track_id) -> expiry. Assignments must expire, or a target
        # that survives its first burst is never engaged again.
        self.assigned: dict[tuple[str, int], float] = {}
        self.salvo_miss: dict[int, float] = {}   # salvo_id -> closest approach
        self.stats = {"rounds": 0, "missiles": 0, "bursts": 0, "uavs": 0,
                      "kills": 0, "seeker_losses": 0,
                      "declined_uncertain": 0,
                      "declined_envelope": 0}

    # -------------------------------------------------------------- battery
    @property
    def n_guns(self) -> int:
        return len(self.guns)

    @property
    def magazine(self) -> int:
        """Tracking missiles loaded and ready right now."""
        return sum(1 for c in self.cells if c["loaded"])

    def _refill(self, t: float) -> None:
        for c in self.cells:
            if not c["loaded"] and t >= c["ready_at"]:
                c["loaded"] = True
        for c in self.uav_cells:
            if not c["loaded"] and t >= c["ready_at"]:
                c["loaded"] = True

    def _loaded_cell(self, prefer: np.ndarray) -> dict:
        """Cheapest loaded tube for the requested aim direction."""
        best, best_off = None, float("inf")
        for c in self.cells:
            if not c["loaded"]:
                continue
            off = float(np.linalg.norm(c["position"] - prefer))
            if off < best_off:
                best, best_off = c, off
        return best

    # ---------------------------------------------------------------- firing
    def _designate_gun(self, t: float, gi: int, threat: Threat) -> None:
        self.gun_tracks[gi] = threat.track_id
        # Longer than the decision loop: a mount that already owns a target
        # is allowed to keep it (see `decide`), so this only gates other
        # mounts from grabbing the same threat mid-engagement.
        self.assigned[("gun", threat.track_id)] = t + 1.0

    def _fire_missile(self, t: float, threat: Threat) -> None:
        trk = threat.track
        accel = trk.filter.x[6:9] if len(trk.filter.x) >= 9 else None
        tgo, aim = solve_intercept(self.missile_pos, trk.position,
                                   trk.velocity, self.missile_speed_est,
                                   target_accel=accel)
        if tgo is None:
            self.stats["declined_envelope"] += 1
            return

        cell = self._loaded_cell(aim)
        if cell is None:
            return
        cell["loaded"] = False
        cell["ready_at"] = t + self.missile_reload_s

        launch_dir = unit(aim - cell["position"])
        # Missiles leave the rail slowly and accelerate; launching at the
        # full estimate would flatter the intercept solution.
        self.missiles.append(Interceptor(
            cell["position"], launch_dir * 45.0, trk.id,
            seeker=Seeker(self.rng, range_m=self.missile_seeker_range,
                          fov_deg=self.missile_seeker_fov_deg),
            seeker_range=self.missile_seeker_range,
            seeker_fov_deg=self.missile_seeker_fov_deg))
        self.stats["missiles"] += 1
        # Hold the assignment for the whole plausible flight, with margin:
        # releasing it early spends a second round on a target already
        # covered.
        self.assigned[("missile", trk.id)] = t + (tgo or 8.0) * 1.6 + 6.0
        self.records.append(EngagementRecord(t, trk.id, "missile",
                                             next(_salvo_ids), 1))

    def _fire_uav(self, t: float, threat: Threat) -> None:
        """Launch a counter-drone chaser from the hangar.

        A chaser is slow and must be able to *catch* before it can hit, so
        the intercept solution is computed at its modest cruise estimate and
        a target outside that envelope is honestly declined. It carries the
        only seeker in the battery: without it, a slow chaser could never
        land a manoeuvring quad.
        """
        trk = threat.track
        tgo, aim = solve_intercept(self.uav_pad, trk.position, trk.velocity,
                                   self.uav_speed_est)
        if tgo is None:
            self.stats["declined_envelope"] += 1
            return

        cell = next((c for c in self.uav_cells if c["loaded"]), None)
        if cell is None:
            return
        cell["loaded"] = False
        cell["ready_at"] = t + self.uav_reload_s

        launch_dir = unit(np.asarray(aim, float) - self.uav_pad)
        self.uavs.append(Interceptor(
            self.uav_pad, launch_dir * 45.0, trk.id,
            uav=True,
            seeker=Seeker(self.rng, range_m=self.uav_seeker_range,
                          fov_deg=self.uav_seeker_fov_deg),
            boost_accel=30.0, boost_time=1.0, max_lateral=140.0,
            fuze_radius=2.0, max_time=60.0,
            seeker_range=self.uav_seeker_range,
            seeker_fov_deg=self.uav_seeker_fov_deg,
            drag_coeff=0.32, area=0.02, mass=11.0))
        self.stats["uavs"] += 1
        self.assigned[("uav", trk.id)] = t + (tgo or 8.0) * 1.4 + 8.0
        self.records.append(EngagementRecord(t, trk.id, "uav",
                                             next(_salvo_ids), 1))

    # ------------------------------------------------------------------ step
    def decide(self, t: float, tracks) -> list[Threat]:
        """Rank threats and commit shots where doctrine allows.

        Layered, not exclusive: missiles go first at every ranked inbound
        that is still unassigned, and the CRAM guns independently engage the
        highest-ranked threats each of them can reach -- same target can be
        under both at once.
        """
        self._refill(t)
        threats = evaluate_threats(
            [tk for tk in tracks if tk.confirmed], self.asset_pos)
        inbound = [th for th in threats
                   if th.engageable and th.track.age >= self.min_track_age]

        # --- missiles: primary weapon, fired first -------------------------
        if self.fire_mode in (FireMode.AUTO, FireMode.MISSILE):
            active = len([m for m in self.missiles if m.alive])
            for th in inbound:
                if self.assigned.get(("missile", th.track_id)) is not None:
                    continue
                if self.magazine <= 0 or active >= self.max_simultaneous:
                    break
                rng = float(np.linalg.norm(th.track.position - self.missile_pos))
                if not (self.missile_min_range <= rng <= self.missile_max_range):
                    continue
                if th.track.position_sigma > self.missile_max_sigma:
                    self.stats["declined_uncertain"] += 1
                    continue
                self._fire_missile(t, th)
                active += 1

        # --- counter-drone UAVs: only where the primary weapons aren't -----
        if self.fire_mode is FireMode.AUTO:
            active_uavs = len([m for m in self.uavs if m.alive])
            pool = sum(1 for c in self.uav_cells if c["loaded"])
            for th in inbound:
                # Never stack a slow chaser on a target the missile already
                # owns -- it would only arrive late and burn the magazine.
                if self.assigned.get(("missile", th.track_id)) is not None:
                    continue
                if self.assigned.get(("uav", th.track_id)) is not None:
                    continue
                if pool <= 0 or active_uavs >= len(self.uav_cells):
                    break
                if th.track.position_sigma > self.uav_max_sigma:
                    continue
                rng = float(np.linalg.norm(th.track.position - self.uav_pad))
                if not (self.uav_min_range <= rng <= self.uav_max_range):
                    continue
                self._fire_uav(t, th)
                active_uavs += 1

        # --- CRAM guns: simultaneous, close-in coverage --------------------
        if self.fire_mode in (FireMode.AUTO, FireMode.GUN):
            covered: set[int] = set()
            for gi, gun in enumerate(self.guns):
                if not gun.ready:
                    self.gun_tracks[gi] = None
                    continue
                owned = self.gun_tracks[gi]
                best: Threat | None = None
                for th in inbound:
                    if th.track_id in covered:
                        continue
                    # A mount keeps what it already owns; a new target must
                    # not be stolen while another gun is already covering it.
                    if (owned != th.track_id
                            and self.assigned.get(("gun", th.track_id)) is not None):
                        continue
                    if th.track.position_sigma > self.gun_max_sigma:
                        continue
                    rng = float(np.linalg.norm(th.track.position - gun.position))
                    if not (self.gun_min_range <= rng <= self.gun_max_range):
                        continue
                    d, tof, _ = solve_gun(gun.position, th.track.position,
                                          th.track.velocity, self.gun_muzzle)
                    if d is None or tof is None or tof > self.gun_max_tof:
                        continue
                    best = th
                    break
                if best is None:
                    self.gun_tracks[gi] = None
                    continue
                covered.add(best.track_id)
                self._designate_gun(t, gi, best)
        else:
            self.gun_tracks = [None] * len(self.guns)

        return threats

    def step_effectors(self, t: float, dt: float, tracks, targets) -> list[str]:
        """Fly rounds and missiles, then resolve outcomes.

        Guidance below reads only track estimates. Whether anything actually
        connected is decided in `lethality`, the single module permitted to
        see ground truth -- and only after the shot is beyond recall.
        """
        events: list[str] = []
        by_id = {tk.id: tk for tk in tracks}

        # --- guns: each mount solves and fires for its own track -----------
        self.gun_aims = []
        for gi, gun in enumerate(self.guns):
            aim_dir = None
            impact = None
            tid = self.gun_tracks[gi]
            trk = by_id.get(tid) if tid is not None else None
            if (trk is not None and trk.confirmed
                    and self.fire_mode is not FireMode.HOLD
                    and trk.position_sigma <= self.gun_max_sigma):
                accel = trk.filter.x[6:9] if len(trk.filter.x) >= 9 else None
                d, tof, imp = solve_gun(gun.position, trk.position, trk.velocity,
                                        self.gun_muzzle, target_accel=accel)
                if d is not None and tof is not None and tof <= self.gun_max_tof:
                    aim_dir, impact = d, imp

            if aim_dir is not None:
                if self._gun_burst[gi] is None:
                    self._gun_burst[gi] = next(_salvo_ids)
                    self.stats["bursts"] += 1
            else:
                self._gun_burst[gi] = None

            new_rounds = gun.update(t, dt, aim_dir)
            for r in new_rounds:
                r.salvo_id = self._gun_burst[gi]
            if new_rounds:
                self.projectiles.extend(new_rounds)
                self.stats["rounds"] += len(new_rounds)
            if aim_dir is not None:
                self.gun_aims.append((gun.position, aim_dir, impact))
            elif self.gun_tracks[gi] is not None:
                self.gun_tracks[gi] = None

        step_projectiles(self.projectiles, t, dt)

        # --- guided vehicles: missiles and counter-drones ------------------
        # Guidance reads only track estimates, except where a vehicle carries
        # its own seeker -- and then it observes truth only as a sensor and
        # steers on the measurement. See `seeker.py`.
        for m in self.missiles:
            self._guide_interceptor(m, t, dt, by_id, targets, events)
        for m in self.uavs:
            self._guide_interceptor(m, t, dt, by_id, targets, events)

        flyers = self.missiles + self.uavs

        events += lethality.check_impacts(t, self.projectiles, targets,
                                          self.stats, self.salvo_miss)
        events += lethality.check_fuzes(t, flyers, targets, self.stats,
                                        self.records, self.rng)

        self.projectiles = [p for p in self.projectiles if p.alive]
        self.missiles = [m for m in self.missiles if m.alive]
        self.uavs = [m for m in self.uavs if m.alive]

        # Release assignments so a surviving target can be re-engaged. For
        # guided weapons the honest condition is whether one is still on its
        # way, not a timer.
        in_flight = ({m.target_track_id for m in self.missiles if m.alive}
                     | {m.target_track_id for m in self.uavs if m.alive})
        for (weapon, tid) in list(self.assigned):
            if tid not in by_id:
                del self.assigned[(weapon, tid)]
            elif weapon in ("missile", "uav"):
                if tid not in in_flight:
                    del self.assigned[(weapon, tid)]
            elif t > self.assigned[(weapon, tid)]:
                del self.assigned[(weapon, tid)]
        return events

    def _guide_interceptor(self, m: Interceptor, t: float, dt: float,
                           by_id: dict, targets, events: list) -> None:
        if not m.alive:
            return
        trk = by_id.get(m.target_track_id)
        if trk is None:
            # Track lost: fly the last solution out and time out. The
            # vehicle does not get to fall back on truth.
            m.guide(t, dt, m.position + m.velocity * 2.0, m.velocity)
            return
        accel = trk.filter.x[6:9] if len(trk.filter.x) >= 9 else None
        if m.seeker is not None:
            out, ev = m.seeker.observe(t, dt, m.position, m.velocity,
                                       targets, trk.position)
            if ev is not None:
                if ev.startswith("SEEKER LOSS"):
                    self.stats["seeker_losses"] += 1
                events.append(f"t={t:6.2f}  {ev}")
            m.guide(t, dt, trk.position, trk.velocity, accel, seeker_out=out)
        else:
            m.guide(t, dt, trk.position, trk.velocity, accel)

    def mode_status(self) -> str:
        n_ready = sum(1 for g in self.guns if g.ready)
        uav_pool = sum(1 for c in self.uav_cells if c["loaded"])
        uav_air = len([m for m in self.uavs if m.alive])
        return (f"mode={self.fire_mode.value.upper()}  "
                f"guns {n_ready}/{self.n_guns} ready  "
                f"msl {self.magazine}/{len(self.cells)} loaded  "
                f"uav {uav_pool}/{len(self.uav_cells)} hangar "
                f"({uav_air} airborne)")

    def cycle_mode(self) -> FireMode:
        order = [FireMode.AUTO, FireMode.GUN, FireMode.MISSILE, FireMode.HOLD]
        self.fire_mode = order[(order.index(self.fire_mode) + 1) % len(order)]
        return self.fire_mode

    def burst_summary(self) -> str:
        """Closest approach achieved by each gun burst."""
        if not self.salvo_miss:
            return "no bursts fired"
        misses = sorted(self.salvo_miss.values())
        return (f"{len(misses)} bursts, closest approach: "
                f"best={misses[0]:.2f} m median={misses[len(misses)//2]:.2f} m "
                f"worst={misses[-1]:.2f} m")