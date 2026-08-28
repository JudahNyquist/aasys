"""Scenario library.

Each scenario is a dict the Simulation consumes. They are sized around the
carving resolution rule (voxel <= target_diameter / 1.5), because a grid too
coarse for its targets is simply blind to them.
"""

from __future__ import annotations

import numpy as np

from ..carving.grid import VoxelGrid
from ..entities.flight_profiles import (Hover, Jink, Orbit, TerminalDive,
                                        Waypoint)
from ..entities.target import Target
from ..sensing.rig import CameraRig
from ..tracking.factory import cv_ekf_factory, imm_factory

DRONE_RADIUS = 0.45          # 0.9 m across -- a large quadcopter


def _default_grid(size: float = 0.5) -> VoxelGrid:
    """The optical 'bubble': the volume worth reconstructing precisely.

    The floor is set by the rig, not by taste. Cameras on 6-26 m masts aimed
    at mid-altitude cannot see the ground beneath themselves, and the carver
    refuses any voxel fewer than `min_cameras` can see -- so below about 18 m
    no voxel can ever be occupied whatever flies through it. Carrying that
    band anyway cost 16% of the grid in memory and lookup-table build time to
    hold voxels guaranteed to stay empty. `CarvingLUT.coverage_check` reports
    the usable band and is printed at startup.
    """
    return VoxelGrid(lo=[-50.0, -50.0, 17.0], hi=[50.0, 50.0, 85.0], size=size)


def _default_rig(n: int = 6) -> CameraRig:
    return CameraRig.ring_staggered(
        n=n, radius=90.0, fov_deg=45.0, width=1920, height_px=1080,
        aim=np.array([0.0, 0.0, 40.0]), heights=(6.0, 26.0))


def _base(**over) -> dict:
    cfg = {
        "dt": 1.0 / 120.0,
        "grid": _default_grid(),
        "rig": _default_rig(),
        "asset": [0.0, 0.0, 0.0],
        "filter_factory": imm_factory(),
        "optical_params": {"rate_hz": 20.0, "min_cameras": 3},
        "radar_params": {"clutter_rate_hz": 0.4},
        "tracker_params": {},
        "engagement_params": {
            "gun_positions": ((-14.0, 0.0, 3.0), (14.0, 0.0, 3.0)),
            "missile_sets": 2,
            "missiles_per_set": 6,
        },
        "engage": True,
    }
    cfg.update(over)
    return cfg


# --------------------------------------------------------------- scenarios
def single_inbound() -> dict:
    """One drone flying straight at the asset. The baseline."""
    t = Target([900.0, 250.0, 90.0], [-38.0, -10.0, -1.5],
               Waypoint([[0.0, 0.0, 25.0]], speed=40.0),
               radius=DRONE_RADIUS, rcs=0.04, name="inbound-1")
    return _base(targets=[t])


def maneuvering() -> dict:
    """A weaving drone -- the case a single constant-velocity model fails on
    and an IMM is meant to catch."""
    t = Target([800.0, -200.0, 80.0], [-40.0, 12.0, 0.0],
               Jink(np.array([0.0, 0.0, 30.0]), speed=38.0,
                    amplitude=22.0, period=3.5),
               radius=DRONE_RADIUS, rcs=0.04, name="jinker")
    return _base(targets=[t])


def hovering_gap() -> dict:
    """A hovering drone inside the optical bubble plus an inbound crosser.

    The hoverer sits in the radar's zero-Doppler clutter notch and is
    invisible to it; only voxel carving holds the track. This is the
    scenario that justifies having both sensors.
    """
    hover = Target([25.0, -15.0, 45.0], [0.0, 0.0, 0.0],
                   Hover(np.array([25.0, -15.0, 45.0])),
                   radius=DRONE_RADIUS, rcs=0.04, name="hoverer")
    runner = Target([700.0, 400.0, 70.0], [-35.0, -22.0, -1.0],
                    Waypoint([[0.0, 0.0, 30.0]], speed=40.0),
                    radius=DRONE_RADIUS, rcs=0.04, name="inbound-2")
    return _base(targets=[hover, runner])


def swarm() -> dict:
    """A small saturation raid: several drones converging from all sides.

    Stresses data association, the array's dwell budget, and the magazine.
    """
    rng = np.random.default_rng(11)
    targets = []
    for i in range(5):
        a = 2 * np.pi * i / 5 + 0.3
        r = rng.uniform(650, 950)
        start = np.array([r * np.cos(a), r * np.sin(a), rng.uniform(60, 110)])
        targets.append(Target(
            start, -start / np.linalg.norm(start) * 35.0,
            Waypoint([[rng.uniform(-25, 25), rng.uniform(-25, 25), 25.0]],
                     speed=rng.uniform(32, 46)),
            radius=DRONE_RADIUS, rcs=float(rng.uniform(0.02, 0.06)),
            name=f"raid-{i+1}"))
    return _base(targets=targets,
                 engagement_params={"missile_sets": 1, "missiles_per_set": 6})


def orbiter() -> dict:
    """A drone loitering over the site: a sustained turn, so the turn models
    in the IMM bank should take over from constant velocity."""
    t = Target([40.0, 0.0, 50.0], [0.0, 20.0, 0.0],
               Orbit(np.array([0.0, 0.0, 50.0]), radius=40.0, speed=20.0),
               radius=DRONE_RADIUS, rcs=0.04, name="loiterer")
    return _base(targets=[t], engage=False)


def diving_attack() -> dict:
    """Cruise then terminal dive -- a sharp manoeuvre late in the engagement,
    exactly when the fire-control solution is least forgiving.

    This scenario needs its own optics. The standard rig sits on 6-26 m masts
    aimed at mid-altitude and cannot see the ground beneath itself, so with
    the default bubble the diver spent its entire terminal phase below the
    lowest carvable voxel and the optical channel reported nothing at all --
    silently, since a sensor that sees nothing looks exactly like a sensor
    with nothing to see. Shorter masts, a wider field of view and a lower aim
    trade reach for a bubble that reaches the ground, which is where this
    threat ends up.
    """
    t = Target([600.0, 300.0, 120.0], [-35.0, -18.0, 0.0],
               TerminalDive(np.array([0.0, 0.0, 5.0]), cruise_speed=34.0,
                            dive_speed=58.0, trigger_range=260.0),
               radius=DRONE_RADIUS, rcs=0.05, name="diver")
    return _base(targets=[t],
                 grid=VoxelGrid(lo=[-55.0, -55.0, 3.0],
                                hi=[55.0, 55.0, 55.0], size=0.5),
                 rig=CameraRig.ring_staggered(
                     n=6, radius=80.0, fov_deg=55.0, width=1920,
                     height_px=1080, aim=np.array([0.0, 0.0, 15.0]),
                     heights=(5.0, 18.0)))


def standoff() -> dict:
    """Fixed-wing threats engaged at standoff range by the interceptor.

    Weapon choice is really driven by *detection* range, not by the weapons.
    A quadcopter's 0.05 m^2 cross-section is not seen until roughly 900 m,
    which is comfortably inside gun envelope, so the gun always wins. Only a
    target radar can find far out -- an aircraft-sized RCS, detected near
    1.5 km -- is beyond the gun's 2.2 s time-of-flight limit and falls to the
    missile. That is the real division of labour between the two, and it
    comes from the radar equation rather than from doctrine.
    """
    rng = np.random.default_rng(21)
    targets = []
    for i in range(3):
        a = 2 * np.pi * i / 3 + 0.6
        r = rng.uniform(2800, 3300)
        start = np.array([r * np.cos(a), r * np.sin(a), rng.uniform(150, 240)])
        targets.append(Target(
            start, -start / np.linalg.norm(start) * 80.0,
            Waypoint([[0.0, 0.0, 40.0]], speed=80.0),
            radius=1.5, rcs=1.5, mass=120.0, max_accel=45.0,
            name=f"standoff-{i+1}"))
    return _base(targets=targets,
                 engagement_params={"missile_sets": 2, "missiles_per_set": 6,
                                    "max_simultaneous": 4})


def counterdrone() -> dict:
    """A close-in raid stopped by counter-drone chasers rather than the guns.

    Three jinking quads converge inside a kilometre -- well inside gun range,
    but weaving. A gun needs a tight track (sigma <= 2 m) to commit rounds; a
    counter-drone tolerates a looser solution and closes the last gap with its
    own seeker. Missiles are reserved for standoff work (`missile_min_range`),
    so the close-in ring belongs to the third weapon.
    """
    rng = np.random.default_rng(31)
    targets = []
    for i in range(3):
        a = 2 * np.pi * i / 3 + 0.5
        r = rng.uniform(700, 1100)
        start = np.array([r * np.cos(a), r * np.sin(a), rng.uniform(55, 90)])
        targets.append(Target(
            start, -start / np.linalg.norm(start) * rng.uniform(28, 36),
            Jink(np.array([0.0, 0.0, 25.0]), speed=rng.uniform(28, 38),
                 amplitude=20.0, period=3.8),
            radius=DRONE_RADIUS, rcs=0.04, name=f"raider-{i+1}"))
    return _base(targets=targets,
                 engagement_params={"uav_hangar": 3, "missile_sets": 1,
                                    "missile_min_range": 1300,
                                    "missile_speed_est": 160.0})


def cv_baseline() -> dict:
    """Identical to `maneuvering` but with a single constant-velocity filter,
    for a like-for-like comparison against the IMM."""
    cfg = maneuvering()
    cfg["filter_factory"] = cv_ekf_factory(q=8.0)
    return cfg


SCENARIOS = {
    "single": single_inbound,
    "maneuvering": maneuvering,
    "hovering": hovering_gap,
    "swarm": swarm,
    "standoff": standoff,
    "orbiter": orbiter,
    "diving": diving_attack,
    "counterdrone": counterdrone,
    "cv-baseline": cv_baseline,
}


def build(name: str) -> dict:
    if name not in SCENARIOS:
        raise SystemExit(f"unknown scenario {name!r}; "
                         f"choose from {', '.join(sorted(SCENARIOS))}")
    return SCENARIOS[name]()
