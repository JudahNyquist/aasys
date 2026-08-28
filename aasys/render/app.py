"""Interactive 3D view.

Everything on screen is computed live: physics, silhouette rasterisation,
voxel carving, clustering, association, filtering and guidance all run
inside the frame loop. Nothing is precomputed or replayed.

The layout exists to make the estimation legible. Ground truth, the raw
detections, and the filter's estimate are three visually distinct layers,
with the uncertainty ellipsoid over the top. When the filter is working the
estimate sits on the truth and the ellipsoid is small; when it is not, the
picture says so before any number does.

Two things keep the frame budget: static geometry (ground grid, camera
frusta, the carving volume) is uploaded once instead of every frame, and
dynamic geometry is assembled as numpy arrays rather than by appending
floats in Python. Without either, the window runs near 20 fps and drags the
simulation to a third of real time.

Controls
    W/A/S/D         tilt up / tilt down / pan left / pan right
    space           pause
    G / M / V       fire mode: gun, missile, auto
    TAB             cycle fire mode
    1..5            toggle layers
"""

from __future__ import annotations

import time

import numpy as np
import pyglet
from pyglet.gl import (GL_BLEND, GL_DEPTH_TEST, GL_LINES, GL_ONE_MINUS_SRC_ALPHA,
                       GL_POINTS, GL_PROGRAM_POINT_SIZE, GL_SRC_ALPHA,
                       glBlendFunc, glClearColor, glEnable)
from pyglet.graphics.shader import Shader, ShaderProgram
from pyglet.math import Mat4, Vec3

from ..fire_control.mounts import FireMode
from ..fire_control.seeker import SeekState
from ..tracking.track import TrackState
from . import scene
from .scene import GeometryBuffer

BG = (0.055, 0.065, 0.085, 1.0)

C_TRUTH        = (0.30, 0.95, 0.55, 1.0)
C_TRUTH_TRAIL  = (0.20, 0.58, 0.35, 0.70)
C_DEAD         = (0.42, 0.42, 0.48, 0.8)
C_MEAS_OPT     = (1.00, 0.85, 0.25, 0.95)
C_MEAS_RADAR   = (1.00, 0.45, 0.40, 0.90)
C_EST          = (0.40, 0.75, 1.00, 1.0)
C_EST_TRAIL    = (0.26, 0.48, 0.76, 0.80)
C_TENT         = (0.68, 0.60, 0.98, 0.9)
C_VOXEL        = (0.95, 0.70, 0.15, 0.85)
C_ELLIPSE      = (1.00, 0.72, 0.20, 0.70)
C_ROUND        = (0.85, 0.45, 0.15, 0.9)
C_TRACER       = (1.00, 0.80, 0.35, 1.0)
C_MISSILE      = (1.00, 0.30, 0.55, 1.0)
C_MSL_TRAIL    = (0.72, 0.24, 0.44, 0.8)
C_UAV          = (0.30, 0.95, 0.95, 1.0)
C_UAV_TRAIL    = (0.18, 0.62, 0.66, 0.85)
C_UAV_LISTEN   = (0.45, 0.75, 0.80, 0.60)
C_UAV_BLIND    = (0.55, 0.55, 0.60, 0.90)
C_AIM          = (1.00, 0.35, 0.30, 0.55)
C_STATIC_BOX   = (0.28, 0.48, 0.62, 0.75)
C_FRUSTUM      = (0.20, 0.42, 0.52, 0.22)
C_BEAM_SEARCH  = (1.00, 0.45, 0.40, 0.16)
C_BEAM_TRACK   = (1.00, 0.75, 0.35, 0.45)


class Viewer(pyglet.window.Window):
    def __init__(self, sim, duration: float = 1e9, **kw):
        super().__init__(width=1500, height=940, resizable=True,
                         caption="aasys — live voxel carving + radar tracking", **kw)
        self.sim = sim
        self.duration = duration

        self.program = ShaderProgram(Shader(scene.VERT_SRC, "vertex"),
                                     Shader(scene.FRAG_SRC, "fragment"))
        glClearColor(*BG)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_BLEND)
        glEnable(GL_PROGRAM_POINT_SIZE)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        # The camera holds a fixed standoff distance and aim point: no mouse
        # orbit, zoom or follow, so the picture never moves with the
        # action. W/A/S/D pan and tilt it around the same aim point.
        self.az, self.el, self.dist = np.radians(45.0), np.radians(26.0), 300.0
        self.focus = np.array([0.0, 0.0, 40.0])
        self.paused = False
        self.show = {"voxels": True, "frusta": True, "ellipsoids": True,
                     "detections": True, "trails": True}

        self.truth_trails: dict[int, list] = {}
        self.est_trails: dict[int, list] = {}
        self._recent_meas: list = []
        self._fps = 0.0
        self._sim_rate = 0.0
        self._last_wall = time.perf_counter()
        self._last_sim = sim.t

        self._static_batch = None
        self._static_lists = []
        self._build_static()

        self.hud = pyglet.text.Label("", font_name="Menlo", font_size=10,
                                     x=12, y=self.height - 18, width=620,
                                     multiline=True, color=(220, 228, 236, 255))
        self.legend = pyglet.text.Label("", font_name="Menlo", font_size=10,
                                        x=12, y=132, width=620,
                                        multiline=True, color=(146, 160, 176, 255))
        self.events_label = pyglet.text.Label("", font_name="Menlo", font_size=10,
                                              x=self.width - 520, y=self.height - 18,
                                              width=500, multiline=True,
                                              color=(255, 190, 140, 255))
        pyglet.clock.schedule_interval(self.update, 1.0 / 60.0)

    # -------------------------------------------------------- static geometry
    def _build_static(self):
        """Upload geometry that never moves exactly once.

        The ground grid, the camera frusta and the carving volume are fixed
        for the whole run. Rebuilding them per frame is pure waste.
        """
        buf = GeometryBuffer()
        verts, cols = scene.ground_grid(450.0, 50.0)
        buf.add_lines_colored(verts, cols)

        sim = self.sim
        if sim.optical is not None:
            g = sim.optical.grid
            buf.add_lines(scene.box_wire(g.lo, g.hi), C_STATIC_BOX)
            cams = np.array([c.C for c in sim.optical.true_rig])
            for cam in sim.optical.true_rig:
                buf.add_lines(scene.camera_frustum(cam, depth=55.0), C_FRUSTUM)
            buf.add_points(cams, (0.35, 0.70, 0.85, 0.9))
        if sim.radar is not None:
            buf.add_points(sim.radar.position[None, :], (1.0, 0.5, 0.45, 1.0))
        eng = sim.engagement
        if eng is not None:
            buf.add_lines(scene.crosses(eng.gun_positions, 3.0), C_AIM)
            buf.add_points(np.array(eng.cell_positions), (0.72, 0.5, 0.35, 0.9))

        lv, lc, pv, pc = buf.build()
        self._static_batch = pyglet.graphics.Batch()
        self._static_lists = []
        if lv:
            self._static_lists.append(self.program.vertex_list(
                len(lv) // 3, GL_LINES, batch=self._static_batch,
                position=("f", lv), colors=("f", lc)))
        if pv:
            self._static_lists.append(self.program.vertex_list(
                len(pv) // 3, GL_POINTS, batch=self._static_batch,
                position=("f", pv), colors=("f", pc)))

    # ------------------------------------------------------------- controls
    def on_key_press(self, symbol, mods):
        k = pyglet.window.key
        if symbol == k.SPACE:
            self.paused = not self.paused
        elif symbol == k.TAB:
            self.sim.engagement.cycle_mode()
        elif symbol == k.A:
            self.az = (self.az - np.radians(3.0)) % (2 * np.pi)
        elif symbol == k.D:
            self.az = (self.az + np.radians(3.0)) % (2 * np.pi)
        elif symbol in (k.W, k.S):
            step = np.radians(3.0) * (1 if symbol == k.W else -1)
            self.el = float(np.clip(self.el + step,
                                    np.radians(-5.0), np.radians(85.0)))
        elif symbol == k.G:
            self.sim.engagement.fire_mode = FireMode.GUN
        elif symbol == k.M:
            self.sim.engagement.fire_mode = FireMode.MISSILE
        elif symbol == k.V:
            self.sim.engagement.fire_mode = FireMode.AUTO
        elif symbol == k.ESCAPE:
            self.close()
        else:
            for i, key in enumerate(("voxels", "frusta", "ellipsoids",
                                     "detections", "trails")):
                if symbol == getattr(k, f"_{i+1}"):
                    self.show[key] = not self.show[key]

    # --------------------------------------------------------------- update
    def update(self, dt):
        now = time.perf_counter()
        wall = now - self._last_wall
        if wall > 0.25:
            self._sim_rate = (self.sim.t - self._last_sim) / wall
            self._last_wall, self._last_sim = now, self.sim.t

        if self.paused or self.sim.t >= self.duration:
            return

        # Advance a fixed slice of simulated time per frame so trajectories
        # never depend on how fast the last frame drew.
        for _ in range(max(1, int((1.0 / 60.0) / self.sim.dt))):
            self.sim.step()

        for g in self.sim.targets:
            tr = self.truth_trails.setdefault(g.id, [])
            tr.append(g.position.copy())
            if len(tr) > 700:
                del tr[0]
        for trk in self.sim.tracker.tracks:
            tr = self.est_trails.setdefault(trk.id, [])
            tr.append(trk.position.copy())
            if len(tr) > 700:
                del tr[0]
        live = {t.id for t in self.sim.tracker.tracks}
        for tid in [k for k in self.est_trails if k not in live]:
            del self.est_trails[tid]

        if self.sim.measurements:
            self._recent_meas = [(m.sensor_id, m.position_hint())
                                 for m in self.sim.measurements
                                 if m.position_hint() is not None][-150:]

    # ----------------------------------------------------------------- draw
    def _eye(self) -> Vec3:
        p = self.focus + self.dist * np.array([
            np.cos(self.el) * np.cos(self.az),
            np.cos(self.el) * np.sin(self.az),
            np.sin(self.el)])
        return Vec3(*p)

    def _dynamic(self) -> GeometryBuffer:
        buf = GeometryBuffer()
        sim = self.sim
        mk = max(self.dist * 0.034, 1.6)

        # Radar beams actually commanded this step.
        if sim.radar is not None and sim.radar.last_dwells:
            for kind, col in (("search", C_BEAM_SEARCH), ("track", C_BEAM_TRACK)):
                d = [x for x in sim.radar.last_dwells if x.kind == kind][:10]
                if d:
                    dirs = scene.beam_directions([x.az for x in d],
                                                 [x.el for x in d])
                    buf.add_lines(scene.rays_from(sim.radar.position, dirs, 700.0),
                                  col)

        # Carved voxels -- the live output of the optical chain.
        if (self.show["voxels"] and sim.optical is not None
                and sim.optical.last_result is not None
                and len(sim.optical.last_result)):
            centers = sim.optical.last_result.centers()
            if len(centers) < 4000:
                buf.add_lines(scene.cubes_wire(centers, sim.optical.grid.size),
                              C_VOXEL)
            buf.add_points(centers, C_VOXEL)

        # Ground truth.
        alive = [g for g in sim.targets if not g.destroyed]
        dead = [g for g in sim.targets if g.destroyed]
        if alive:
            buf.add_lines(scene.crosses([g.position for g in alive], mk), C_TRUTH)
            buf.add_points([g.position for g in alive], C_TRUTH)
        if dead:
            buf.add_lines(scene.crosses([g.position for g in dead], mk * 0.8), C_DEAD)
        if self.show["trails"]:
            for g in sim.targets:
                buf.add_lines(scene.polyline(self.truth_trails.get(g.id), 700, 3),
                              C_TRUTH_TRAIL)

        # Raw detections, before any filtering.
        if self.show["detections"] and self._recent_meas:
            opt = [p for s, p in self._recent_meas if s == "optical"]
            rad = [p for s, p in self._recent_meas if s != "optical"]
            buf.add_points(opt, C_MEAS_OPT)
            buf.add_points(rad, C_MEAS_RADAR)

        # Track estimates.
        conf_pos, tent_pos = [], []
        for trk in sim.tracker.tracks:
            if trk.state is TrackState.TENTATIVE:
                tent_pos.append(trk.position)
                continue
            conf_pos.append(trk.position)
            if self.show["trails"]:
                buf.add_lines(scene.polyline(self.est_trails.get(trk.id), 700, 3),
                              C_EST_TRAIL)
            if self.show["ellipsoids"]:
                cov = trk.position_cov
                floor = (mk * 0.22) ** 2
                if np.trace(cov) / 3.0 < floor:
                    cov = cov + np.eye(3) * floor
                buf.add_lines(scene.covariance_ellipsoid(trk.position, cov, 3.0),
                              C_ELLIPSE)
        if conf_pos:
            buf.add_lines(scene.crosses(conf_pos, mk * 0.9), C_EST)
        if tent_pos:
            buf.add_lines(scene.crosses(tent_pos, mk * 0.6), C_TENT)

        # Gun lines of fire and their predicted impact points.
        eng = sim.engagement
        for gun_pos, aim_dir, impact in eng.gun_aims:
            buf.add_lines(scene.rays_from(gun_pos, aim_dir[None, :], 900.0),
                          C_AIM)
            if impact is not None:
                buf.add_lines(scene.crosses([impact], mk * 0.6), C_AIM)

        # Effectors.
        rounds = [p.position for p in eng.projectiles if not p.tracer]
        tracers = [p.position for p in eng.projectiles if p.tracer]
        buf.add_points(rounds, C_ROUND)
        buf.add_points(tracers, C_TRACER)
        if self.show["trails"]:
            for p in eng.projectiles:
                if p.tracer:
                    buf.add_lines(scene.polyline(p.trail, 60, 1), C_TRACER)
        for m in eng.missiles:
            buf.add_lines(scene.crosses([m.position], mk * 0.8), C_MISSILE)
            if self.show["trails"]:
                buf.add_lines(scene.polyline(m.trail, 400, 3), C_MSL_TRAIL)
        for m in eng.uavs:
            st = getattr(m.seeker, "state", None)
            if st is SeekState.LOCKED:
                col = C_UAV
            elif st is SeekState.DROPPED:
                col = C_UAV_BLIND
            else:
                col = C_UAV_LISTEN
            buf.add_lines(scene.crosses([m.position], mk), col)
            if self.show["trails"]:
                buf.add_lines(scene.polyline(m.trail, 800, 3), C_UAV_TRAIL)
        return buf

    def on_draw(self):
        self.clear()
        aspect = self.width / max(self.height, 1)
        self.projection = Mat4.perspective_projection(aspect, 1.0, 8000.0, 55.0)
        self.view = Mat4.look_at(self._eye(), Vec3(*self.focus), Vec3(0, 0, 1))
        self.program["point_size"] = 10.0

        self._static_batch.draw()

        lv, lc, pv, pc = self._dynamic().build()
        batch = pyglet.graphics.Batch()
        if lv:
            self.program.vertex_list(len(lv) // 3, GL_LINES, batch=batch,
                                     position=("f", lv), colors=("f", lc))
        if pv:
            self.program.vertex_list(len(pv) // 3, GL_POINTS, batch=batch,
                                     position=("f", pv), colors=("f", pc))
        batch.draw()
        self._draw_hud()

    def _draw_hud(self):
        self.projection = Mat4.orthogonal_projection(0, self.width, 0,
                                                     self.height, -1, 1)
        self.view = Mat4()
        sim = self.sim
        eng = sim.engagement
        st = sim.tracker.stats

        lines = [
            f"t={sim.t:7.1f} s   {self._sim_rate:4.2f}x realtime"
            + ("   [PAUSED]" if self.paused else ""),
            f"{eng.mode_status()}",
            "",
            "TRACKING PIPELINE  (live, from detections only)",
            f"  detections this frame  {len(sim.measurements)}",
            f"  associated / missed    {st['hits']} / {st['misses']}",
            f"  tracks  init={st['initiated']} confirmed={st['confirmed']} "
            f"deleted={st['deleted']} suppressed={st['suppressed']}",
            f"  {sim.tracker.summary()}",
        ]
        if sim.radar:
            s = sim.radar.stats
            lines.append(f"radar   det={s['detections']} mti_rej={s['mti_rejected']} "
                         f"fa={s['false_alarms']} frame={sim.radar.search_frame_s:.2f}s "
                         f"dwell s/t={s['search_dwells']}/{s['track_dwells']}")
        if sim.optical and sim.optical.last_result is not None:
            lines.append(f"optical voxels={len(sim.optical.last_result)} "
                         f"blobs={len(sim.optical.last_blobs)} "
                         f"path={sim.optical.last_result.path}")
        lines.append(f"engage  rounds={eng.stats['rounds']} msl={eng.stats['missiles']} "
                     f"uavs={eng.stats['uavs']} kills={eng.stats['kills']} "
                     f"seek_loss={eng.stats['seeker_losses']} "
                     f"declined={eng.stats['declined_uncertain']}")
        lines.append("")
        for trk in sim.tracker.confirmed[:6]:
            got = sim.truth_for_track(trk)
            err = f"{got[1]:5.1f}" if got else "  -  "
            mp = trk.model_probabilities()
            top = max(mp, key=mp.get)
            lines.append(f"  T{trk.id:<3d} {trk.state.value[:5]:<5s} v={trk.speed:5.1f} "
                         f"sig={trk.position_sigma:5.2f} err={err} "
                         f"{top}={mp[top]:.2f} src={trk.last_sensor[:7]}")
        self.hud.text = "\n".join(lines)
        self.hud.y = self.height - 18
        self.hud.draw()

        self.legend.text = (
            "green truth   blue estimate   yellow optical det   red radar det\n"
            "orange voxels + 3sigma ellipsoid   purple tentative   tracers = gun\n"
            "pink missile   cyan counter-drone (bright lock / grey blind)\n"
            "W/A/S/D tilt+pan  space pause  G gun  M missile  V auto  TAB cycle\n"
            "1 voxels  2 frusta  3 ellipsoids  4 detections  5 trails")
        self.legend.draw()

        self.events_label.x = self.width - 520
        self.events_label.y = self.height - 18
        self.events_label.text = "\n".join(sim.events[-14:])
        self.events_label.draw()


def run_window(sim, duration: float = 1e9):
    Viewer(sim, duration=duration)
    pyglet.app.run()
