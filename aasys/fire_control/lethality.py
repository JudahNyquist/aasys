"""Outcome resolution -- the only place allowed to touch ground truth.

There is a real distinction between two kinds of truth access, and the
architecture makes it explicit rather than trusting anyone to remember it.

**Deciding where to shoot** must never see truth. Threat ranking, intercept
solutions, gun lead and missile guidance all read track estimates, so a
filter that is two metres off puts the burst two metres off. That is the
entire point of the simulation and `aasys.core.truth_guard` enforces it.

**Deciding whether a shot connected** must see truth. Once a round is in the
air, whether it passes within lethal radius of the actual target is physics,
not estimation -- no amount of filtering changes where the metal went. That
resolution lives here, in one module, after the shot is already committed
and beyond influence.

Keeping these apart in separate files means the cheat detector can allow
exactly one of them. Any truth read from `engagement`, `guidance` or
`intercept` is then unambiguously a bug rather than a judgement call.
"""

from __future__ import annotations

import numpy as np

from ..core.vecmath import segment_distance
from .effectors import kill_probability


def check_impacts(t: float, projectiles, targets, stats,
                  salvo_miss: dict) -> list[str]:
    """Resolve already-moved projectiles against true target positions."""
    events: list[str] = []
    # Not `p.alive`: a round that expired during this step -- ran out of
    # time or reached the ground -- still swept a segment, and the caller
    # prunes the list only after this returns, so anything dead and unspent
    # here died this step. Skipping those threw away every hit that landed
    # on the round's last leg, which for a low, flat engagement is most of
    # them.
    live = [p for p in projectiles if not p.spent]
    if not live:
        return events

    # One burst puts hundreds of rounds against a handful of targets, so the
    # pairwise distance is one small matrix per target rather than a norm per
    # pair. The loop is target-major to make that vectorisation possible; a
    # round still kills at most once, but where a single round is inside the
    # lethal radius of two targets at the same instant, which one it is
    # credited with can differ from the round-major order. That case needs two
    # targets within ~1 m of each other and does not change the kill count.
    a = np.array([p.prev_position for p in live], dtype=float)
    b = np.array([p.position for p in live], dtype=float)
    for g in targets:
        if g.destroyed:
            continue
        d_all = segment_distance(a, b, g.position)
        for i, p in enumerate(live):
            if p.spent:
                continue
            # Re-check every pass: an earlier round in this same burst may
            # already have killed it, and a dead target must not be counted
            # again by the rounds still arriving.
            if g.destroyed:
                break
            d = float(d_all[i])
            prev = salvo_miss.get(p.salvo_id)
            if prev is None or d < prev:
                salvo_miss[p.salvo_id] = d
            if d <= p.lethal_radius + g.radius:
                g.kill(t)
                p.alive = False
                p.spent = True
                stats["kills"] += 1
                events.append(f"t={t:6.2f}  GUN HIT  {g.name} "
                              f"(round {p.id}, miss {d:.2f} m)")
    return events


def check_fuzes(t: float, missiles, targets, stats, records, rng) -> list[str]:
    """Resolve proximity fuzes and lethality against true target positions."""
    events: list[str] = []
    for m in missiles:
        if not m.alive:
            continue
        for g in targets:
            if g.destroyed:
                continue
            if not m.check_fuze(g.position, key=g.id):
                continue
            pk = kill_probability(m.miss_distance, m.fuze_radius)
            hit = rng.random() < pk
            if hit:
                g.kill(t)
                stats["kills"] += 1
            events.append(
                f"t={t:6.2f}  MISSILE {'KILL' if hit else 'MISS'} "
                f"{g.name} (miss {m.miss_distance:.2f} m, Pk={pk:.2f})")
            for rec in records:
                if (rec.weapon in ("missile", "uav")
                        and rec.track_id == m.target_track_id
                        and rec.result == "pending"):
                    rec.result = "kill" if hit else "miss"
                    rec.miss_distance = m.miss_distance
                    break
            break
    return events
