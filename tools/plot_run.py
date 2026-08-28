"""Plot a recorded run.

Three views answer most questions about a track:

* **error over time** -- where the tracker lost it, not just how badly on
  average. An RMSE hides a two-second dropout inside twenty good seconds.
* **NEES over time**, against its consistency band -- whether the filter's
  claimed covariance was ever entitled to belief. A track that is accurate
  but overconfident will pass an RMSE check and then fragment as soon as the
  gate gets tight.
* **IMM model probabilities** -- what the bank thought the target was doing.
  Reading these against the error trace is the quickest way to see a
  manoeuvre the filter noticed late.

matplotlib is imported lazily and is not in `requirements.txt`. The
simulation itself installs from prebuilt wheels with no compiler, and an
optional analysis tool should not be what breaks that.

Run:  python -m tools.plot_run run.npz [--out run.png]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

# Chi-square 6-DOF, 95% central interval -- the band a consistent NEES over a
# position+velocity state should mostly sit inside.
NEES_LO, NEES_HI = 1.24, 14.45
STATE_CONFIRMED, STATE_COASTING = 1, 2


def _load(path):
    d = np.load(path, allow_pickle=True)
    fields = {name: i for i, name in enumerate(d["track_fields"])}
    return d, d["tracks"], fields


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("npz", help="a .npz written by --record")
    ap.add_argument("--out", help="write a PNG instead of opening a window")
    ap.add_argument("--track", type=int, help="plot only this track id")
    args = ap.parse_args()

    try:
        import matplotlib
        if args.out:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit(
            "plotting needs matplotlib, which the simulation itself does not:\n"
            "    python -m pip install -r requirements-dev.txt")

    d, tracks, F = _load(args.npz)
    if tracks.size == 0:
        raise SystemExit(f"{args.npz} contains no track frames")

    conf = np.isin(tracks[:, F["state"]], (STATE_CONFIRMED, STATE_COASTING))
    rows = tracks[conf]
    if args.track is not None:
        rows = rows[rows[:, F["track_id"]] == args.track]
    if rows.size == 0:
        raise SystemExit("no confirmed-track frames to plot")

    ids = np.unique(rows[:, F["track_id"]])
    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)

    for tid in ids:
        r = rows[rows[:, F["track_id"]] == tid]
        t = r[:, F["t"]]
        ax[0].plot(t, r[:, F["err"]], lw=1.0, label=f"track {int(tid)}")
        ax[1].plot(t, r[:, F["nees"]], lw=1.0)
        ax[2].plot(t, r[:, F["p_manoeuvre"]], lw=1.0)

    ax[0].set_ylabel("position error (m)")
    ax[0].set_yscale("log")
    ax[0].legend(fontsize="small", ncol=4)
    ax[0].set_title(Path(args.npz).stem)

    ax[1].axhspan(NEES_LO, NEES_HI, color="0.85",
                  label="95% consistency band (6 DOF)")
    ax[1].axhline(6.0, color="0.4", ls="--", lw=0.8, label="ideal (state dim)")
    ax[1].set_ylabel("NEES")
    ax[1].set_yscale("log")
    ax[1].legend(fontsize="small")

    ax[2].set_ylabel("P(manoeuvring)")
    ax[2].set_ylim(-0.05, 1.05)
    ax[2].set_xlabel("time (s)")

    for a in ax:
        a.grid(alpha=0.3)
    fig.tight_layout()

    if args.out:
        fig.savefig(args.out, dpi=130)
        print(f"wrote {args.out}")
    else:
        plt.show()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
