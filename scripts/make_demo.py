"""Records the demo video: the filtered system removing seeds from one berry.

    python scripts/make_demo.py [--seed 7] [--seeds 6] [--mode filtered]

Writes demo/seed_removal_demo.mp4 and demo/seed_removal_demo.gif (plus a few key frames).
"""
import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import seedsafe  # noqa: E402,F401  (selects the rendering backend before mujoco is imported)

import numpy as np

from seedsafe import viz
from seedsafe.planner import NAMES
from seedsafe.runner import run_berry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--seeds", type=int, default=6)
    ap.add_argument("--mode", default="filtered")
    ap.add_argument("--out", default="seed_removal_demo")
    a = ap.parse_args()
    out_dir = ROOT / "demo"
    out_dir.mkdir(exist_ok=True)
    frames, keys = [], {}
    state = dict(n=0, dist=0.30, look=None)
    W, H = 960, 720
    labels = dict(transit="move to the seed", descend="descend to the flesh", close="close the jaws",
                  pull="pull the seed out", carry="carry to the tray", release="release", lift="lift (missed)",
                  done="done")

    def cb(env, st, plan, k):
        state["n"] += 1
        if state["n"] % 6:
            return
        ph = NAMES[plan.phase]
        near = ph in ("descend", "close", "pull", "lift")
        target = 0.062 if near else 0.30
        state["dist"] += 0.12 * (target - state["dist"])                 # smooth zoom
        tip = env.tcp_true()
        state["look"] = tip.copy() if state["look"] is None else state["look"] + 0.25 * (tip - state["look"])
        cam = viz.tracking_camera(env, distance=state["dist"], lookat=state["look"] + np.array([0, 0, 0.004 if near else 0.03]))
        img = viz.render(env, cam, W, H)
        small = viz.render(env, "overview", 320, 240)
        img = viz.inset(img, small, "tr")
        tr = env.truth()
        safe = env.cfg.safe
        c = env.counts()
        act = st.filt.active
        names = ["indentation", "force", "approach speed", "drag", "grip stability", "squeeze"]
        active = ", ".join(n for n, a_ in zip(names, act) if a_) or "-"
        lines = [f"seed {k + 1}: {labels[ph]}",
                 f"seeds in the tray: {c['collected']}",
                 f"safety filter limits: {active}" if env.cfg.filter.enabled or st.cfg.filter.enabled else "no safety filter"]
        bars = [("indentation", tr["indent"].max() * 1e3, safe.max_indent * 1e3, "mm"),
                ("tip force", tr["f_n"].max(), safe.max_tip_force, "N"),
                ("jaw squeeze", tr["f_seed"].min(), safe.max_squeeze, "N")]
        img = viz.hud(img, lines, bars)
        frames.append(img)
        if ph not in keys and ph in ("descend", "close", "pull", "carry"):
            keys[ph] = None
        if ph in keys and keys[ph] is None and (plan.phase_t0 is not None and env.time - plan.phase_t0 > 0.35):
            keys[ph] = img

    bounds = None
    cal = ROOT / "results" / "calibration.json"
    if cal.exists() and "filtered" in a.mode:
        import json
        bounds = json.loads(cal.read_text())["bounds"]["0.1"]
    out = run_berry(a.seed, a.mode, max_seeds=a.seeds, frame_cb=cb, bounds=bounds)
    s = out["summary"]
    print(f"removed {s['removal_rate']:.2f}, violations {s['violation_rate']:.2f}, {len(frames)} frames")
    viz.write_video(frames, out_dir / f"{a.out}.mp4", fps=30)
    viz.write_gif(frames[::5], out_dir / f"{a.out}.gif", fps=8)
    import imageio
    for name, img in keys.items():
        if img is not None:
            imageio.imwrite(out_dir / f"{a.out}_{name}.png", img)
    print("wrote", out_dir / f"{a.out}.mp4")


if __name__ == "__main__":
    main()
