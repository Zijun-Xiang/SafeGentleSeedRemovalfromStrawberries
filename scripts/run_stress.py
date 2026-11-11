"""Stress test: what happens when the nominal planner is badly tuned?

The planner is told to go twice as deep (0.8 mm instead of 0.4 mm) and to approach four
times faster (16 mm/s instead of 4 mm/s) than it should. Compared: the planner alone vs the same planner behind the safety filter.

    python scripts/run_stress.py [--berries 8] [--seeds 8] [--workers 2]
"""
import argparse
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import seedsafe  # noqa: E402,F401  (selects the rendering backend before mujoco is imported)
sys.path.insert(0, str(ROOT / "scripts"))

import run_experiments as X
from seedsafe.config import Config


def aggressive():
    cfg = Config()
    cfg.control.target_indent = 0.0008
    cfg.control.v_contact = 0.016
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--berries", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    bounds = X.load("calibration")["bounds"]
    b = bounds["0.1"]
    jobs = []
    for mode, tag in (("touch", "mis-tuned planner"), ("filtered", "mis-tuned planner + filter")):
        for i in range(a.berries):
            jobs.append(dict(tag=tag, seed=i, mode=mode, max_seeds=a.seeds, robot_cfg=aggressive(),
                             bounds=b if mode == "filtered" else None))
    # how conservative must the filter be? Here the filter (not the planner) decides the depth,
    # so the size of the uncertainty margins matters.
    zero = dict(surface=0.0, indent=0.0, force=0.0, squeeze=0.0, lateral=b["lateral"])
    wide = {k: (v if k == "lateral" else 3.0 * v) for k, v in bounds["0.01"].items()}
    variants = [("margins: none", zero), ("margins: 50%", bounds["0.5"]), ("margins: 99%", bounds["0.01"]),
                ("margins: 3 x 99%", wide)]
    for tag, bb in variants:
        for i in range(a.berries):
            jobs.append(dict(tag=tag, seed=i, mode="filtered", max_seeds=a.seeds, robot_cfg=aggressive(), bounds=bb))
    X.save("e1b_stress", dict(results=X.run_jobs(jobs, a.workers), bounds=b,
                              variants={t: bb for t, bb in variants}))
