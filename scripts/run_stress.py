"""Stress test: what happens when the nominal planner is badly tuned?

The planner is told to go twice as deep, approach four times faster and squeeze 2.5 times
harder than it should. Compared: the planner alone vs the same planner behind the safety filter.

    python scripts/run_stress.py [--berries 8] [--seeds 8] [--workers 2]
"""
import argparse
import os
import pathlib
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import run_experiments as X
from seedsafe.config import Config


def aggressive():
    cfg = Config()
    cfg.control.target_indent = 0.0008
    cfg.control.v_contact = 0.016
    cfg.control.grip_force = 0.60
    return cfg


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--berries", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    b = X.load("calibration")["bounds"]["0.1"]
    jobs = []
    for mode, tag in (("touch", "mis-tuned planner"), ("filtered", "mis-tuned planner + filter")):
        for i in range(a.berries):
            jobs.append(dict(tag=tag, seed=i, mode=mode, max_seeds=a.seeds, robot_cfg=aggressive(),
                             bounds=b if mode == "filtered" else None))
    X.save("e1b_stress", dict(results=X.run_jobs(jobs, a.workers), bounds=b))
