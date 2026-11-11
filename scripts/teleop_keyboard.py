"""Drive the tweezers yourself (human-in-the-loop trial in the MuJoCo viewer).

    python scripts/teleop_keyboard.py [--no-filter] [--shared] [--seed 7]

Keys (click the viewer window first):
    arrow keys        move the tip sideways (x / y)
    PageUp / PageDown move the tip up / down
    G                 grip   (the robot closes the jaws until the grip force is reached)
    R                 release (open the jaws)
    T                 zero the force estimate (do this in free space before touching)
    N                 select the next seed (its surface normal is used by the safety filter)
    hold Left-Shift   fine motion (x 0.15)
The title bar text in the terminal shows indentation, force and which limits are active.

--shared   the robot controls the depth (motion along the surface normal) from touch;
           you choose the seed and steer sideways.
"""
import argparse
import os
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import seedsafe  # noqa: E402,F401  (selects the rendering backend before mujoco is imported)

import numpy as np


KEY = dict(right=262, left=263, down=264, up=265, pgup=266, pgdn=267, shift=340)


class KeyTeleop:
    """Turns key presses into a tip velocity command; testable without a window."""

    def __init__(self, speed=0.006):
        self.speed = speed
        self.v = np.zeros(3)
        self.grip = False
        self.fine = False
        self.events = []
        self.last = {}

    def key(self, code):
        now = time.time()
        if code in (KEY["right"], KEY["left"], KEY["up"], KEY["down"], KEY["pgup"], KEY["pgdn"]):
            self.last[code] = now
        elif code == ord("G"):
            self.grip = True
        elif code == ord("R"):
            self.grip = False
        elif code == KEY["shift"]:
            self.fine = not self.fine
        elif code in (ord("T"), ord("N")):
            self.events.append(chr(code))

    def velocity(self, hold=0.18):
        """Keys repeat while held; a key counts as held for `hold` seconds after its last event."""
        now = time.time()
        on = lambda k: now - self.last.get(KEY[k], -1e9) < hold
        v = np.array([float(on("up")) - float(on("down")), float(on("left")) - float(on("right")),
                      float(on("pgup")) - float(on("pgdn"))])
        return v * self.speed * (0.15 if self.fine else 1.0)


def main():
    import mujoco.viewer
    from seedsafe import perception
    from seedsafe.config import Config
    from seedsafe.env import frame_from_axis
    from seedsafe.runner import Stepper, fit_sphere, make_env
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-filter", action="store_true")
    ap.add_argument("--shared", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    cfg = Config()
    cfg.filter.enabled = not a.no_filter
    env = make_env(cfg, seed=a.seed)
    targets, _ = perception.survey(env)
    sphere = fit_sphere([t["p"] for t in targets], [t["n"] for t in targets])
    st = Stepper(env, cfg)
    k = 0
    st.est.reset(targets[k]["p"], targets[k]["n"], sphere)
    R = frame_from_axis(-targets[k]["n"], (0.0, 1.0, 0.0))
    env.goto(targets[k]["p"] + targets[k]["n"] * 0.008, R)
    env.tare_force(); st.est.armed = True
    tele = KeyTeleop()
    c_pre = env.jaw_cal.closure_for_gap(cfg.control.pre_gap)
    print(__doc__)
    with mujoco.viewer.launch_passive(env.m, env.d, key_callback=tele.key) as viewer:
        viewer.cam.lookat[:] = env.berry_center(); viewer.cam.distance = 0.16
        viewer.cam.azimuth, viewer.cam.elevation = 130, -18
        t_print = 0.0
        while viewer.is_running():
            t0 = time.time()
            for ev in tele.events:
                if ev == "T":
                    env.tare_force(); st.est.armed = True
                if ev == "N":
                    k = (k + 1) % len(targets)
                    st.est.reset(targets[k]["p"], targets[k]["n"], sphere)
                    R = frame_from_axis(-targets[k]["n"], (0.0, 1.0, 0.0))
                    print(f"seed {k + 1} selected")
            tele.events.clear()
            v = tele.velocity()
            n = st.est.n
            if a.shared and st.est.local:                 # robot owns the depth
                goal = (st.est.h_touch if st.est.touched else st.est.h_cam - cfg.control.search_depth) - cfg.control.target_indent
                err = goal - env.obs["p"] @ n
                v = v - (v @ n) * n + n * np.clip(8.0 * err, -cfg.control.v_contact, 0.02) * (1.0 if tele.grip or st.est.height < 0.004 else 0.0)
            if tele.grip:
                rate = 0.25 if env.obs["squeeze"] < cfg.control.grip_force else 0.0
            else:
                rate = float(np.clip(6.0 * (c_pre - env.closure), -0.6, 0.6))
            st.step(np.r_[v, rate], R)
            viewer.sync()
            if time.time() - t_print > 0.25:
                tr = env.truth(); t_print = time.time()
                names = ["indent", "force", "speed", "drag", "grip", "squeeze"]
                act = ",".join(nm for nm, on in zip(names, st.filt.active) if on) or "-"
                print(f"\rindent {tr['indent'].max() * 1e3:5.2f} mm | tip force {tr['f_n'].max():5.3f} N | squeeze "
                      f"{tr['f_seed'].min():5.3f} N | in tray {env.counts()['collected']} | filter: {act:28s}", end="")
            dt = env.dt_ctrl - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
