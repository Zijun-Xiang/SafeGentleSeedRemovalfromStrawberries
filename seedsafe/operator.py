"""Human-in-the-loop control.

`SimulatedOperator` is a person steering the tweezer tip with a 3-axis joystick while
looking at the berry: they pick the seed, move the tip there and press a "grip" button.
Human limits that are modelled: reaction delay, hand tremor, speed-dependent motor noise
and - most important for this task - poor depth judgement.

`TeleopPlanner` turns the operator into the same interface as the autonomous planner and
implements the authority split:

  mode "teleop"   the person commands all tip motion; gripping is automatic after the button
  mode "shared"   the person chooses the seed and steers sideways; the robot controls the
                  motion along the surface normal (depth) from touch
The safety filter is separate and can be switched on for either mode.
"""
from collections import deque

import numpy as np

from .env import frame_from_axis
from .planner import CARRY, CLOSE, DESCEND, DONE, LIFT, PULL, RELEASE, TRANSIT


class SimulatedOperator:
    def __init__(self, rng, delay=0.2, gain=3.0, v_max=0.06, v_fine=0.004, tremor=0.0004,
                 noise_frac=0.12, depth_sigma=0.00045, lateral_sigma=0.00015, dt=0.005):
        self.rng, self.gain, self.v_max, self.v_fine = rng, gain, v_max, v_fine
        self.tremor_amp, self.noise_frac = tremor, noise_frac
        self.depth_sigma, self.lateral_sigma = depth_sigma, lateral_sigma
        self.buf = deque([np.zeros(3)] * max(1, int(round(delay / dt))))
        self.tremor = np.zeros(3)

    def perceive(self, p_surf, n):
        """Where the person believes the seed surface is (good sideways, poor in depth)."""
        t1 = np.cross(n, [1.0, 0, 0]); t1 /= np.linalg.norm(t1)
        t2 = np.cross(n, t1)
        return (p_surf + self.rng.normal(0, self.lateral_sigma) * t1 + self.rng.normal(0, self.lateral_sigma) * t2
                + self.rng.normal(0, self.depth_sigma) * n)

    def steer(self, tip_seen, goal, fine=False):
        """Joystick velocity toward `goal` given where the person sees the tip."""
        e = goal - tip_seen
        v = self.gain * e
        lim = self.v_fine if fine else self.v_max
        sp = np.linalg.norm(v)
        if sp > lim:
            v *= lim / sp
        self.tremor = 0.9 * self.tremor + 0.1 * self.rng.normal(0, self.tremor_amp * 3, 3)
        v = v + self.noise_frac * np.linalg.norm(v) * self.rng.normal(0, 1, 3) + self.tremor
        self.buf.append(v)
        return self.buf.popleft()                      # reaction delay


class TeleopPlanner:
    """Same interface as SeedPlanner, but the motion commands come from a person."""

    def __init__(self, cfg, jaw_cal, operator, env, mode="teleop"):
        self.cfg, self.cal, self.op, self.env, self.mode = cfg, jaw_cal, operator, env, mode
        self.c_pre = jaw_cal.closure_for_gap(cfg.control.pre_gap)
        self.c_open = 0.2
        self.phase = DONE
        self.berry_c = self.berry_r = None
        self.request_tare = False
        self.v_human = np.zeros(3)                      # what the person asked for (for the authority metric)

    def start(self, target, attempt=0, pre_gap=None):
        c = self.cfg.control
        self.t = dict(p=np.asarray(target["p"], float), n=np.asarray(target["n"], float))
        j = target.get("id", -1)
        p_true, n_true = self.env.seed_surface_world(j) if j >= 0 else (self.t["p"], self.t["n"])
        self.seen = self.op.perceive(p_true, n_true)    # the person's own view of the seed
        self.attempt = attempt
        self.depth = c.target_indent + 0.0001 * attempt
        if pre_gap is not None:
            self.c_pre = self.cal.closure_for_gap(pre_gap)
        self.R_des = frame_from_axis(-self.t["n"], (0.0, 1.0, 0.0))
        self.phase, self.phase_t0, self.result = TRANSIT, None, None
        self._opened, self._hold = False, 0
        self.h_surf = float(self.t["p"] @ self.t["n"])

    def _set(self, ph, now):
        self.phase, self.phase_t0, self._hold = ph, now, 0

    def _travel(self, p, goal):
        if self.berry_c is None or np.linalg.norm(p[:2] - goal[:2]) < 0.012:
            return goal
        z_clear = self.berry_c[2] + self.berry_r + 0.012
        if np.linalg.norm(p[:2] - goal[:2]) > 0.004:
            if p[2] < z_clear - 0.003 and np.linalg.norm(p - goal) > 0.01:
                return np.array([p[0], p[1], z_clear])
            return np.array([goal[0], goal[1], z_clear])
        return goal

    def act(self, obs, est):
        c, cal = self.cfg.control, self.cal
        now = obs["time"]
        if self.phase_t0 is None:
            self.phase_t0 = now
        n = self.t["n"]
        tip = self.env.tcp_true()                       # the person watches the real tip
        clos = obs["closure"]
        dt_phase = now - self.phase_t0
        tray = np.array(self.cfg.tray_pos) + np.array([0, 0, 0.006])
        goal_c, fine, v_auto = self.c_pre, False, None
        if self.phase == TRANSIT:
            pre = self.seen + n * c.pre_height
            goal = self._travel(tip, pre)
            if not self._opened:
                goal_c = self.c_open
                self._opened = clos <= self.c_open + 0.01
            if np.linalg.norm(tip - pre) < 4e-4 and abs(clos - self.c_pre) < 0.01 and dt_phase > 0.3:
                self._set(DESCEND, now)
                self.request_tare = True
                self.v_human = np.zeros(3)
                return np.zeros(4), self.R_des
        if self.phase == DESCEND:
            fine = True                                 # a careful person slows down for the whole descent
            goal = self.seen - n * self.depth          # the depth the person aims for by eye
            if self.mode == "shared":                   # robot owns the normal direction (touch-referenced)
                if est.touched:
                    self.h_surf = est.h_touch
                    h_goal = self.h_surf - self.depth
                else:
                    h_goal = self.h_surf - self.depth - c.search_depth
                err = h_goal - obs["p"] @ n
                v_auto = n * np.clip(8.0 * err, -(c.v_contact if est.height < 0.003 else 0.02), 0.02)
                done = abs(err) < 4e-5
            else:
                done = abs((tip - goal) @ n) < 6e-5
            slow = np.linalg.norm(obs["v_exec"][:3]) < 1.5e-4
            self._hold = self._hold + 1 if (done or slow) else 0
            if (done and dt_phase > 0.4) or (self._hold > 120 and dt_phase > 1.5) or dt_phase > 14.0:
                self.z_grip = obs["p"] @ n
                self._set(CLOSE, now)                   # the person presses "grip"
        if self.phase == CLOSE:                         # automatic: close until the grip force is reached
            self.v_human = np.zeros(3)
            sq = obs["squeeze"]
            rate = 0.25 if sq < c.grip_force else 0.0
            if sq >= c.grip_force and dt_phase > 0.15:
                self._set(PULL, now)
            elif (clos >= cal.c_closed + 0.012 and obs["gap"] < 5e-4) or dt_phase > 6.0:
                self.result = "miss"
                self._set(LIFT, now)
            return np.r_[0, 0, 0, rate], self.R_des
        hold_rate = float(np.clip((c.grip_force - obs["squeeze"]) / cal.dsqueeze_dc, -0.02, 0.02)) * 8.0
        if self.phase in (PULL, LIFT):
            fine = dt_phase < 0.5
            goal = tip + n * 0.004 if (tip - self.seen) @ n < (c.retract if self.phase == PULL else 0.003) else tip
            if (tip - self.seen) @ n >= (c.retract if self.phase == PULL else 0.003) - 3e-4:
                if self.phase == LIFT:
                    self._set(DONE, now)
                elif obs["gap"] > 7e-4 and obs["squeeze"] > 0.04:
                    self._set(CARRY, now)
                else:
                    self.result = "lost"
                    self._set(DONE, now)
            elif dt_phase > 8.0:
                self.result = self.result or "lost"
                self._set(DONE, now)
        if self.phase == CARRY:
            goal = self._travel(tip, tray)
            if np.linalg.norm(tip - tray) < 2e-3:
                self._set(RELEASE, now)
            elif dt_phase > 20.0:
                self.result = "lost"; self._set(DONE, now)
        if self.phase == RELEASE:
            goal = tray
            if clos <= self.c_pre and dt_phase > 0.25:
                self.result = "placed"
                self._set(DONE, now)
            v = self.op.steer(tip, goal)
            self.v_human = v
            return np.r_[v, -0.6], self.R_des
        if self.phase == DONE:
            self.v_human = np.zeros(3)
            return np.zeros(4), self.R_des
        v = self.op.steer(tip, goal, fine)
        self.v_human = v.copy()
        if v_auto is not None:                          # shared control: robot replaces the normal component
            v = v - (v @ n) * n + v_auto
        if self.phase in (PULL, CARRY):
            rate = hold_rate
        else:
            rate = float(np.clip(6.0 * (goal_c - clos), -0.6, 0.6))
        return np.r_[v, rate], self.R_des
