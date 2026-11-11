"""Nominal seed-removal planner: a phase machine whose motions come from iLQR/MPC.

Phases for one seed:
  TRANSIT  hover above the seed, jaws pre-opened, tool axis along the surface normal
  DESCEND  go down until the tips are at gripping depth beside the seed
  CLOSE    close the jaws until the squeeze force reaches the grip target
  PULL     retract along the normal
  CARRY    move over the tray
  RELEASE  open the jaws
  LIFT     after a failed grip: straight up out of the flesh before anything else
The planner knows nothing about the safe set; the safety filter sits behind it.
"""
import numpy as np

from .env import frame_from_axis
from .ilqr import ilqr

TRANSIT, DESCEND, CLOSE, PULL, CARRY, RELEASE, DONE, LIFT = range(8)
NAMES = ["transit", "descend", "close", "pull", "carry", "release", "done", "lift"]


class SeedPlanner:
    def __init__(self, cfg, jaw_cal, use_touch=True, horizon=12, dt_plan=0.04):
        self.cfg, self.cal, self.use_touch = cfg, jaw_cal, use_touch
        self.N, self.dt = horizon, dt_plan
        self.U = np.zeros((horizon, 4))
        self.c_pre = jaw_cal.closure_for_gap(cfg.control.pre_gap)
        self.phase = DONE
        self.berry_c = None                      # estimated berry centre and radius (keep-out)
        self.berry_r = None

    # ------------------------------------------------------------------
    def start(self, target, attempt=0, pre_gap=None):
        """target: dict(p=surface point above the seed, n=outward normal)."""
        c = self.cfg.control
        self.t = dict(p=np.asarray(target["p"], float), n=np.asarray(target["n"], float))
        self.attempt = attempt
        self.depth = c.target_indent + 0.0001 * attempt       # retry a little deeper
        self.c_pre = self.cal.closure_for_gap(pre_gap if pre_gap is not None else c.pre_gap)
        self.R_des = frame_from_axis(-self.t["n"], (0.0, 1.0, 0.0))
        self.phase = TRANSIT
        self.phase_t0 = None
        self.result = None
        self.U[:] = 0
        self.h_surf = float(self.t["p"] @ self.t["n"])
        self._hold = 0
        self._u = None
        self._tick = 0
        self.request_tare = False
        self._opened = False
        self.c_open = 0.2                                     # fingers just clear of the tweezer arms

    def _set_phase(self, ph, now):
        self.phase, self.phase_t0 = ph, now
        self.U[:] = 0
        self._hold = 0
        self._u = None

    # ------------------------------------------------------------------
    def act(self, obs, est):
        """Returns (u_nom = [v(3), closure rate], R_des)."""
        c, cal = self.cfg.control, self.cal
        now = obs["time"]
        if self.phase_t0 is None:
            self.phase_t0 = now
        p, n, ps = obs["p"], self.t["n"], self.t["p"]
        clos = obs["closure"]
        dt_phase = now - self.phase_t0
        if self.use_touch and est.touched:                    # surface height from touch
            self.h_surf = est.h_touch
        v_lim, keep_out = c.v_max, True
        goal_c = self.c_pre
        tray = np.array(self.cfg.tray_pos) + np.array([0, 0, 0.006])

        if self.phase == TRANSIT:
            pre = ps + n * c.pre_height
            goal_p = self._travel(p, pre)
            # always reach the pre-opening from the open side: the fingers have some hysteresis,
            # and the calibration sweep was recorded while closing
            if not self._opened:
                goal_c = self.c_open
                self._opened = clos <= self.c_open + 0.01
            if np.linalg.norm(p - pre) < 3e-4 and abs(clos - goal_c) < 0.01 and dt_phase > 0.2:
                self._set_phase(DESCEND, now)
                self.request_tare = True                       # zero the force estimate in free space
                return np.zeros(4), self.R_des
        if self.phase == DESCEND:
            keep_out = False
            v_lim = c.v_contact if est.height < 0.003 else 0.02
            search = self.use_touch and not est.touched        # keep going down until touch is felt
            h_goal = self.h_surf - self.depth - (c.search_depth if search else 0.0)
            goal_p = ps + n * (h_goal - ps @ n)
            reached = abs(p @ n - h_goal) < 4e-5
            self._hold = self._hold + 1 if (reached or np.linalg.norm(obs["v_exec"][:3]) < 1.5e-4) else 0
            if (reached and dt_phase > 0.3) or (self._hold > 80 and dt_phase > 1.0) or dt_phase > 12.0:
                self.z_grip = p @ n
                self._set_phase(CLOSE, now)
        if self.phase == CLOSE:
            keep_out = False
            goal_p = ps + n * (self.z_grip - ps @ n)
            sq = obs["squeeze"]
            goal_c = clos + 0.06 if sq < c.grip_force else clos
            if sq >= c.grip_force and dt_phase > 0.15:
                self._set_phase(PULL, now)
            elif clos >= cal.c_closed + 0.012 and obs["gap"] < 5e-4 or dt_phase > 6.0:
                self.result = "miss"                           # jaws shut on nothing
                self._set_phase(LIFT, now)
        if self.phase == LIFT:
            keep_out = False
            v_lim = 0.006
            goal_p = ps + n * (self.z_grip - ps @ n + 0.003)
            goal_c = clos
            if (p - goal_p) @ n > -3e-4:
                self._set_phase(DONE, now)
        if self.phase == PULL:
            keep_out = False
            v_lim = 0.004 if dt_phase < 0.4 else 0.03
            goal_p = ps + n * (self.z_grip - ps @ n + c.retract)
            goal_c = clos + np.clip((c.grip_force - obs["squeeze"]) / cal.dsqueeze_dc, -0.02, 0.02)
            if np.linalg.norm(p - goal_p) < 5e-4:
                if obs["gap"] > 7e-4 and obs["squeeze"] > 0.04:
                    self._set_phase(CARRY, now)
                else:
                    self.result = "lost"                        # the seed slipped out of the jaws
                    self._set_phase(DONE, now)
            elif dt_phase > 8.0:
                self.result = "lost"
                self._set_phase(DONE, now)
        if self.phase == CARRY:
            goal_p = self._travel(p, tray)
            goal_c = clos + np.clip((c.grip_force - obs["squeeze"]) / cal.dsqueeze_dc, -0.02, 0.02)
            if np.linalg.norm(p - goal_p) < 1.5e-3:
                self._set_phase(RELEASE, now)
        if self.phase == RELEASE:
            goal_p = tray
            goal_c = self.c_open
            if clos <= self.c_pre and dt_phase > 0.25:
                self.result = "placed"
                self._set_phase(DONE, now)
        if self.phase == DONE:
            return np.zeros(4), self.R_des
        self._tick += 1
        if self._tick % c.plan_decim == 1 or self._u is None:     # replan at 50 Hz
            self._u = self._mpc(p, clos, goal_p, goal_c, v_lim, keep_out)
        return self._u.copy(), self.R_des

    def _travel(self, p, goal):
        """Long moves go up, across at a clearance height, then down (never through the berry)."""
        if self.berry_c is None or np.linalg.norm(p[:2] - goal[:2]) < 0.012:
            return goal
        z_clear = self.berry_c[2] + self.berry_r + 0.012
        if np.linalg.norm(p[:2] - goal[:2]) > 0.004:
            if p[2] < z_clear - 0.003 and np.linalg.norm(p - goal) > 0.01:
                return np.array([p[0], p[1], z_clear])             # rise
            return np.array([goal[0], goal[1], z_clear])           # traverse
        return goal

    # ------------------------------------------------------------------
    def _mpc(self, p, clos, goal_p, goal_c, v_lim, keep_out):
        c = self.cfg.control
        x0 = np.r_[p, clos]
        e = goal_p - p                                   # look at most 4 cm ahead (keeps the problem well scaled)
        dist = np.linalg.norm(e)
        if dist > 0.04:
            goal_p = p + e * 0.04 / dist
        wp, wc, wu, wcu = 8.0, 0.16, 1.0, 0.02
        rate_c = c.jaw_rate_max / max(abs(self.cal.dgap_dc(0.4)), 1e-6)      # closure rate limit
        bc, br = self.berry_c, self.berry_r

        def res(X, U):
            out = [wp * (X[:, :3] - goal_p), wc * (X[:, 3:4] - goal_c), wu * U[:, :3], wcu * U[:, 3:4]]
            speed = np.sqrt(np.sum(U[:, :3] ** 2, axis=1, keepdims=True) + 1e-12)
            out.append(40.0 * np.maximum(0.0, speed - v_lim))
            out.append(1.0 * np.maximum(0.0, np.abs(U[:, 3:4]) - rate_c))
            if keep_out and bc is not None:          # stay clear of the berry between seeds
                d = np.linalg.norm(X[:, :3] - bc, axis=1, keepdims=True) - br
                out.append(60.0 * np.maximum(0.0, 0.004 - d))
            return np.hstack(out)

        def res_T(x):
            return np.r_[3 * wp * (x[:3] - goal_p), 3 * wc * (x[3] - goal_c)]

        self.U[:-1] = self.U[1:]
        self.U, _, _ = ilqr(x0, self.U, res, res_T, self.dt, iters=1)
        u = self.U[0].copy()
        sp = np.linalg.norm(u[:3])
        if sp > v_lim:
            u[:3] *= v_lim / sp
        u[3] = np.clip(u[3], -rate_c, rate_c)
        return u
