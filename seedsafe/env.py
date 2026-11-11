"""Simulation environment: scene + low-level control + sensors.

The planner / safety filter talk to the robot through a small interface:

    env.command(v_tip, closure_rate)   desired tip velocity [m/s] and jaw closure rate [1/s]
    env.tick()                         advance one control period (5 ms)
    env.obs                            what the robot can measure (noisy)
    env.truth()                        ground truth, for evaluation only
"""
import copy

import mujoco
import numpy as np

from . import berry as berry_geom
from .config import Config
from .contact import ATTACHED, COLLECTED, HELD, LOST, MicroContact
from .scene import N_DECOR, build_model

ARM_HOME = np.array([-0.068, -0.099, 0.181, 1.712, 0.375, 0.599, -1.02])
_MODEL_CACHE = {}


def rotation_error(R_des, R):
    """Rotation vector (world frame) that turns R into R_des."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, (R_des @ R.T).flatten())
    if q[0] < 0:
        q = -q
    v = np.zeros(3)
    mujoco.mju_quat2Vel(v, q, 1.0)
    return v


def frame_from_axis(z, x_hint=(0.0, 1.0, 0.0)):
    """Right-handed frame whose z axis is `z` and whose x axis is close to `x_hint`."""
    z = np.asarray(z, float) / np.linalg.norm(z)
    x = np.asarray(x_hint, float) - (np.asarray(x_hint, float) @ z) * z
    x /= np.linalg.norm(x)
    return np.column_stack([x, np.cross(z, x), z])


class SeedEnv:
    def __init__(self, cfg: Config = None):
        self.cfg = copy.deepcopy(cfg) if cfg is not None else Config()
        key = repr((self.cfg.tweezers, self.cfg.berry.max_targets, self.cfg.berry.center, self.cfg.tray_pos,
                    self.cfg.berry.cradle_k, self.cfg.berry.mass, self.cfg.control.dt, self.cfg.sensors.cam_fovy))
        if key not in _MODEL_CACHE:
            _MODEL_CACHE[key] = build_model(self.cfg)
        m, self.grasp = _MODEL_CACHE[key]
        self.m = copy.copy(m)
        self.d = mujoco.MjData(self.m)
        m = self.m
        self.arm_q = np.array([m.joint(f"joint{i}").qposadr[0] for i in range(1, 8)])
        self.arm_v = np.array([m.joint(f"joint{i}").dofadr[0] for i in range(1, 8)])
        self.arm_u = np.array([m.actuator(f"joint{i}").id for i in range(1, 8)])
        rng_ = m.jnt_range[[m.joint(f"joint{i}").id for i in range(1, 8)]]
        self.arm_lo, self.arm_hi = rng_[:, 0] + 0.03, rng_[:, 1] - 0.03
        self.arm_kp = m.actuator_gainprm[self.arm_u, 0].copy()
        self.arm_kv = -m.actuator_biasprm[self.arm_u, 2].copy()
        fj = [f"right_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)]
        self.fin_q = np.array([m.joint(n).qposadr[0] for n in fj])
        self.fin_u = np.array([m.actuator(n + "_actuator").id for n in fj])
        self.jaw_q = np.array([m.joint("jaw_a").qposadr[0], m.joint("jaw_b").qposadr[0]])
        self.seat_q = np.array([m.joint(n).qposadr[0] for n in
                                ("seat_tx", "seat_ty", "seat_tz", "seat_rx", "seat_ry", "seat_rz")])
        self.berry_q = np.array([m.joint(f"berry_{a}").qposadr[0] for a in "xyz"])
        self.sid_tcp = m.site("tcp").id
        self.sid_tip = [m.site("tip_a").id, m.site("tip_b").id]
        self.bid_palm = m.body("right_palm_link").id
        self.bid_arm_ab = [m.body("arm_a").id, m.body("arm_b").id]
        self.gid_flesh = m.geom("berry_flesh").id
        self.gid_decor = np.array([m.geom(f"dseed_{i}").id for i in range(N_DECOR)])
        self.pad_sid = [m.site("pad1").id, m.site("pad2").id]
        self.bid_tip_links = [m.body("right_finger1_link4").id, m.body("right_finger2_link4").id]
        # nominal tool tip in the palm frame (what the controller believes)
        R_pt = np.zeros(9); mujoco.mju_quat2Mat(R_pt, np.array(self.grasp["tool_quat"]))
        self.R_palm_tool = R_pt.reshape(3, 3)
        self.p_palm_tcp = np.array(self.grasp["tool_pos"]) + self.R_palm_tool @ np.array(
            [0, 0, self.cfg.tweezers.length - 0.0002])
        self.c_grid = np.array(self.grasp["c_grid"])
        self.q_pinch = np.array(self.grasp["q_thumb_index"])
        self.q_wrap = np.array(self.grasp["q_wrap"])
        self.pinch_axis = np.array(self.grasp["pinch_axis"])
        self.jaw_cal = None                     # set by hand.calibrate_jaws
        self.contact = MicroContact(self.m, self.d, self.cfg)
        self._jacp = np.zeros((3, m.nv)); self._jacr = np.zeros((3, m.nv))
        self.rng = np.random.default_rng(0)
        self.dt_ctrl = self.cfg.control.dt * self.cfg.control.ctrl_decim
        self.reset(seed=0)

    # ------------------------------------------------------------------ reset
    def reset(self, seed=0, berry=None, randomize=True, settle=True):
        """New berry. `berry` overrides BerryParams fields for this episode."""
        cfg, m, d = self.cfg, self.m, self.d
        self.rng = np.random.default_rng(seed)
        rng = self.rng
        bp = copy.deepcopy(cfg.berry)
        if randomize:                           # real berries vary
            scale = rng.uniform(0.9, 1.12)
            bp.semi_axes = tuple(np.array(bp.semi_axes) * scale * rng.uniform(0.95, 1.05, 3))
            bp.hertz_k *= rng.uniform(0.65, 1.5)
            bp.pull_force *= rng.uniform(0.75, 1.3)
        for k, v in (berry or {}).items():
            setattr(bp, k, v)
        self.bp = bp
        a = np.array(bp.semi_axes)
        mujoco.mj_resetData(m, d)
        m.qpos_spring[self.seat_q] = 0.0
        # berry rests on the cradle: keep its lowest point fixed, so its top moves with size
        z_shift = a[2] - cfg.berry.semi_axes[2]
        offset = np.array([rng.normal(0, 0.0015), rng.normal(0, 0.0015), z_shift]) if randomize else np.array([0, 0, z_shift])
        m.body_pos[m.body("berry").id] = np.array(cfg.berry.center) + offset
        m.geom_size[self.gid_flesh] = a
        m.geom_rbound[self.gid_flesh] = a.max()
        pts, nrm = berry_geom.seed_layout(a, bp.n_seeds, rng)
        tgt = berry_geom.select_patch(pts, a, bp.patch_radius, bp.max_targets)
        self.seed_pts, self.seed_nrm = pts[tgt], nrm[tgt]
        others = [i for i in range(len(pts)) if i not in tgt][:N_DECOR]
        m.geom_pos[self.gid_decor] = 0.0
        m.geom_rgba[self.gid_decor, 3] = 0.0
        for k, i in enumerate(others):
            gid = self.gid_decor[k]
            m.geom_pos[gid] = pts[i] - 0.00015 * nrm[i]
            R = frame_from_axis(nrm[i], (0.0, 1.0, 0.01))
            q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten()); m.geom_quat[gid] = q
            m.geom_rgba[gid, 3] = 1.0
        pulls = bp.pull_force * rng.uniform(0.8, 1.25, len(tgt)) if randomize else None
        d.mocap_pos[:] = np.array([0, 0, -1.0])
        for k in range(len(tgt)):
            R = frame_from_axis(self.seed_nrm[k], (0.0, 1.0, 0.01))
            q = np.zeros(4); mujoco.mju_mat2Quat(q, R.flatten()); d.mocap_quat[self.contact.seed_mocap[k]] = q
        # robot: home pose, fingers holding the tweezers with jaws open
        d.qpos[self.arm_q] = ARM_HOME
        self.closure = 0.0
        d.qpos[self.fin_q] = self._finger_targets(0.0)
        theta0 = m.qpos_spring[self.jaw_q[0]]
        d.qpos[self.jaw_q] = theta0
        mujoco.mj_forward(m, d)
        self.contact.reset(a, self.seed_pts, self.seed_nrm, bp, pulls)
        # sensor state
        s = cfg.sensors
        self.tau_bias = rng.normal(0, s.torque_bias, 7) if randomize else np.zeros(7)
        self.tool_calib = rng.normal(0, s.tool_calib_err, 3) if randomize else np.zeros(3)
        self.q_cmd = ARM_HOME.copy()
        self.v_cmd = np.zeros(3); self.closure_rate = 0.0
        self.x_ref = self.tcp_est()[0].copy()
        self.R_ref = self.tcp_est()[1].copy()
        self.f_est = np.zeros(3); self.f_zero = np.zeros(3); self._f_pad = np.zeros(2)
        self.time = 0.0
        self.obs = {}
        if settle:
            for _ in range(60):
                self.tick()
            self.x_ref = self.tcp_est()[0].copy()
            self.f_zero = self.f_est.copy()      # tare the force estimate in free space
        return self.obs

    @property
    def n_targets(self):
        return len(self.seed_pts)

    # ------------------------------------------------------------- kinematics
    def tcp_est(self):
        """Tip midpoint pose the controller believes in: arm FK + calibrated tool offset.
        It does not see the tool slipping in the hand."""
        R = self.d.xmat[self.bid_palm].reshape(3, 3)
        off = self.p_palm_tcp + self.tool_calib
        if self.jaw_cal is not None:                      # calibrated uneven closing of the two arms
            off = off + self.R_palm_tool[:, 0] * self.jaw_cal.tip_shift(self.closure)
        p = self.d.xpos[self.bid_palm] + R @ off
        return p, R @ self.R_palm_tool

    def tcp_true(self):
        return 0.5 * (self.d.site_xpos[self.sid_tip[0]] + self.d.site_xpos[self.sid_tip[1]])

    def _jac_tcp(self):
        p, _ = self.tcp_est()
        mujoco.mj_jac(self.m, self.d, self._jacp, self._jacr, p, self.bid_palm)
        return self._jacp[:, self.arm_v], self._jacr[:, self.arm_v]

    def _finger_targets(self, c):
        q = np.empty(20)
        for k in range(8):
            q[k] = np.interp(c, self.c_grid, self.q_pinch[:, k])
        q[8:] = self.q_wrap
        return q

    # ---------------------------------------------------------------- control
    def command(self, v_tip, closure_rate=0.0, R_des=None):
        self.v_cmd = np.asarray(v_tip, float)
        self.closure_rate = float(closure_rate)
        if R_des is not None:
            self.R_ref = R_des

    def tick(self):
        """One control period: resolved-rate arm control + finger targets, then physics."""
        cfg, m, d = self.cfg, self.m, self.d
        dt = self.dt_ctrl
        self.x_ref = self.x_ref + self.v_cmd * dt
        p, R = self.tcp_est()
        Jp, Jr = self._jac_tcp()
        e_p = self.x_ref - p
        n_e = np.linalg.norm(e_p)
        if n_e > 0.004:                                   # never let the reference run away
            self.x_ref = p + e_p * 0.004 / n_e; e_p = self.x_ref - p
        v = self.v_cmd + cfg.control.clik_gain * e_p
        w = cfg.control.rot_gain * rotation_error(self.R_ref, R)
        w_n = np.linalg.norm(w)
        if w_n > cfg.control.w_max:                       # re-orient gently
            w *= cfg.control.w_max / w_n
        J = np.vstack([Jp, Jr])
        JJt = J @ J.T + 1e-5 * np.eye(6)
        Jpinv = J.T @ np.linalg.inv(JJt)
        dq = Jpinv @ np.r_[v, w]
        dq += (np.eye(7) - Jpinv @ J) @ (0.5 * (ARM_HOME - d.qpos[self.arm_q]))   # keep the elbow posture
        dq *= min(1.0, 1.5 / max(np.abs(dq).max(), 1e-9))    # joint speed limit, direction kept
        self.q_cmd = np.clip(self.q_cmd + dq * dt, self.arm_lo, self.arm_hi)
        d.ctrl[self.arm_u] = self.q_cmd + self.arm_kv / self.arm_kp * dq       # velocity feed-forward
        c_hi = 1.0 if self.jaw_cal is None else self.jaw_cal.c_max
        self.closure = float(np.clip(self.closure + self.closure_rate * dt, 0.0, c_hi))
        d.ctrl[self.fin_u] = self._finger_targets(self.closure)
        # physics
        tau = np.zeros(7)
        for _ in range(cfg.control.ctrl_decim):
            self.contact.step()
            mujoco.mj_step(m, d)
            tau += self._ext_torque()
        self._seat_slip()
        self.time += dt
        self._update_obs(tau / cfg.control.ctrl_decim)
        return self.obs

    # ---------------------------------------------------------------- sensing
    def _ext_torque(self):
        """Arm joint torques caused by the tip contacts (what joint torque sensors add
        on top of the robot's own dynamics model)."""
        tau = np.zeros(7)
        for i in range(2):
            f = self.contact.f_tip_world[i]
            if f[0] == 0.0 and f[1] == 0.0 and f[2] == 0.0:
                continue
            mujoco.mj_jac(self.m, self.d, self._jacp, None, self.d.site_xpos[self.sid_tip[i]], self.bid_arm_ab[i])
            tau += self._jacp[:, self.arm_v].T @ f
        return tau

    def _update_obs(self, tau_mean):
        s, rng = self.cfg.sensors, self.rng
        n = self.cfg.control.ctrl_decim
        tau_meas = tau_mean + self.tau_bias + rng.normal(0, s.torque_noise / np.sqrt(n), 7)
        Jp, _ = self._jac_tcp()
        f_raw = -np.linalg.solve(Jp @ Jp.T + 1e-9 * np.eye(3), Jp @ tau_meas)   # force the tip exerts
        a = 1.0 - np.exp(-2 * np.pi * s.force_lpf_hz * self.dt_ctrl)
        self.f_est = self.f_est + a * (f_raw - self.f_est)
        p, R = self.tcp_est()
        q_fin = self.d.qpos[self.fin_q[:8]] + rng.normal(0, s.finger_pos_noise, 8)
        f_pad = self._pad_forces() + rng.normal(0, s.fingertip_force_noise, 2)
        self._f_pad = self._f_pad + 0.3 * (f_pad - self._f_pad)                # light low-pass
        self.obs = dict(time=self.time, p=p.copy(), R=R.copy(), f=self.f_est - self.f_zero,
                        q_fin=q_fin, f_pad=self._f_pad.copy(), closure=self.closure,
                        q_arm=self.d.qpos[self.arm_q].copy())
        if self.jaw_cal is not None:
            self.obs.update(self.jaw_cal.estimate(self, q_fin, self._f_pad, self.closure))

    def tare_force(self):
        """Zero the tip force estimate (call in free space, just before an approach)."""
        self.f_zero = self.f_est.copy()
        self.obs["f"] = np.zeros(3)

    def _pad_forces(self):
        """Normal force on the thumb and index fingertips (the hand's fingertip force sensors)."""
        out = np.zeros(2)
        w = np.zeros(6)
        for i in range(self.d.ncon):
            con = self.d.contact[i]
            b = self.m.geom_bodyid[con.geom2]
            k = 0 if b == self.bid_tip_links[0] else 1 if b == self.bid_tip_links[1] else -1
            if k < 0:
                b = self.m.geom_bodyid[con.geom1]
                k = 0 if b == self.bid_tip_links[0] else 1 if b == self.bid_tip_links[1] else -1
            if k >= 0:
                mujoco.mj_contactForce(self.m, self.d, i, w)
                out[k] += w[0]
        return out

    def goto(self, p_goal, R_des=None, speed=0.06, tol=2e-4, timeout=6.0, closure_rate=0.0):
        """Scripted straight-line move of the (estimated) tip. For setup and calibration."""
        t0 = self.time
        while self.time - t0 < timeout:
            e = np.asarray(p_goal) - self.tcp_est()[0]
            dist = np.linalg.norm(e)
            rot = 0.0 if R_des is None else np.linalg.norm(rotation_error(R_des, self.tcp_est()[1]))
            if dist < tol and rot < 0.004:
                break
            v = e * min(speed / max(dist, 1e-9), 8.0)
            self.command(v, closure_rate, R_des)
            self.tick()
        self.command(np.zeros(3), 0.0)
        return self.tcp_est()[0]

    def pad_width(self, q_fin8):
        """Thumb-index pad separation along the pinch axis from (measured) finger joints."""
        d2 = self._scratch()
        d2.qpos[self.fin_q[:8]] = q_fin8
        mujoco.mj_kinematics(self.m, d2)
        Rp = d2.xmat[self.bid_palm].reshape(3, 3)
        return float((Rp.T @ (d2.site_xpos[self.pad_sid[1]] - d2.site_xpos[self.pad_sid[0]])) @ self.pinch_axis)

    def _scratch(self):
        if not hasattr(self, "_d2"):
            self._d2 = mujoco.MjData(self.m)
        return self._d2

    # ----------------------------------------------------------------- truth
    def jaw_gap(self):
        """Signed distance between the two tips along the jaw axis."""
        x_tool = self.d.site_xmat[self.sid_tcp].reshape(3, 3)[:, 0]
        return float((self.d.site_xpos[self.sid_tip[1]] - self.d.site_xpos[self.sid_tip[0]]) @ x_tool)

    def _seat_slip(self):
        """Stick-slip of the sleeve in the hand: the seat is elastic up to the friction
        limit; beyond it the rest position moves (the tool has slipped and stays there)."""
        tw, m = self.cfg.tweezers, self.m
        for sl, k, cap in ((slice(0, 3), tw.seat_stiff_lin, tw.seat_friction_lin),
                           (slice(3, 6), tw.seat_stiff_rot, tw.seat_friction_rot)):
            adr = self.seat_q[sl]
            load = k * (self.d.qpos[adr] - m.qpos_spring[adr])
            mag = np.linalg.norm(load)
            if mag > cap:
                m.qpos_spring[adr] = self.d.qpos[adr] - load / mag * cap / k

    def jaw_midpoint_offset(self):
        """Offset of the tip midpoint from the tool axis, along the jaw axis [m]."""
        th = self.d.qpos[self.jaw_q]
        return float(0.5 * self.cfg.tweezers.length * (th[1] - th[0]))

    def tool_slip(self):
        """How far the tip has moved inside the hand (sleeve sliding / twisting) [m]."""
        q = self.m.qpos_spring[self.seat_q]
        L = self.cfg.tweezers.length
        return float(np.linalg.norm(q[:3] + np.cross(q[3:], [0, 0, L])))

    def berry_center(self):
        return self.d.xpos[self.contact.berry_bid].copy()

    def seed_surface_world(self, j):
        """True surface point above seed j and its outward normal (world)."""
        return self.berry_center() + self.seed_pts[j], self.seed_nrm[j]

    def truth(self):
        c = self.contact
        return dict(indent=c.indent.copy(), f_n=c.f_n.copy(), f_t=c.f_t.copy(), f_seed=c.f_seed.copy(),
                    gap=self.jaw_gap(), slip=self.tool_slip(), state=c.state.copy(), tcp=self.tcp_true(),
                    f_total=c.f_tip_world.sum(0))

    def counts(self):
        st = self.contact.state
        return dict(removed=int(np.sum(st != ATTACHED) - (self.contact.held >= 0)),
                    collected=int(np.sum(st == COLLECTED)), lost=int(np.sum(st == LOST)), total=len(st))
