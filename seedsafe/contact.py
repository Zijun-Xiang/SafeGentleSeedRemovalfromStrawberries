"""Tip-level contact model and damage measure.

MuJoCo simulates the arm, the hand, the elastic tweezers and the finger-tweezer
contacts. The millimetre-scale contacts at the two tweezer tips are computed
here, because the safe set is defined on exactly these quantities:

  tip <-> flesh : Hunt-Crossley normal force  F = k * delta^1.5 * (1 + d * delta_dot)
                  plus a shear spring that slides at mu * F   (delta = indentation)
  tip <-> seed  : stiff penalty contact against a 1 mm sphere
  seed <-> pit  : spring that breaks at the pull-out force

Forces are applied to the tweezer arms and, with opposite sign, to the berry.
"""
import numpy as np
import mujoco

from .berry import ellipsoid_sdf

ATTACHED, HELD, COLLECTED, LOST = 0, 1, 2, 3


class MicroContact:
    def __init__(self, model, data, cfg):
        self.m, self.d, self.cfg = model, data, cfg
        self.tip_sid = [model.site("tip_a").id, model.site("tip_b").id]
        self.arm_bid = [model.body("arm_a").id, model.body("arm_b").id]
        self.berry_bid = model.body("berry").id
        self.seed_mocap = [model.body(f"seed_{i}").mocapid[0] for i in range(cfg.berry.max_targets)]
        self.r_tip = cfg.tweezers.tip_radius
        self.pad_half = cfg.tweezers.pad_half_width
        self.tcp_sid = model.site("tcp").id
        self.gauge = False            # calibration block between the tips
        self._vel = np.zeros(6)
        self.reset(np.array(cfg.berry.semi_axes), np.zeros((0, 3)), np.zeros((0, 3)), cfg.berry)

    # ------------------------------------------------------------------
    def reset(self, semi_axes, seeds_local, seed_normals, bp, pull_forces=None):
        """seeds_local: seed surface points in the berry frame; the seed centre
        sits `seed_depth` below the surface along the normal."""
        self.a = np.asarray(semi_axes, float)
        self.bp = bp
        self.n_seed = len(seeds_local)
        self.seat = np.asarray(seeds_local, float) - bp.seed_depth * np.asarray(seed_normals, float)
        self.seed_n = np.asarray(seed_normals, float)
        self.state = np.full(self.n_seed, ATTACHED)
        self.pull = np.full(self.n_seed, bp.pull_force) if pull_forces is None else np.asarray(pull_forces, float)
        self.gripped = -1                      # index of the seed currently gripped in its pit
        self.grip_mid0 = np.zeros(3)
        self.u = np.zeros(3)                   # displacement of the gripped seed from its seat
        self.held = -1                         # index of the seed carried by the tweezers
        self.held_off = np.zeros(3)
        self.anchor = [None, None]             # shear spring anchors (berry frame)
        self.n_collected = 0
        # live values (per tip) -- the "ground truth" the estimators try to match
        self.indent = np.zeros(2); self.f_n = np.zeros(2); self.f_t = np.zeros(2)
        self.f_seed = np.zeros(2)              # squeeze force of each tip on a seed
        self.f_tip_world = np.zeros((2, 3))    # total external force on each tip
        self.reset_stats()
        self.marks = []                        # places with visible damage (berry frame)

    def reset_stats(self):
        self.peak = dict(indent=0.0, force=0.0, shear=0.0, squeeze=0.0)

    def seed_world(self, j):
        if self.state[j] == ATTACHED:
            c = self.d.xpos[self.berry_bid] + self.seat[j]
            return c + self.u if j == self.gripped else c
        return self.d.mocap_pos[self.seed_mocap[j]].copy()

    # ------------------------------------------------------------------
    def _apply(self, bid, point, force):
        self.d.xfrc_applied[bid, :3] += force
        self.d.xfrc_applied[bid, 3:] += np.cross(point - self.d.xipos[bid], force)

    def step(self):
        """Call once before every mj_step."""
        m, d, bp = self.m, self.d, self.bp
        d.xfrc_applied[self.arm_bid[0]] = 0; d.xfrc_applied[self.arm_bid[1]] = 0
        d.xfrc_applied[self.berry_bid] = 0
        cb = d.xpos[self.berry_bid]
        mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.berry_bid, self._vel, 0)
        vb = self._vel[3:].copy()
        tips = [d.site_xpos[s].copy() for s in self.tip_sid]
        vt = []
        for s in self.tip_sid:
            mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_SITE, s, self._vel, 0)
            vt.append(self._vel[3:].copy())
        mid = 0.5 * (tips[0] + tips[1])
        R_tool = d.site_xmat[self.tcp_sid].reshape(3, 3)
        y_tool = R_tool[:, 1]
        self.f_tip_world[:] = 0
        self.f_seed[:] = 0

        # ---- tip <-> flesh ------------------------------------------------
        for i in range(2):
            pl = tips[i] - cb
            dist, n = ellipsoid_sdf(pl, self.a)
            delta = self.r_tip - dist
            if delta <= 0:
                self.indent[i] = 0; self.f_n[i] = 0; self.f_t[i] = 0; self.anchor[i] = None
                continue
            vrel = vt[i] - vb
            ddot = -(vrel @ n)
            fn = bp.hertz_k * delta ** 1.5 * max(0.0, 1.0 + bp.hunt_d * ddot)
            if self.anchor[i] is None:
                self.anchor[i] = pl.copy()
            s = pl - self.anchor[i]
            s -= (s @ n) * n
            v_tan = vrel - (vrel @ n) * n
            ft = -bp.shear_k * s - 0.4 * v_tan
            ft_mag = np.linalg.norm(ft)
            cap = bp.mu_flesh * fn
            if ft_mag > cap:                       # sliding: drag the anchor along
                ft *= cap / ft_mag
                self.anchor[i] = pl + ft / bp.shear_k
                ft_mag = cap
            f = fn * n + ft
            self._apply(self.arm_bid[i], tips[i], f)
            self._apply(self.berry_bid, tips[i], -f)
            self.f_tip_world[i] += f
            self.indent[i] = delta; self.f_n[i] = fn; self.f_t[i] = ft_mag
            pk = self.peak
            if delta > pk["indent"]: pk["indent"] = delta
            if fn > pk["force"]: pk["force"] = fn
            if ft_mag > pk["shear"]: pk["shear"] = ft_mag
            if delta > self.cfg.safe.visible_indent or fn > self.cfg.safe.visible_force:
                if all(np.linalg.norm(mk - pl) > 5e-4 for mk in self.marks):
                    self.marks.append(pl.copy())           # a new visibly damaged spot

        # ---- tip <-> seed ---------------------------------------------------
        r_c = bp.seed_radius + self.r_tip
        if self.gauge:                                # rigid calibration ball between the tips
            for i in range(2):
                dv = tips[i] - mid; dist = np.linalg.norm(dv) + 1e-12
                pen = r_c - dist
                if pen > 0:
                    f = bp.seed_k * pen * dv / dist
                    self._apply(self.arm_bid[i], tips[i], f)
                    self.f_seed[i] = bp.seed_k * pen
        cand = [self.held] if self.held >= 0 else [
            j for j in range(self.n_seed)
            if self.state[j] == ATTACHED and np.linalg.norm(cb + self.seat[j] - mid) < 0.004]
        for j in cand:
            c = mid + R_tool @ self.held_off if j == self.held else self.seed_world(j)
            nrm, lat, ok = np.zeros(2), np.zeros(2), True
            for i in range(2):
                # the tip is a flat pad: a short segment along the tool's y axis
                tpad = float(np.clip((c - tips[i]) @ y_tool, -self.pad_half, self.pad_half))
                q = tips[i] + tpad * y_tool
                dv = q - c; dist = np.linalg.norm(dv) + 1e-12
                pen = r_c - dist
                if pen <= 0:
                    ok = False
                    continue
                e = dv / dist
                fmag = bp.seed_k * pen * max(0.0, 1.0 - 5.0 * ((vt[i] - vb) @ e))
                f = fmag * e
                self._apply(self.arm_bid[i], q, f)
                if j != self.held:
                    self._apply(self.berry_bid, c, -f)
                self.f_tip_world[i] += f
                nrm[i] = fmag; lat[i] = e @ self.seed_n[j]
                if fmag > self.f_seed[i]: self.f_seed[i] = fmag
                if j != self.held:                       # pressing a seed into its pit loads the flesh too
                    inward = -(f @ self.seed_n[j])
                    if inward > self.f_n[i]: self.f_n[i] = inward
                    if inward > self.peak["force"]: self.peak["force"] = inward
            if j == self.held:
                d.mocap_pos[self.seed_mocap[j]] = c
                if not ok and nrm.sum() == 0:           # jaws opened: drop the seed
                    self._release(j, mid)
                continue
            good = ok and nrm.min() > 0.01 and np.all(np.abs(lat) < bp.grip_cone)
            if good and self.gripped < 0:
                self.gripped = j; self.grip_mid0 = mid - cb; self.u = np.zeros(3)
            if j == self.gripped:
                if not good:
                    self.gripped = -1; self.u = np.zeros(3)
                    continue
                self.u = (mid - cb) - self.grip_mid0
                f_seat = -bp.seat_k * self.u            # pit pulls the seed (and the tips) back
                cap = bp.mu_seed * nrm.sum()
                if np.linalg.norm(f_seat) > cap:        # seed slips out of the jaws
                    self.gripped = -1; self.u = np.zeros(3)
                    continue
                for i in range(2):
                    self._apply(self.arm_bid[i], tips[i], 0.5 * f_seat)
                    self.f_tip_world[i] += 0.5 * f_seat
                self._apply(self.berry_bid, c, -f_seat)
                if np.linalg.norm(f_seat) >= self.pull[j]:   # pulled out of the pit
                    self.state[j] = HELD; self.held = j; self.gripped = -1; self.u = np.zeros(3)
                    self.held_off = R_tool.T @ (c - mid)        # where the seed sits between the pads
                    self.held_off[0] = 0.0                      # free seed: equal force from both pads
        sq = self.f_seed.min()                           # force passed through the seed, jaw to jaw
        if sq > self.peak["squeeze"]: self.peak["squeeze"] = sq
        for j in range(self.n_seed):                     # keep seed markers on the berry
            if self.state[j] == ATTACHED:
                d.mocap_pos[self.seed_mocap[j]] = self.seed_world(j) + (bp.seed_depth - 0.3 * bp.seed_radius) * self.seed_n[j]

    def _release(self, j, pos):
        tray = np.array(self.cfg.tray_pos)
        self.held = -1
        if np.linalg.norm(pos[:2] - tray[:2]) < 0.02 and pos[2] > tray[2] - 0.002:
            self.state[j] = COLLECTED
            k = self.n_collected; self.n_collected += 1
            ang = 2.4 * k
            self.d.mocap_pos[self.seed_mocap[j]] = tray + np.array(
                [0.004 * np.sqrt(k) * np.cos(ang), 0.004 * np.sqrt(k) * np.sin(ang), 0.0006])
        else:
            self.state[j] = LOST
            self.d.mocap_pos[self.seed_mocap[j]] = np.array([pos[0], pos[1], 0.001])
