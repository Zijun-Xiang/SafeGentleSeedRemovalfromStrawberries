"""Contact state estimation at the tool tip.

The robot cannot measure tip force or indentation directly. It has
  * a tip force estimate from the arm's joint torque sensors (noisy, low-passed),
  * the tip position from arm kinematics (precise),
  * a camera estimate of where the surface is (uncertain by ~0.5 mm).
From these it keeps an *upper bound* on the indentation, which the safety filter uses.

Near the current target seed the surface is modelled as the camera's local tangent plane
(re-referenced by touch). Elsewhere a coarse sphere fitted to the berry is used.
"""
import numpy as np

R_LOCAL = 0.004        # the tangent plane is trusted within this distance of the target [m]
EPS_SPHERE = 0.0035    # error bound of the coarse sphere model [m]


class ContactEstimator:
    def __init__(self, cfg):
        self.cfg = cfg
        self.sphere = None
        self.reset(np.zeros(3), np.array([0.0, 0.0, 1.0]))

    def reset(self, p_surf, n, sphere=None):
        """New target: camera estimate of the surface point above the seed and its normal."""
        if sphere is not None:
            self.sphere = sphere
        self.p_t = np.asarray(p_surf, float)
        self.n_t = np.asarray(n, float)
        self.n = self.n_t
        self.h_cam = float(self.p_t @ self.n_t)            # surface height along n (camera)
        self.h_touch = None                                # surface height from touch
        self.touched = False
        self.armed = False                                 # touch detection starts after the force tare
        self._count = 0
        self.f_n = 0.0
        self.f_lat = np.zeros(3)
        self.height, self.indent, self.indent_ub, self.local = 1.0, -1.0, -1.0, False

    def update(self, obs):
        fp, sp = self.cfg.filter, self.cfg.berry
        f, p = obs["f"], obs["p"]
        d = p - self.p_t
        rho = np.linalg.norm(d - (d @ self.n_t) * self.n_t)
        self.local = rho < R_LOCAL
        if self.local:
            self.n = self.n_t
        elif self.sphere is not None:
            r = p - self.sphere[0]
            dist = np.linalg.norm(r)
            self.n = r / dist
        self.f_n = float(-(f @ self.n))                    # compression on the surface (+)
        self.f_lat = f + self.f_n * self.n                 # sideways force the tip exerts
        if self.local:
            h_tip = float(p @ self.n)
            if self.armed and not self.touched:
                self._count = self._count + 1 if self.f_n > fp.touch_force else 0
                if self._count >= 3:                       # touch: re-reference the surface
                    self.touched = True
                    d0 = (max(self.f_n, 0.0) / (2 * sp.hertz_k)) ** (2.0 / 3.0)
                    self.h_touch = h_tip + d0
            h_surf = self.h_touch if self.touched else self.h_cam
            self.indent = h_surf - h_tip                   # best estimate (negative = above surface)
            eps_kin = fp.eps_indent if self.touched else fp.eps_surface
        else:
            self.indent = -(dist - self.sphere[1]) if self.sphere is not None else -1.0
            eps_kin = EPS_SPHERE
        # upper bound: kinematics + its error bound, but never more than the force allows
        ub_kin = self.indent + eps_kin
        k_low = 0.55 * sp.hertz_k
        ub_force = (max(self.f_n + fp.eps_force, 0.0) / (2 * k_low)) ** (2.0 / 3.0)
        self.indent_ub = min(ub_kin, ub_force) if ub_kin > 0 else ub_kin
        self.height = -self.indent                         # tip height above the estimated surface
        self.eps_kin = eps_kin
        return self
