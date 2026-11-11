"""Safety filter: minimally change the nominal command so the system stays in the safe set.

    minimize   || u - u_nom ||_W^2          u = [tip velocity (3), jaw closure rate (1)]
    subject to control-barrier-function (CBF) constraints, one per limit of the safe set

Each constraint has the form  h_dot(x, u) >= -alpha * h(x)  with  h >= 0  inside the safe set.
All of them are linear in u, so the filter is a tiny quadratic program (QP), solved here
with Hildreth's method (no external solver needed).
"""
import numpy as np


def solve_qp(u_nom, w, A, b, iters=60):
    """min sum w_i (u_i - u_nom_i)^2  s.t.  A u <= b.  Hildreth's dual coordinate ascent."""
    if len(A) == 0:
        return u_nom.copy()
    A = np.asarray(A, float); b = np.asarray(b, float)
    Hinv = 1.0 / (2.0 * np.asarray(w, float))
    viol = A @ u_nom - b
    if np.all(viol <= 0):
        return u_nom.copy()
    P = (A * Hinv) @ A.T
    lam = np.zeros(len(b))
    for _ in range(iters):
        for i in range(len(b)):
            if P[i, i] < 1e-14:
                continue
            g = viol[i] - (P[i] @ lam - P[i, i] * lam[i])
            lam[i] = max(0.0, g / P[i, i])
    return u_nom - Hinv * (A.T @ lam)


def brake_speed(dist, v_touch, a_brake, delay):
    """Largest approach speed from which the tip can still slow to `v_touch` before it has
    travelled `dist`, with deceleration a_brake and a reaction delay. This is the boundary
    of the backward reachable set of {speed > v_touch at the surface} for a double integrator;
    see `reach_value` for the same set computed numerically."""
    if dist <= 0:
        return v_touch
    # solve  delay*v + (v^2 - v_touch^2) / (2a) = dist  for v
    a = a_brake
    return -a * delay + np.sqrt((a * delay) ** 2 + v_touch ** 2 + 2 * a * dist)


def reach_value(v_touch, a_brake, delay, d_max=0.01, v_max=0.08, n=201):
    """Numerical check of `brake_speed`: grid over (distance, approach speed); a state is safe
    if braking at a_brake (after the delay) brings the speed to v_touch before distance 0.
    Returns (d_grid, v_grid, safe mask)."""
    d = np.linspace(0, d_max, n); v = np.linspace(0, v_max, n)
    D, V = np.meshgrid(d, v, indexing="ij")
    stop = V * delay + np.maximum(V ** 2 - v_touch ** 2, 0.0) / (2 * a_brake)
    return d, v, stop <= D


class SafetyFilter:
    def __init__(self, cfg, jaw_cal):
        self.cfg, self.cal = cfg, jaw_cal
        self.active = np.zeros(6, bool)
        self.last = dict(u=np.zeros(4), u_nom=np.zeros(4))
        self.w = np.array([1.0, 1.0, 1.0, 1e-4])     # closure rate is in 1/s: scale to ~m/s

    def force_limit(self):
        s, f = self.cfg.safe, self.cfg.filter
        # total normal force of both tips; also keeps friction-limited shear below its limit
        return min(2 * s.max_tip_force * f.force_share, 2 * s.max_shear / 0.6)

    def filter(self, u_nom, obs, est):
        """est: ContactEstimator (already updated for this tick)."""
        cfg = self.cfg
        s, f = cfg.safe, cfg.filter
        u_nom = np.asarray(u_nom, float)
        if not f.enabled:
            self.last = dict(u=u_nom, u_nom=u_nom)
            self.active[:] = False
            return u_nom
        n = est.n
        A, b, tag = [], [], []
        in_dir = np.r_[-n, 0.0]                       # v_in = in_dir . u  (speed into the surface)
        possible = est.height < est.eps_kin              # contact cannot be excluded

        # 1) indentation:  h = max_indent - indent_ub
        h = s.max_indent - est.indent_ub
        A.append(in_dir); b.append(f.alpha * h); tag.append(0)
        # 2) normal force:  h = F_lim - (F_est + eps);  F_dot <= k_bar * v_in
        if possible or est.touched:
            hF = self.force_limit() - (est.f_n + f.eps_force)
            k_bar = 2 * 1.5 * f.k_upper * np.sqrt(max(est.indent_ub, 2.5e-5))
            A.append(in_dir); b.append(f.alpha * hF / k_bar); tag.append(1)
        # 3) approach speed (reachability): slow enough to stop once touch is felt
        gap_to_zone = est.height - est.eps_kin
        A.append(in_dir); b.append(brake_speed(gap_to_zone, f.v_touch, f.a_brake, f.delay) if not est.touched
                                   else max(f.v_touch, cfg.control.v_contact)); tag.append(2)
        # 4) no dragging: sideways speed limited while contact is possible
        if possible:
            for t in _tangent_basis(n):
                A.append(np.r_[t, 0.0]); b.append(f.v_drag); tag.append(3)
                A.append(np.r_[-t, 0.0]); b.append(f.v_drag); tag.append(3)
        # 5) grip stability: sideways force must stay below what the hand can hold
        f_lat = np.linalg.norm(est.f_lat)
        lat_max = 0.5 * cfg.tweezers.seat_friction_rot / cfg.tweezers.length
        if f_lat > 0.3 * lat_max:
            dirn = est.f_lat / f_lat
            A.append(np.r_[dirn, 0.0]); b.append(f.alpha * (lat_max - f_lat - f.eps_force) / 1000.0); tag.append(4)
        # 6) jaw squeeze:  h = max_squeeze - (squeeze_est + eps)
        hS = s.max_squeeze - (obs.get("squeeze", 0.0) + f.eps_squeeze)
        A.append(np.array([0, 0, 0, 1.0])); b.append(f.alpha * hS / max(self.cal.dsqueeze_dc, 1e-6)); tag.append(5)

        A = np.array(A); b = np.array(b)
        u = solve_qp(u_nom, self.w, A, b)
        self.active[:] = False
        act = (A @ u_nom - b) > 1e-9
        for i in np.nonzero(act)[0]:
            self.active[tag[i]] = True
        self.last = dict(u=u, u_nom=u_nom)
        return u


def _tangent_basis(n):
    a = np.array([1.0, 0, 0]) if abs(n[0]) < 0.9 else np.array([0, 1.0, 0])
    t1 = np.cross(n, a); t1 /= np.linalg.norm(t1)
    return t1, np.cross(n, t1)
