"""Tweezer calibration: what the finger joints tell us about the jaws.

Run once per session, away from the berry (the proposal's first hardware step):

  1. free sweep  - close the jaws slowly with nothing in them and record, for every
                   closure command c, the jaw gap and the thumb-index pad distance
                   computed from the finger encoders;
     The sideways drift of the tip midpoint is recorded too (the arms close unevenly).
  2. gauge sweep - close on a rigid 1 mm gauge ball and record the squeeze force.

After that the robot estimates, from finger encoders and fingertip force sensors:
  gap      jaw opening at the tips            (finger encoders: pad distance -> gap)
  squeeze  force on whatever blocks the jaws  (fingertip force above the free sweep)
"""
import numpy as np


class JawCalibration:
    def __init__(self, c, gap, padw, tau, c_closed, shift=None):
        self.c, self.gap, self.padw = np.asarray(c), np.asarray(gap), np.asarray(padw)
        self.shift = np.zeros(len(self.c)) if shift is None else np.asarray(shift)   # tip midpoint offset along the jaw axis
        self.tau = np.asarray(tau)        # thumb / index fingertip force in the free sweep (N x 2)
        self.c_block = 0.0                # closure at which the gauge ball is first touched
        self.w_squeeze = np.zeros(2)      # squeeze force = w . (fingertip force - free force)
        self.c_closed = c_closed          # closure at which the empty jaws meet
        self.c_max = c_closed + 0.025
        self.gauge_err = 0.0              # rms error of the squeeze estimate on the gauge [N]
        self.dsqueeze_dc = 0.0            # squeeze force per unit closure once blocked [N]
        o = np.argsort(self.padw)
        self._pw_sorted, self._gap_by_pw = self.padw[o], self.gap[o]

    def tip_shift(self, c):
        """The two arms do not close perfectly symmetrically: sideways offset of the tip midpoint [m]."""
        return float(np.interp(c, self.c, self.shift))

    def tau_free(self, c):
        return np.array([np.interp(c, self.c, self.tau[:, k]) for k in range(2)])

    def gap_free(self, c):
        return float(np.interp(c, self.c, self.gap))

    def padw_free(self, c):
        return float(np.interp(c, self.c, self.padw))

    def closure_for_gap(self, gap):
        return float(np.interp(gap, self.gap[::-1], self.c[::-1]))

    def dgap_dc(self, c):
        h = 0.01
        return (self.gap_free(c + h) - self.gap_free(c - h)) / (2 * h)

    def dpadw_dc(self, c):
        h = 0.01
        return (self.padw_free(c + h) - self.padw_free(c - h)) / (2 * h)

    def estimate(self, env, q_fin, f_pad, closure):
        pw = env.pad_width(q_fin)
        gap = float(np.interp(pw, self._pw_sorted, self._gap_by_pw))
        # fingertip force beyond what bending the tweezers to this gap needs = force on the object
        c_ref = min(closure, self.closure_for_gap(gap))
        squeeze = max(0.0, float(self.w_squeeze @ (f_pad - self.tau_free(c_ref))))
        return dict(gap=gap, squeeze=squeeze, padw=pw)


def calibrate_jaws(env, verbose=False):
    """Runs both sweeps in the simulator and attaches the result to env.jaw_cal."""
    env.jaw_cal = None
    env.command(np.zeros(3), 0.0)
    env.closure = 0.0
    for _ in range(40):
        env.tick()
    rate = 0.25
    cs, gaps, pws, taus, shifts = [], [], [], [], []
    c_closed = None
    while env.closure < 0.9:
        env.command(np.zeros(3), rate)
        env.tick()
        g = env.jaw_gap()
        cs.append(env.closure); gaps.append(max(g, 0.0)); pws.append(env.pad_width(env.obs["q_fin"]))
        taus.append(env.obs["f_pad"])
        shifts.append(env.jaw_midpoint_offset())      # on hardware: seen by the wrist camera
        if c_closed is None and g < 5e-5:
            c_closed = env.closure
        if c_closed is not None and env.closure > c_closed + 0.03:
            break
    cs, gaps, pws, taus = np.array(cs), np.array(gaps), np.array(pws), np.array(taus)
    k = 21                                           # smooth sensor noise
    ker = np.ones(k) / k
    pad = lambda x: np.convolve(np.r_[np.full(k // 2, x[0]), x, np.full(k // 2, x[-1])], ker, "valid")
    gaps_s, pws_s = np.maximum.accumulate(pad(gaps)[::-1])[::-1], np.maximum.accumulate(pad(pws)[::-1])[::-1]
    gaps_s = gaps_s + np.linspace(1e-9, 0, len(gaps_s))      # strictly decreasing for the lookups
    pws_s = pws_s + np.linspace(1e-9, 0, len(pws_s))
    taus_s = np.column_stack([pad(taus[:, j]) for j in range(2)])
    cal = JawCalibration(cs, gaps_s, pws_s, taus_s, c_closed, pad(np.array(shifts)))
    # gauge sweep
    env.command(np.zeros(3), -1.0)
    while env.closure > 0.2:
        env.tick()
    env.contact.gauge = True
    r_c = env.cfg.berry.seed_radius + env.cfg.tweezers.tip_radius
    c_block = cal.closure_for_gap(2 * r_c)
    X, force, cc, gg = [], [], [], []
    env.command(np.zeros(3), rate * 0.5)
    while env.closure < cal.c_max:
        env.tick()
        if env.closure > c_block - 0.05:
            g_est = float(np.interp(env.pad_width(env.obs["q_fin"]), cal._pw_sorted, cal._gap_by_pw))
            X.append(env.obs["f_pad"] - cal.tau_free(min(env.closure, cal.closure_for_gap(g_est))))
            gg.append(g_est)
            force.append(env.contact.f_seed.mean()); cc.append(env.closure)
    X, force, cc = np.array(X), np.array(force), np.array(cc)
    x1 = X.mean(1)                                   # mean extra fingertip force
    cal.w_squeeze = np.full(2, 0.5 * float(x1 @ force / (x1 @ x1)))
    est = np.maximum(0.0, X @ cal.w_squeeze)
    cal.gauge_err = float(np.sqrt(np.mean((est - force) ** 2)))
    on = force > 0.02
    cal.dsqueeze_dc = float(np.polyfit(cc[on], force[on], 1)[0])
    cal.c_block = c_block
    cal.max_squeeze = float(np.median(force[-10:]))
    env.contact.gauge = False
    env.command(np.zeros(3), -1.0)
    while env.closure > 0.0:
        env.tick()
    env.command(np.zeros(3), 0.0)
    env.jaw_cal = cal
    if verbose:
        print(f"jaws meet at closure {c_closed:.3f}; gauge touched at {c_block:.3f}; "
              f"squeeze at max closure {cal.max_squeeze:.3f} N ({cal.dsqueeze_dc:.2f} N per unit closure); "
              f"squeeze estimate rms error on the gauge {cal.gauge_err:.4f} N")
    return cal, dict(c=cs, gap=gaps, padw=pws, c_gauge=cc, force=force, force_est=est, gap_gauge=np.array(gg))
