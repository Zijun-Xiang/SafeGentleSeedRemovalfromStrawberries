"""On-site calibration steps that close the sim-to-real gap.

measure_force_noise  hold still for a moment and measure how noisy the tip force estimate is;
                  the touch threshold is set from it
probe_stiffness   press the closed tweezer tips gently on the fruit (or the proxy berry) and
                  fit the flesh stiffness from the force-displacement curve
recalibrate       recompute the conformal bounds from a few calibration runs in the new domain
"""
import numpy as np

from .conformal import bounds_from_scores
from .env import frame_from_axis
from .estimation import ContactEstimator
from .safety import SafetyFilter


def probe_stiffness(env, rcfg, targets, n_spots=3, **kw):
    """Median of a few gentle presses at seed-free spots. Returns the Hertz stiffness k
    (F_total = 2 k delta^1.5), or None if no press gave a usable curve."""
    P = np.array([t["p"] for t in targets]); N = np.array([t["n"] for t in targets])
    centre, n = P.mean(axis=0), N.mean(axis=0) / np.linalg.norm(N.mean(axis=0))
    t1 = np.cross(n, [1.0, 0, 0]); t1 /= np.linalg.norm(t1); t2 = np.cross(n, t1)
    cands = [centre + r * (np.cos(a) * t1 + np.sin(a) * t2)
             for a in np.linspace(0, 2 * np.pi, 24, endpoint=False) for r in (0.0015, 0.003, 0.0045)]
    cands.sort(key=lambda q: -np.min(np.linalg.norm(P - q, axis=1)))       # far from seeds first
    spots = []
    for q in cands:
        if all(np.linalg.norm(q - s) > 0.002 for s in spots):
            spots.append(q)
        if len(spots) == n_spots:
            break
    ks = [k for k in (_press(env, rcfg, q, n, **kw) for q in spots) if k]
    return float(np.median(ks)) if ks else None


def _press(env, rcfg, best, n, f_stop=0.11, speed=0.0005):
    """One force-limited press with the tips closed; fits the stiffness."""
    R = frame_from_axis(-n, (0.0, 1.0, 0.0))
    env.goto(best + n * 0.006, R, speed=0.04, closure_rate=0.5)
    while env.closure < env.jaw_cal.c_closed + 0.005:               # tips together
        env.command(np.zeros(3), 0.4, R); env.tick()
    for _ in range(60):
        env.command(np.zeros(3), 0.0, R); env.tick()
    env.tare_force()
    est = ContactEstimator(rcfg); est.reset(best, n); est.armed = True
    filt = SafetyFilter(rcfg, env.jaw_cal)
    zs, fs = [], []
    for _ in range(6000):
        est.update(env.obs)
        u = filt.filter(np.r_[-n * speed, 0.0], env.obs, est)
        env.command(u[:3], 0.0, R); env.tick()
        zs.append(env.obs["p"] @ n); fs.append(est.f_n)
        if est.f_n > f_stop or (best - env.obs["p"]) @ n > 0.0025:
            break
    env.goto(best + n * 0.006, R, speed=0.01)
    env.command(np.zeros(3), -0.6, R)
    while env.closure > 0.2:
        env.tick()
    zs, fs = np.array(zs), np.array(fs)
    if (fs > 0.03).sum() < 8:
        return None
    # fit F = 2k * max(z0 - z, 0)^1.5 to the whole press (the force-free part pins the surface z0)
    best_fit = None
    for z0 in np.linspace(zs.min(), zs.min() + 0.002, 201):
        x = np.maximum(z0 - zs, 0.0) ** 1.5
        k2 = float(x @ fs / max(x @ x, 1e-30))
        res = float(np.sum((k2 * x - fs) ** 2))
        if best_fit is None or res < best_fit[0]:
            best_fit = (res, k2 / 2.0)
    return best_fit[1]


def measure_force_noise(env, seconds=1.5):
    """Standard deviation of the tip force estimate while holding still in free space [N]."""
    env.tare_force()
    f = []
    for _ in range(int(seconds / env.dt_ctrl)):
        env.tick()
        f.append(env.obs["f"].copy())
    return float(np.std(np.array(f)[:, 2]))          # along the (near-vertical) approach direction


def recalibrate(results, alpha):
    """Conformal bounds from run_berry outputs collected in the new domain."""
    from .runner import collect_scores
    return bounds_from_scores(collect_scores(results), alpha)
