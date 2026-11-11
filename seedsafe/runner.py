"""Runs seed-removal episodes and scores them against the safe set (ground truth)."""
import copy
import time as _time

import numpy as np

from .config import Config
from .contact import ATTACHED, COLLECTED
from .env import SeedEnv
from .estimation import ContactEstimator
from .hand import calibrate_jaws
from .planner import DONE, NAMES, SeedPlanner
from .safety import SafetyFilter


def make_env(cfg=None, seed=0, **reset_kw):
    env = SeedEnv(cfg)
    env.reset(seed=seed, **reset_kw)
    calibrate_jaws(env)
    return env


def true_targets(env):
    """Perfect knowledge of the seeds (used only by tests and demos)."""
    return [dict(p=env.seed_surface_world(j)[0], n=env.seed_surface_world(j)[1], id=j) for j in range(env.n_targets)]


def fit_sphere(points, normals):
    """Coarse berry model for keep-out: a sphere through the detected seed points.
    Each point votes for the centre  c = p - r * n ; the radius (12..26 mm) whose votes
    agree best is kept. Using the normals keeps the fit stable on a small surface patch."""
    P, Nn = np.asarray(points), np.asarray(normals)
    best = None
    for r in np.arange(0.012, 0.0261, 0.001):
        C = P - r * Nn
        spread = float(np.sum(np.var(C, axis=0)))
        if best is None or spread < best[0]:
            best = (spread, C.mean(axis=0), float(r))
    return best[1], best[2]


class Stepper:
    """One control tick = planner (or human) command -> estimator -> safety filter -> robot."""

    def __init__(self, env, cfg=None):
        self.env = env
        self.cfg = cfg or env.cfg
        self.est = ContactEstimator(self.cfg)
        self.filt = SafetyFilter(self.cfg, env.jaw_cal)
        self.u_exec = np.zeros(4)
        self.log = []
        self.record = False
        self.reset_stats()
        env.obs["v_exec"] = np.zeros(4)

    def reset_stats(self):
        self.n_ticks = 0
        self.interv = 0.0          # integral of |u_exec - u_nom| (tip velocity) [m]
        self.cmd = 0.0             # integral of |u_nom| [m]
        self.n_active = 0

    def step(self, u_nom, R_des):
        env = self.env
        self.est.update(env.obs)
        u = self.filt.filter(u_nom, env.obs, self.est)
        env.command(u[:3], u[3], R_des)
        obs = env.tick()
        obs["v_exec"] = u
        dt = env.dt_ctrl
        self.n_ticks += 1
        self.interv += np.linalg.norm(u[:3] - u_nom[:3]) * dt
        self.cmd += np.linalg.norm(u_nom[:3]) * dt
        self.n_active += int(self.filt.active.any())
        if self.record:
            tr = env.truth()
            self.log.append(dict(t=env.time, indent=tr["indent"].max(), f_n=tr["f_n"].max(), f_t=tr["f_t"].max(),
                                 squeeze=tr["f_seed"].min(), gap=tr["gap"], slip=tr["slip"],
                                 indent_est=self.est.indent, indent_ub=self.est.indent_ub, f_est=self.est.f_n,
                                 squeeze_est=obs.get("squeeze", 0.0), gap_est=obs.get("gap", 0.0),
                                 u_nom=np.array(u_nom), u=u.copy(), active=self.filt.active.copy(),
                                 touched=self.est.touched, p=obs["p"].copy(), tcp=tr["tcp"].copy()))
        return obs


def violations(peak, slip, safe):
    return dict(indent=peak["indent"] > safe.max_indent, force=peak["force"] > safe.max_tip_force,
                shear=peak["shear"] > safe.max_shear, squeeze=peak["squeeze"] > safe.max_squeeze,
                slip=slip > safe.max_slip)


def run_autonomous(env, targets, cfg=None, use_touch=True, frame_cb=None, record=False, pre_gaps=None,
                   max_time_per_seed=40.0, max_seeds=None):
    """Remove every target seed. Returns (per-seed records, stepper)."""
    cfg = cfg or env.cfg
    st = Stepper(env, cfg)
    st.record = record
    plan = SeedPlanner(cfg, env.jaw_cal, use_touch=use_touch)
    sphere = fit_sphere([t["p"] for t in targets], [t["n"] for t in targets]) if len(targets) >= 3 else None
    if sphere is not None:
        plan.berry_c, plan.berry_r = sphere
    records = []
    for k, tgt in enumerate(targets[:max_seeds]):
        t0 = env.time
        env.contact.reset_stats()
        st.reset_stats()
        slip0 = env.tool_slip()
        n_marks0 = len(env.contact.marks)
        collected0 = env.counts()["collected"]
        result, attempts = None, 0
        for attempt in range(cfg.control.max_attempts):
            attempts += 1
            plan.start(tgt, attempt, None if pre_gaps is None else pre_gaps[k])
            st.est.reset(tgt["p"], tgt["n"], sphere)
            while plan.phase != DONE and env.time - t0 < max_time_per_seed:
                u_nom, R_des = plan.act(env.obs, st.est)
                if plan.request_tare:
                    env.tare_force(); plan.request_tare = False
                st.step(u_nom, R_des)
                if frame_cb is not None:
                    frame_cb(env, st, plan, k)
            result = plan.result or "timeout"
            if result in ("placed", "timeout"):
                break
        j = tgt.get("id", -1)
        peak = dict(env.contact.peak)
        slip = env.tool_slip()
        rec = dict(seed=k, id=j, result=result, attempts=attempts, time=env.time - t0, **peak, slip=slip,
                   slip_inc=slip - slip0, visible=len(env.contact.marks) - n_marks0,
                   collected=env.counts()["collected"] - collected0,
                   removed=bool(j >= 0 and env.contact.state[j] != ATTACHED),
                   interv=st.interv, cmd=st.cmd, active_frac=st.n_active / max(st.n_ticks, 1))
        rec["viol"] = violations(peak, slip, cfg.safe)
        rec["any_viol"] = bool(any(rec["viol"].values()))
        records.append(rec)
    return records, st


def summarize(records):
    """Episode-level metrics from per-seed records."""
    n = max(len(records), 1)
    keys = ("indent", "force", "shear", "squeeze", "slip")
    out = dict(
        n_seeds=len(records),
        removal_rate=sum(r["collected"] > 0 for r in records) / n,
        violation_rate=sum(r["any_viol"] for r in records) / n,
        visible_damage=sum(r["visible"] for r in records),
        time_per_seed=float(np.mean([r["time"] for r in records])) if records else 0.0,
        attempts=float(np.mean([r["attempts"] for r in records])) if records else 0.0,
        peak_force=float(max([r["force"] for r in records], default=0.0)),
        peak_indent=float(max([r["indent"] for r in records], default=0.0)),
        peak_squeeze=float(max([r["squeeze"] for r in records], default=0.0)),
        slip=float(max([r["slip"] for r in records], default=0.0)),
        intervention=float(sum(r["interv"] for r in records) / max(sum(r["cmd"] for r in records), 1e-9)),
    )
    for k in keys:
        out["viol_" + k] = sum(r["viol"][k] for r in records) / n
    return out
