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
    env.contact.reset_stats()
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
        self.u_last = np.zeros(4)
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
        self.u_last = u
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
                   max_time_per_seed=40.0, max_seeds=None, planner=None):
    """Remove every target seed. Returns (per-seed records, stepper).
    `cfg` is what the robot believes (it may differ from the true physics in env.cfg).
    `planner` replaces the autonomous planner (used for human-in-the-loop control)."""
    cfg = cfg or env.cfg
    st = Stepper(env, cfg)
    st.record = record
    plan = planner or SeedPlanner(cfg, env.jaw_cal, use_touch=use_touch)
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
        sc = dict(surface=None, indent=-1.0, force=-1.0, squeeze=-1.0)
        human_cmd = human_dev = 0.0
        for attempt in range(cfg.control.max_attempts):
            attempts += 1
            plan.start(tgt, attempt, None if pre_gaps is None else pre_gaps[k])
            st.est.reset(tgt["p"], tgt["n"], sphere)
            while plan.phase != DONE and env.time - t0 < max_time_per_seed:
                u_nom, R_des = plan.act(env.obs, st.est)
                if plan.request_tare:
                    env.tare_force(); plan.request_tare = False
                    st.est.armed = True
                st.step(u_nom, R_des)
                # calibration scores (estimate errors against ground truth), one per seed
                c = env.contact
                if st.est.local:
                    ind_true, f_true = c.indent.max(), c.f_n.sum()
                    if sc["surface"] is None and ind_true > 0 and not st.est.touched:
                        sc["surface"] = st.est.height                  # robot thought it was this far above
                    if st.est.touched and ind_true > 0:
                        sc["indent"] = max(sc["indent"], ind_true - st.est.indent)
                    if ind_true > 0:
                        sc["force"] = max(sc["force"], f_true - st.est.f_n)
                    if c.f_seed.min() > 0.02:
                        sc["squeeze"] = max(sc["squeeze"], c.f_seed.min() - env.obs["squeeze"])
                if planner is not None:
                    vh = plan.v_human
                    human_cmd += np.linalg.norm(vh) * env.dt_ctrl
                    human_dev += np.linalg.norm(st.u_last[:3] - vh) * env.dt_ctrl
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
                   interv=st.interv, cmd=st.cmd, active_frac=st.n_active / max(st.n_ticks, 1),
                   human_cmd=human_cmd, human_dev=human_dev, scores=sc)
        rec["viol"] = violations(peak, slip - slip0, cfg_safe(env, cfg))
        rec["any_viol"] = bool(any(rec["viol"].values()))
        records.append(rec)
    return records, st


def cfg_safe(env, cfg):
    return env.cfg.safe          # violations are always judged against the true safe set


def summarize(records):
    """Episode-level metrics from per-seed records."""
    n = max(len(records), 1)
    keys = ("indent", "force", "shear", "squeeze", "slip")
    hc = sum(r.get("human_cmd", 0.0) for r in records)
    out = dict(
        n_seeds=len(records),
        removal_rate=sum(r["collected"] > 0 for r in records) / n,
        violation_rate=sum(r["any_viol"] for r in records) / n,
        visible_damage=int(sum(r["visible"] for r in records)),
        time_per_seed=float(np.mean([r["time"] for r in records])) if records else 0.0,
        attempts=float(np.mean([r["attempts"] for r in records])) if records else 0.0,
        peak_force=float(max([r["force"] for r in records], default=0.0)),
        peak_indent=float(max([r["indent"] for r in records], default=0.0)),
        peak_squeeze=float(max([r["squeeze"] for r in records], default=0.0)),
        slip=float(max([r["slip"] for r in records], default=0.0)),
        intervention=float(sum(r["interv"] for r in records) / max(sum(r["cmd"] for r in records), 1e-9)),
        human_authority=float(1.0 - sum(r.get("human_dev", 0.0) for r in records) / hc) if hc > 0 else None,
    )
    for k in keys:
        out["viol_" + k] = float(sum(bool(r["viol"][k]) for r in records) / n)
    return out


MODES = {
    # name: (safety filter, touch-referenced depth, who commands the motion)
    "open_loop": (False, False, "auto"),      # camera depth only, no force feedback
    "touch": (False, True, "auto"),           # nominal planner with touch-referenced depth, no filter
    "filtered": (True, True, "auto"),         # the proposed system
    "teleop": (False, False, "teleop"),       # person steers, no filter
    "teleop_filtered": (True, False, "teleop"),
    "shared_filtered": (True, True, "shared"),
}


def run_berry(seed, mode="filtered", cfg=None, robot_cfg=None, bounds=None, max_seeds=None, berry=None,
              record=False, frame_cb=None, operator_kw=None, return_env=False):
    """Full pipeline on one berry: calibrate jaws -> camera survey -> remove seeds.
    cfg        true physics / sensors
    robot_cfg  what the robot believes (defaults to cfg)
    bounds     conformal bounds dict (surface, lateral, indent, force, squeeze) -> filter margins"""
    import copy
    from . import perception
    from .operator import SimulatedOperator, TeleopPlanner
    cfg = copy.deepcopy(cfg) if cfg is not None else Config()
    rcfg = copy.deepcopy(robot_cfg) if robot_cfg is not None else copy.deepcopy(cfg)
    filt, touch, who = MODES[mode]
    rcfg.filter.enabled = filt
    pre_gap = rcfg.control.pre_gap
    if bounds is not None:
        apply_bounds(rcfg, bounds)
        r_c = rcfg.berry.seed_radius + rcfg.tweezers.tip_radius
        pre_gap = float(np.clip(2 * (r_c + bounds["lateral"]) + 0.0003, 0.0020, 0.0034))
    env = make_env(cfg, seed=seed, berry=berry)
    targets, info = perception.survey(env)
    lat, hgt, ang = perception.localization_errors(env, targets)
    planner = None
    if who != "auto":
        op = SimulatedOperator(np.random.default_rng(seed + 5000), **(operator_kw or {}))
        planner = TeleopPlanner(rcfg, env.jaw_cal, op, env, mode=who)
    recs, st = run_autonomous(env, targets, cfg=rcfg, use_touch=touch, planner=planner, record=record,
                              frame_cb=frame_cb, max_seeds=max_seeds,
                              pre_gaps=[pre_gap] * len(targets))
    for r, la, hg in zip(recs, lat, hgt):
        r["scores"]["lateral"] = float(la)
        r["scores"]["cam_height"] = float(hg)
    out = dict(seed=seed, mode=mode, summary=summarize(recs), records=recs, survey=info, pre_gap=pre_gap)
    return (out, env, st) if return_env else out


def apply_bounds(cfg, bounds):
    """Use conformal bounds as the filter's uncertainty margins."""
    f = cfg.filter
    f.eps_surface = max(bounds["surface"], 0.0)
    f.eps_indent = max(bounds["indent"], 0.0)
    f.eps_force = max(bounds["force"], 0.0)
    f.eps_squeeze = max(bounds["squeeze"], 0.0)
    f.eps_lateral = max(bounds["lateral"], 0.0)
    return cfg


def collect_scores(results):
    """Calibration scores from run_berry outputs: name -> array (one value per seed)."""
    out = dict(surface=[], indent=[], force=[], squeeze=[], lateral=[])
    for res in results:
        for r in res["records"]:
            s = r["scores"]
            if s.get("surface") is not None:
                out["surface"].append(s["surface"])
            for k in ("indent", "force", "squeeze"):
                if s[k] > -1.0:
                    out[k].append(s[k])
            if "lateral" in s:
                out["lateral"].append(s["lateral"])
    return {k: np.array(v) for k, v in out.items()}
