"""Runs the simulated experiments of the project and writes results/*.json.

    python scripts/run_experiments.py e1     filtered system vs unfiltered baselines
    python scripts/run_experiments.py e2     how conservative must the filter be (conformal levels)
    python scripts/run_experiments.py e3     shared control: who should command what
    python scripts/run_experiments.py e4     transfer to a shifted "real" domain and what calibration helps
    python scripts/run_experiments.py all    everything (about 1.5 h on 2 cores)

Options:  --berries N  --seeds N  --workers N
"""
import argparse
import copy
import json
import multiprocessing as mp
import os
import pathlib
import sys
import time

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from seedsafe.config import Config
from seedsafe.conformal import bounds_from_scores, coverage

RESULTS = ROOT / "results"
ALPHAS = [0.5, 0.2, 0.1, 0.05, 0.01]


def _clean(o):
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer, np.bool_)):
        return o.item()
    return o


def _job(kw):
    from seedsafe.runner import run_berry
    tag = kw.pop("tag")
    t0 = time.time()
    try:
        out = run_berry(**kw)
    except Exception as exc:                      # keep the batch alive, record the failure
        import traceback
        out = dict(seed=kw.get("seed"), mode=kw.get("mode"), error=traceback.format_exc(), summary=None, records=[])
    out["tag"] = tag
    out["wall"] = time.time() - t0
    s = out["summary"]
    if s:
        print(f"  [{tag:>22s}] berry {out['seed']:3d}: removed {s['removal_rate']:.2f}  violations "
              f"{s['violation_rate']:.2f}  {s['time_per_seed']:.1f} s/seed  ({out['wall']:.0f} s wall)", flush=True)
    else:
        print(f"  [{tag}] berry {out['seed']} FAILED\n{out['error']}", flush=True)
    return _clean(out)


def run_jobs(jobs, workers):
    if workers <= 1:
        return [_job(j) for j in jobs]
    with mp.get_context("spawn").Pool(workers) as pool:
        return pool.map(_job, jobs, chunksize=1)


def save(name, data):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(_clean(data)))
    print(f"saved results/{name}.json")


def load(name):
    return json.loads((RESULTS / f"{name}.json").read_text())


def scores_of(results):
    from seedsafe.runner import collect_scores
    return collect_scores([r for r in results if r["summary"]])


# ----------------------------------------------------------------------------------------
def calibration(a):
    """Calibration berries (seeds 1000+): run the filtered system with hand-set margins and
    record the estimate errors. These give the conformal bounds used everywhere else."""
    jobs = [dict(tag="calibration", seed=1000 + i, mode="filtered", max_seeds=a.seeds) for i in range(a.calib)]
    res = run_jobs(jobs, a.workers)
    sc = scores_of(res)
    bounds = {str(al): bounds_from_scores(sc, al) for al in ALPHAS}
    save("calibration", dict(results=res, scores=sc, bounds=bounds))
    return bounds


def get_bounds(a):
    if not (RESULTS / "calibration.json").exists():
        calibration(a)
    return load("calibration")["bounds"]


def e1(a):
    b = get_bounds(a)["0.1"]
    jobs = []
    for mode in ("open_loop", "touch", "filtered"):
        for i in range(a.berries):
            jobs.append(dict(tag=mode, seed=i, mode=mode, max_seeds=a.seeds, bounds=b if mode == "filtered" else None))
    save("e1_filter_vs_baseline", dict(results=run_jobs(jobs, a.workers), bounds=b))


def e2(a):
    bounds = get_bounds(a)
    zero = dict(surface=0.0, indent=0.0, force=0.0, squeeze=0.0, lateral=bounds["0.1"]["lateral"])
    jobs = [dict(tag="no margin", seed=i, mode="filtered", max_seeds=a.seeds, bounds=zero) for i in range(a.berries)]
    for al in ALPHAS:
        jobs += [dict(tag=f"alpha={al}", seed=i, mode="filtered", max_seeds=a.seeds, bounds=bounds[str(al)])
                 for i in range(a.berries)]
    res = run_jobs(jobs, a.workers)
    # coverage of the bounds on these (held-out) berries
    cov = {}
    for al in ALPHAS:
        sc = scores_of([r for r in res if r["tag"] == f"alpha={al}"])
        cov[str(al)] = {k: coverage(v, bounds[str(al)][k]) for k, v in sc.items() if len(v)}
    save("e2_uncertainty", dict(results=res, bounds=bounds, coverage=cov))


def e3(a):
    b = get_bounds(a)["0.1"]
    jobs = []
    for mode in ("teleop", "teleop_filtered", "shared_filtered", "filtered"):
        for i in range(a.berries):
            jobs.append(dict(tag=mode, seed=i, mode=mode, max_seeds=a.seeds, bounds=b if "filtered" in mode else None))
    save("e3_shared_control", dict(results=run_jobs(jobs, a.workers), bounds=b))


def real_config():
    """A shifted domain standing in for hardware: softer fruit, noisier sensing, worse calibration."""
    cfg = Config()
    cfg.berry.hertz_k *= 0.42
    cfg.berry.mu_seed = 0.5
    cfg.berry.pull_force *= 1.15
    cfg.berry.cradle_k *= 0.7
    cfg.sensors.torque_noise *= 1.5
    cfg.sensors.torque_bias *= 1.5
    cfg.sensors.depth_noise *= 1.6
    cfg.sensors.tool_calib_err *= 1.6
    cfg.sensors.handeye_trans *= 1.5
    cfg.sensors.fingertip_force_noise *= 1.5
    return cfg


def _e4_job(kw):
    """One berry in the shifted domain. variant: sim / probe / probe+recal."""
    from seedsafe import perception
    from seedsafe.calib import measure_force_noise, probe_stiffness
    from seedsafe.runner import apply_bounds, make_env, run_autonomous, summarize
    tag, seed, variant, bounds, max_seeds = kw["tag"], kw["seed"], kw["variant"], kw["bounds"], kw["max_seeds"]
    t0 = time.time()
    cfg = real_config()
    rcfg = Config()                               # the robot's belief: nominal simulation model
    rcfg.filter.enabled = variant != "unfiltered"
    apply_bounds(rcfg, bounds)
    r_c = rcfg.berry.seed_radius + rcfg.tweezers.tip_radius
    pre_gap = float(np.clip(2 * (r_c + bounds["lateral"]) + 0.0003, 0.0020, 0.0034))
    env = make_env(cfg, seed=seed)
    targets, info = perception.survey(env)
    k_hat = None
    if variant in ("probe", "probe+recal"):
        sigma = measure_force_noise(env)                    # touch threshold from the real sensor noise
        rcfg.filter.touch_force = max(rcfg.filter.touch_force, 4.0 * sigma)
        k_hat = probe_stiffness(env, rcfg, targets)
        if k_hat:
            rcfg.berry.hertz_k = k_hat
            rcfg.filter.k_upper = 1.3 * k_hat
    probe_peak = dict(env.contact.peak)           # what the calibration press itself did to the fruit
    env.contact.reset_stats()
    lat, hgt, _ = perception.localization_errors(env, targets)
    recs, _ = run_autonomous(env, targets, cfg=rcfg, max_seeds=max_seeds, pre_gaps=[pre_gap] * len(targets))
    for r, la in zip(recs, lat):
        r["scores"]["lateral"] = float(la)
    out = dict(tag=tag, seed=seed, mode=variant, summary=summarize(recs), records=recs, k_hat=k_hat,
               k_true=env.bp.hertz_k, probe_peak=probe_peak, wall=time.time() - t0)
    s = out["summary"]
    print(f"  [{tag:>22s}] berry {seed:3d}: removed {s['removal_rate']:.2f}  violations {s['violation_rate']:.2f}"
          f"  k_hat={k_hat}  k_true={env.bp.hertz_k:.0f}  ({out['wall']:.0f} s wall)", flush=True)
    return _clean(out)


def e4(a):
    sim_b = get_bounds(a)["0.1"]
    pool = mp.get_context("spawn").Pool(a.workers) if a.workers > 1 else None
    mapper = (lambda f, js: pool.map(f, js, chunksize=1)) if pool else (lambda f, js: [f(j) for j in js])
    n = a.berries
    ref = run_jobs([dict(tag="simulation (reference)", seed=i, mode="filtered", max_seeds=a.seeds, bounds=sim_b)
                    for i in range(n)], a.workers)
    jobs = [dict(tag="A: sim-calibrated", seed=i, variant="sim", bounds=sim_b, max_seeds=a.seeds) for i in range(n)]
    jobs += [dict(tag="B: + on-site probes", seed=i, variant="probe", bounds=sim_b, max_seeds=a.seeds) for i in range(n)]
    cal_jobs = [dict(tag="recalibration runs", seed=2000 + i, variant="probe", bounds=sim_b, max_seeds=a.seeds)
                for i in range(max(4, n // 2))]
    res = mapper(_e4_job, jobs + cal_jobs)
    cal = [r for r in res if r["tag"] == "recalibration runs"]
    from seedsafe.runner import collect_scores
    real_b = bounds_from_scores(collect_scores(cal), 0.1)
    res += mapper(_e4_job, [dict(tag="C: + probes + new bounds", seed=i, variant="probe+recal", bounds=real_b,
                                 max_seeds=a.seeds) for i in range(n)])
    if pool:
        pool.close()
    save("e4_sim_to_real", dict(results=ref + res, sim_bounds=sim_b, real_bounds=real_b))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("which", choices=["calib", "e1", "e2", "e3", "e4", "all"])
    ap.add_argument("--berries", type=int, default=10, help="test berries per condition")
    ap.add_argument("--calib", type=int, default=10, help="calibration berries")
    ap.add_argument("--seeds", type=int, default=10, help="target seeds per berry")
    ap.add_argument("--workers", type=int, default=max(1, min(4, (os.cpu_count() or 2))))
    a = ap.parse_args()
    t0 = time.time()
    todo = ["calib", "e1", "e2", "e3", "e4"] if a.which == "all" else [a.which]
    for name in todo:
        print(f"=== {name} ===", flush=True)
        {"calib": calibration, "e1": e1, "e2": e2, "e3": e3, "e4": e4}[name](a)
    print(f"done in {(time.time() - t0) / 60:.1f} min")
