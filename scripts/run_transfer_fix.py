"""E4, part 2: add surface re-referencing to the on-site calibration.

The probe presses also tell where the surface really is. That corrects the camera's height
bias, which is what the filter needs before the first touch on soft fruit.

    D: on-site probes + surface re-referencing
    E: D + conformal bounds recomputed from calibration berries in the shifted domain
"""
import argparse
import multiprocessing as mp
import pathlib
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import seedsafe  # noqa: E402,F401
import numpy as np

import run_experiments as X
from seedsafe.config import Config
from seedsafe.conformal import bounds_from_scores


def job(kw):
    from seedsafe import perception
    from seedsafe.calib import measure_force_noise, probe_stiffness, surface_bias
    from seedsafe.runner import apply_bounds, fit_sphere, make_env, run_autonomous, summarize
    tag, seed, bounds, max_seeds = kw["tag"], kw["seed"], kw["bounds"], kw["max_seeds"]
    t0 = time.time()
    cfg = X.real_config()
    rcfg = Config()
    apply_bounds(rcfg, bounds)
    r_c = rcfg.berry.seed_radius + rcfg.tweezers.tip_radius
    pre_gap = float(np.clip(2 * (r_c + bounds["lateral"]) + 0.0003, 0.0020, 0.0034))
    env = make_env(cfg, seed=seed)
    targets, _ = perception.survey(env)
    sigma = measure_force_noise(env)
    rcfg.filter.touch_force = max(rcfg.filter.touch_force, 4.0 * sigma)
    info = {}
    k_hat = probe_stiffness(env, rcfg, targets, info=info)
    bias = None
    if k_hat:
        rcfg.berry.hertz_k = k_hat
        rcfg.filter.k_upper = 1.3 * k_hat
        _, radius = fit_sphere([t["p"] for t in targets], [t["n"] for t in targets])
        bias = surface_bias(targets, info, radius)
        for t in targets:                           # move the camera's surface to where touch found it
            t["p"] = t["p"] + bias * info["normal"]
    probe_peak = dict(env.contact.peak)
    env.contact.reset_stats()
    lat, hgt, _ = perception.localization_errors(env, targets)
    recs, _ = run_autonomous(env, targets, cfg=rcfg, max_seeds=max_seeds, pre_gaps=[pre_gap] * len(targets))
    for r, la in zip(recs, lat):
        r["scores"]["lateral"] = float(la)
    out = dict(tag=tag, seed=seed, mode="probe+ref", summary=summarize(recs), records=recs, k_hat=k_hat,
               k_true=env.bp.hertz_k, bias=bias, height_err_after=float(np.mean(hgt)), probe_peak=probe_peak,
               wall=time.time() - t0)
    s = out["summary"]
    print(f"  [{tag:>22s}] berry {seed:3d}: removed {s['removal_rate']:.2f}  violations {s['violation_rate']:.2f}"
          f"  bias={bias}  height err after={np.mean(hgt) * 1e3:+.2f} mm  ({out['wall']:.0f} s wall)", flush=True)
    return X._clean(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--berries", type=int, default=8)
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    sim_b = X.load("calibration")["bounds"]["0.1"]
    with mp.get_context("spawn").Pool(a.workers) as pool:
        jobs = [dict(tag="D: + surface re-reference", seed=i, bounds=sim_b, max_seeds=a.seeds) for i in range(a.berries)]
        cal = [dict(tag="recalibration runs (D)", seed=2000 + i, bounds=sim_b, max_seeds=a.seeds) for i in range(4)]
        res = pool.map(job, jobs + cal, chunksize=1)
        from seedsafe.runner import collect_scores
        sc = collect_scores([r for r in res if r["tag"].startswith("recal")])
        real_b = bounds_from_scores(sc, 0.1)
        for k in real_b:                            # too few calibration samples: keep the simulation bound
            if len(sc[k]) < 8:
                real_b[k] = sim_b[k]
        print("bounds: simulation", {k: round(v, 5) for k, v in sim_b.items()})
        print("bounds: recalibrated", {k: round(v, 5) for k, v in real_b.items()}, {k: len(v) for k, v in sc.items()}, flush=True)
        res += pool.map(job, [dict(tag="E: + re-reference + new bounds", seed=i, bounds=real_b, max_seeds=a.seeds)
                              for i in range(a.berries)], chunksize=1)
    X.save("e4b_transfer_fix", dict(results=res, sim_bounds=sim_b, real_bounds=real_b))
