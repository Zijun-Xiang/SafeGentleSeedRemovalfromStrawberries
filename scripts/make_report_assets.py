"""Re-creates the step-by-step screenshots of the report (report/screenshots/).

    python scripts/make_report_assets.py

Every image is produced by running the code of that step, so the report always shows
what the current code does.
"""
import json
import os
import pathlib
import subprocess
import sys
import time

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seedsafe import perception, viz
from seedsafe.config import Config
from seedsafe.env import SeedEnv, frame_from_axis
from seedsafe.hand import calibrate_jaws
from seedsafe.planner import NAMES
from seedsafe.runner import Stepper, make_env, run_berry, true_targets
from seedsafe.safety import brake_speed, reach_value

OUT = ROOT / "report" / "screenshots"
BLUE, ORANGE, GREEN, RED, GREY = "#2f6db5", "#e08a1e", "#2e8b57", "#c0392b", "#7f8c8d"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})
NUM = {}


def terminal_image(text, path, title="terminal"):
    """Render console output as an image (used for the 'screenshots' of commands)."""
    from PIL import Image, ImageDraw, ImageFont
    lines = text.rstrip().splitlines() or [""]
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", 15)
    except Exception:
        font = ImageFont.load_default()
    w = max(60, max(len(l) for l in lines)) * 9 + 30
    h = len(lines) * 20 + 56
    im = Image.new("RGB", (w, h), (24, 26, 31))
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, w, 30], fill=(48, 51, 59))
    d.text((12, 6), title, fill=(200, 200, 205), font=font)
    for i, l in enumerate(lines):
        d.text((14, 40 + 20 * i), l, fill=(214, 222, 235), font=font)
    im.save(path)


def side_by_side(images, path):
    h = min(im.shape[0] for im in images)
    imageio.imwrite(path, np.concatenate([im[:h] for im in images], axis=1))


def step1_scene():
    env = make_env(Config(), seed=7)
    imageio.imwrite(OUT / "01_scene_overview.png", viz.render(env, "overview", 1100, 760))
    hand = env.d.xpos[env.bid_palm] + np.array([0, 0, -0.09])
    a = viz.render(env, viz.tracking_camera(env, 0.34, 205, -12, lookat=hand), 760, 620)
    b = viz.render(env, viz.tracking_camera(env, 0.34, 105, -8, lookat=hand), 760, 620)
    side_by_side([a, b], OUT / "02_hand_holding_tweezers.png")
    c = viz.render(env, "closeup", 760, 620)
    p, R = perception.survey_pose(env)
    env.goto(p, R, timeout=10)
    d = viz.render(env, "wrist", 760, 570)
    side_by_side([c[:570], d], OUT / "03_berry_and_wrist_camera.png")
    NUM["model"] = dict(nq=int(env.m.nq), nu=int(env.m.nu), n_targets=int(env.n_targets), nbody=int(env.m.nbody))
    return env


def step2_calibration():
    env = SeedEnv(Config()); env.reset(seed=7)
    cal, raw = calibrate_jaws(env, verbose=True)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.4))
    ax[0].plot(raw["c"], np.array(raw["gap"]) * 1e3, color=GREY, lw=1, label="measured in the sweep")
    ax[0].plot(cal.c, cal.gap * 1e3, color=BLUE, lw=2, label="calibration table")
    ax[0].set_xlabel("closure command c"); ax[0].set_ylabel("jaw gap at the tips [mm]")
    ax[0].set_title("free sweep: finger closure -> jaw gap"); ax[0].legend(frameon=False, fontsize=8)
    # accuracy of the encoder-based gap estimate on a fresh sweep
    env.command(np.zeros(3), 0.15); est, tru = [], []
    while env.closure < cal.c_closed:
        env.tick(); est.append(env.obs["gap"]); tru.append(env.jaw_gap())
    env.command(np.zeros(3), -1.0)
    while env.closure > 0:
        env.tick()
    est, tru = np.array(est) * 1e3, np.array(tru) * 1e3
    ax[1].plot(tru, est, ".", ms=2, color=BLUE); ax[1].plot([0, 8], [0, 8], color="k", lw=0.8, ls=":")
    ax[1].set_xlabel("true jaw gap [mm]"); ax[1].set_ylabel("gap estimated from finger encoders [mm]")
    ax[1].set_title(f"gap estimate, rms error {np.sqrt(np.mean((est - tru) ** 2)):.3f} mm")
    ax[2].plot(raw["c_gauge"], raw["force"], color=GREY, lw=2, label="true squeeze (gauge)")
    ax[2].plot(raw["c_gauge"], raw["force_est"], color=ORANGE, lw=1.2, label="estimate from fingertip force")
    ax[2].axhline(env.cfg.safe.max_squeeze, color=RED, ls="--", lw=1); ax[2].text(raw["c_gauge"][0], env.cfg.safe.max_squeeze, " limit", color=RED, va="bottom", fontsize=8)
    ax[2].set_xlabel("closure command c"); ax[2].set_ylabel("squeeze force on the 1 mm gauge [N]")
    ax[2].set_title(f"gauge sweep, rms error {cal.gauge_err:.3f} N"); ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "04_tweezer_calibration.png"); plt.close(fig)
    NUM["calibration"] = dict(c_closed=cal.c_closed, gauge_err=cal.gauge_err, max_squeeze=cal.max_squeeze,
                              gap_rms_mm=float(np.sqrt(np.mean((est - tru) ** 2))),
                              shift_max_mm=float(np.abs(cal.shift).max() * 1e3))


def _press(filter_on, record=True):
    cfg = Config(); cfg.filter.enabled = filter_on
    env = make_env(cfg, seed=12)
    tg = true_targets(env)
    n = np.array([0.0, 0.0, 1.0])
    spot = np.array([t["p"] for t in tg]).mean(axis=0) + np.array([0.0012, 0.0012, 0])
    spot[2] = env.berry_center()[2] + env.bp.semi_axes[2]
    R = frame_from_axis(-n, (0.0, 1.0, 0.0))
    env.goto(spot + n * 0.006, R, closure_rate=0.5)
    env.tare_force()
    st = Stepper(env, cfg); st.record = True
    st.est.reset(spot, n); st.est.armed = True
    env.contact.reset_stats()
    for _ in range(600):
        st.step(np.r_[-n * 0.004, 0.0], R)
    return st.log, env


def step3_contact_and_filter():
    log_off, env = _press(False)
    log_on, _ = _press(True)
    safe = env.cfg.safe
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.5))
    d = np.linspace(0, 1.2e-3, 100)
    ax[0].plot(d * 1e3, env.bp.hertz_k * d ** 1.5, color=BLUE, lw=2, label="model  F = k * depth^1.5")
    ax[0].plot([l["indent"] * 1e3 for l in log_off], [l["f_n"] for l in log_off], ".", ms=3, color=GREY, label="simulated press")
    ax[0].axvline(safe.max_indent * 1e3, color=RED, ls="--", lw=1); ax[0].axhline(safe.max_tip_force, color=RED, ls="--", lw=1)
    ax[0].axvline(safe.visible_indent * 1e3, color="k", ls=":", lw=1)
    ax[0].text(safe.max_indent * 1e3, 0.5, " safe limit", color=RED, fontsize=8); ax[0].text(safe.visible_indent * 1e3, 0.42, " visible damage", fontsize=8)
    ax[0].set_xlim(0, 1.25); ax[0].set_ylim(0, 0.6)
    ax[0].set_xlabel("tip indentation [mm]"); ax[0].set_ylabel("tip force [N]"); ax[0].set_title("flesh contact model (one tip)")
    ax[0].legend(frameon=False, fontsize=8, loc="upper left")
    t0 = log_off[0]["t"]
    for log, c, lab in ((log_off, GREY, "no filter"), (log_on, BLUE, "with safety filter")):
        ax[1].plot([l["t"] - t0 for l in log], [l["indent"] * 1e3 for l in log], color=c, lw=2, label=lab)
        ax[2].plot([l["t"] - t0 for l in log], [l["f_n"] for l in log], color=c, lw=2, label=lab)
    ax[1].plot([l["t"] - t0 for l in log_on], [max(l["indent_ub"], 0) * 1e3 for l in log_on], color=ORANGE, lw=1, label="filter's upper bound on indentation")
    ax[1].axhline(safe.max_indent * 1e3, color=RED, ls="--", lw=1); ax[1].axhline(safe.visible_indent * 1e3, color="k", ls=":", lw=1)
    ax[1].set_xlabel("time [s]"); ax[1].set_ylabel("tip indentation [mm]"); ax[1].set_title("same command: push 2 mm into the berry at 4 mm/s")
    ax[1].legend(frameon=False, fontsize=8)
    ax[2].axhline(safe.max_tip_force, color=RED, ls="--", lw=1)
    ax[2].set_xlabel("time [s]"); ax[2].set_ylabel("tip force [N]"); ax[2].set_title("tip force"); ax[2].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "05_contact_model_and_filter.png"); plt.close(fig)
    NUM["press"] = dict(off_indent_mm=max(l["indent"] for l in log_off) * 1e3, on_indent_mm=max(l["indent"] for l in log_on) * 1e3,
                        off_force=max(l["f_n"] for l in log_off), on_force=max(l["f_n"] for l in log_on))
    # reachability picture
    f = env.cfg.filter
    dd, vv, safe_mask = reach_value(f.v_touch, f.a_brake, f.delay, d_max=0.006, v_max=0.06)
    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    ax.contourf(dd * 1e3, vv * 1e3, safe_mask.T.astype(float), levels=[-0.5, 0.5, 1.5], colors=["#f6d5d1", "#d6ead9"])
    ax.plot(dd * 1e3, [brake_speed(x, f.v_touch, f.a_brake, f.delay) * 1e3 for x in dd], color=BLUE, lw=2, label="speed limit used by the filter")
    ax.text(3.2, 8, "safe: can still slow\nto touch speed", fontsize=9); ax.text(0.3, 45, "unsafe: too fast\nto stop in time", fontsize=9)
    ax.set_xlabel("distance to where contact becomes possible [mm]"); ax.set_ylabel("approach speed [mm/s]")
    ax.set_title("approach: safe set and its boundary"); ax.legend(frameon=False, fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(OUT / "06_reachability_approach.png"); plt.close(fig)


def step4_perception():
    env = make_env(Config(), seed=7)
    tg, info = perception.survey(env, return_image=True)
    lat, hgt, ang = perception.localization_errors(env, tg)
    if info["rgb"] is not None:
        from PIL import Image, ImageDraw
        im = Image.fromarray(info["rgb"]).resize((960, 720)); d = ImageDraw.Draw(im)
        for uv in info["uv"]:
            x, y = uv[0] * 1.5, uv[1] * 1.5
            d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(40, 120, 255), width=2)
        im.save(OUT / "07_seed_detection.png")
    errs = dict(lat=[], hgt=[], ang=[])
    for s in range(30, 42):
        e = make_env(Config(), seed=s)
        t, _ = perception.survey(e)
        a, b, c = perception.localization_errors(e, t)
        errs["lat"] += list(a * 1e3); errs["hgt"] += list(b * 1e3); errs["ang"] += list(np.degrees(c))
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    for a_, key, lab in zip(ax, ("lat", "hgt", "ang"), ("sideways error [mm]", "surface height error [mm]\n(+ = surface higher than estimated)", "normal direction error [deg]")):
        a_.hist(errs[key], bins=24, color=BLUE, alpha=0.85); a_.set_xlabel(lab); a_.set_ylabel("seeds")
    ax[1].set_title(f"seed localization errors on 12 random berries ({len(errs['lat'])} seeds)")
    fig.tight_layout(); fig.savefig(OUT / "08_localization_errors.png"); plt.close(fig)
    NUM["perception"] = dict(n=len(errs["lat"]), detected=info["n_detected"], lat_mean=float(np.mean(errs["lat"])),
                             lat_p90=float(np.percentile(errs["lat"], 90)), hgt_mean=float(np.mean(errs["hgt"])),
                             hgt_p90=float(np.percentile(errs["hgt"], 90)), ang_mean=float(np.mean(errs["ang"])))


def step5_single_seed():
    frames = {}
    log = []
    count = [0]

    def cb(env, st, plan, k):
        count[0] += 1
        ph = NAMES[plan.phase]
        tr = env.truth()
        log.append(dict(t=env.time, phase=ph, indent=tr["indent"].max() * 1e3, f=tr["f_n"].max(), sq=tr["f_seed"].min(),
                        gap=tr["gap"] * 1e3, ind_est=st.est.indent * 1e3 if st.est.local else np.nan,
                        ind_ub=max(st.est.indent_ub, 0) * 1e3 if st.est.local else np.nan, f_est=st.est.f_n / 2,
                        sq_est=env.obs["squeeze"], gap_est=env.obs["gap"] * 1e3, active=st.filt.active.any(), k=k))
        if k == 1 and ph in ("descend", "close", "pull", "carry") and ph not in frames and env.time - plan.phase_t0 > (0.9 if ph == "descend" else 0.3):
            near = ph != "carry"
            cam = viz.tracking_camera(env, 0.05 if near else 0.2, 128, -14, lookat=env.tcp_true() + np.array([0, 0, 0.003 if near else 0.02]))
            frames[ph] = viz.hud(viz.render(env, cam, 640, 480), [ph], scale=1.1)

    out = run_berry(7, "filtered", max_seeds=2, frame_cb=cb)
    L = [l for l in log if l["k"] == 1]
    t0 = L[0]["t"]; t = np.array([l["t"] - t0 for l in L])
    fig, ax = plt.subplots(4, 1, figsize=(10, 8.4), sharex=True)
    safe = Config().safe
    cols = dict(transit="#eef2f7", descend="#fdf1dc", close="#e3f1e6", pull="#e6e9f8", carry="#eef2f7", release="#f6e6ee", lift="#f8e0dc", done="#ffffff")
    ph = [l["phase"] for l in L]
    starts = [0] + [i for i in range(1, len(L)) if ph[i] != ph[i - 1]] + [len(L) - 1]
    for a_ in ax:
        for i0, i1 in zip(starts[:-1], starts[1:]):
            a_.axvspan(t[i0], t[i1], color=cols[ph[i0]], lw=0)
    for i0, i1 in zip(starts[:-1], starts[1:]):
        ax[0].text((t[i0] + t[i1]) / 2, 0.69, ph[i0], ha="center", fontsize=8)
    ax[0].plot(t, [l["indent"] for l in L], color=BLUE, lw=2, label="true")
    ax[0].plot(t, np.clip([l["ind_est"] for l in L], 0, None), color=GREY, lw=1, label="estimate")
    ax[0].plot(t, [l["ind_ub"] for l in L], color=ORANGE, lw=1, label="upper bound used by the filter")
    ax[0].axhline(safe.max_indent * 1e3, color=RED, ls="--", lw=1); ax[0].set_ylim(0, 0.75); ax[0].set_ylabel("indentation [mm]")
    ax[0].legend(frameon=False, fontsize=8, ncol=3, loc="upper left")
    ax[1].plot(t, [l["f"] for l in L], color=BLUE, lw=2, label="true (largest tip)")
    ax[1].plot(t, [l["f_est"] for l in L], color=GREY, lw=1, label="estimate from arm joint torques (per tip)")
    ax[1].axhline(safe.max_tip_force, color=RED, ls="--", lw=1); ax[1].set_ylim(-0.03, 0.2); ax[1].set_ylabel("tip force [N]")
    ax[1].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax[2].plot(t, [l["sq"] for l in L], color=BLUE, lw=2, label="true")
    ax[2].plot(t, [l["sq_est"] for l in L], color=GREY, lw=1, label="estimate from fingertip force")
    ax[2].axhline(safe.max_squeeze, color=RED, ls="--", lw=1); ax[2].set_ylabel("jaw squeeze [N]"); ax[2].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    ax[3].plot(t, [l["gap"] for l in L], color=BLUE, lw=2, label="true")
    ax[3].plot(t, [l["gap_est"] for l in L], color=GREY, lw=1, label="estimate from finger encoders")
    act = np.array([l["active"] for l in L])
    ax[3].fill_between(t, 0, 0.35, where=act, color=ORANGE, alpha=0.8, lw=0, label="safety filter is changing the command")
    ax[3].set_ylabel("jaw gap [mm]"); ax[3].set_xlabel("time [s]"); ax[3].legend(frameon=False, fontsize=8, ncol=3, loc="upper right")
    fig.suptitle("one seed, start to finish (filtered system)", y=0.995, fontsize=11)
    fig.tight_layout(); fig.savefig(OUT / "09_single_seed_timeline.png"); plt.close(fig)
    order = [k for k in ("descend", "close", "pull", "carry") if k in frames]
    if order:
        side_by_side([frames[k] for k in order], OUT / "10_filmstrip.png")
    r = out["records"][1]
    NUM["single"] = dict(time=r["time"], indent_mm=r["indent"] * 1e3, force=r["force"], squeeze=r["squeeze"], result=r["result"])


def step6_conformal():
    p = ROOT / "results" / "calibration.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    sc, b = d["scores"], d["bounds"]
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.1))
    spec = (("surface", "robot's height estimate at first real contact [mm]", 1e3), ("indent", "indentation: true - estimate [mm]", 1e3),
            ("lateral", "seed position error, sideways [mm]", 1e3), ("force", "force: true - estimate [N]", 1), ("squeeze", "squeeze: true - estimate [N]", 1))
    for a_, (k, lab, s) in zip(ax, spec):
        v = np.array(sc[k]) * s
        a_.hist(v, bins=22, color=BLUE, alpha=0.85)
        for al, c in (("0.5", GREY), ("0.1", ORANGE), ("0.01", RED)):
            a_.axvline(b[al][k] * s, color=c, lw=1.5, label=f"{100 * (1 - float(al)):.0f}% bound")
        a_.set_xlabel(lab, fontsize=8.5); a_.set_ylabel("seeds")
    ax[0].legend(frameon=False, fontsize=7)
    ax[2].set_title(f"calibration errors ({len(sc['surface'])} seeds on {len(d['results'])} berries) and their conformal bounds")
    fig.tight_layout(); fig.savefig(OUT / "11_conformal_calibration.png"); plt.close(fig)
    NUM["conformal"] = dict(n=len(sc["surface"]), berries=len(d["results"]), bounds=b)


def step7_tests():
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"], cwd=ROOT, capture_output=True, text=True)
    txt = "$ python -m pytest -q tests\n" + (r.stdout + r.stderr).strip()
    terminal_image(txt, OUT / "12_tests.png", "tests")
    (ROOT / "report" / "logs").mkdir(exist_ok=True)
    (ROOT / "report" / "logs" / "tests.txt").write_text(txt)
    NUM["tests"] = dict(passed="passed" in r.stdout and "failed" not in r.stdout, seconds=time.time() - t0, last=r.stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:]
    steps = dict(scene=step1_scene, calibration=step2_calibration, contact=step3_contact_and_filter,
                 perception=step4_perception, single=step5_single_seed, conformal=step6_conformal, tests=step7_tests)
    num_path = OUT / "numbers.json"
    if num_path.exists():
        NUM.update(json.loads(num_path.read_text()))
    for name, fn in steps.items():
        if only and name not in only:
            continue
        t0 = time.time()
        fn()
        print(f"step {name}: {time.time() - t0:.0f} s", flush=True)
    num_path.write_text(json.dumps(NUM, indent=1, default=float))
