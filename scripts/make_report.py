"""Writes report/REPORT.md from the experiment results, tables and step screenshots."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import make_figures as F
from seedsafe.config import Config

REP = ROOT / "report"


def pct(x):
    return f"{100 * x:.0f}%"


def q3_text(m0, m50, m90, m99, m3, pct):
    v = lambda s: pct(s["violation"][0])
    r = lambda s: pct(s["removal"][0])
    return ("With the well-tuned planner the margins made no difference: 0% violations at every setting. They matter "
            "when a mis-tuned planner makes the filter decide the depth. Violations then: "
            f"{v(m0)} with no margins, {v(m50)} with 50% conformal bounds, {v(m90)} with 90% bounds, {v(m99)} with 99% "
            f"bounds ({r(m99)} of seeds removed). Three times the 99% bounds was too conservative: {r(m3)} removed.")


def q5_text(A, B, C, D, pct):
    v = lambda s: pct(s["violation"][0])
    r = lambda s: pct(s["removal"][0])
    return ("Only tested against a second, shifted simulation (softer fruit, noisier sensors), not on hardware. "
            f"Not as it is: with the simulation calibration {v(A)} of seeds had a violation ({r(A)} removed). "
            f"Partial on-site calibration made the robot too cautious to grip ({r(B)} and {r(C)} removed). "
            f"With the full on-site calibration: {r(D)} removed, {v(D)} with a violation.")


def _indent_by_result(results, tag):
    out = {}
    for cat in ("placed", "miss"):
        x = [q["indent"] * 1e3 for r in results if r["tag"] == tag for q in r["records"] if q["result"] == cat]
        out[cat] = (len(x), float(np.median(x)) if x else float("nan"))
    return out


def e4_text(ref, A, B, C, D, probe, pct, results, sim_b, real_b, n_scores):
    v = lambda s: pct(s["violation"][0])
    r = lambda s: pct(s["removal"][0])
    nv = lambda s: int(round(s["violation"][0] * s["n"]))
    mx = lambda s: float(np.max(s["indent"]))
    ib = _indent_by_result(results, "B: + on-site probes")
    return f"""Reading the results:

- **A** removes almost everything ({r(A)}) but breaks the indentation limit on {v(A)} of the seeds
  (deepest press {mx(A):.2f} mm, limit 0.6 mm). The fruit is softer than the robot believes, so
  the same force means a deeper press and touch is felt later.
- **B** has fewer violations ({v(B)}) but removes only {r(B)}. In the {ib['miss'][0]} missed seeds the
  tips went {ib['miss'][1]:.2f} mm deep (median); in the {ib['placed'][0]} gripped ones
  {ib['placed'][1]:.2f} mm. The tips stayed too shallow to grip, and after two attempts the
  planner gave up. The likely reason is the filter's indentation bound: on soft fruit the
  measured force says little about depth, so the filter depends on the surface-height bound,
  and that bound is wide.
- **C** does not help by itself ({r(C)} removed, {v(C)} with a violation). The surface estimate
  is corrected, but the filter still uses the bounds from the simulation.
- **D** changes only the bounds compared with C. The surface-height bound, now measured after
  re-referencing, is {real_b['surface'] * 1e3:.2f} mm instead of {sim_b['surface'] * 1e3:.2f} mm. The force and
  squeeze bounds grow (force {sim_b['force']:.3f} N to {real_b['force']:.3f} N, squeeze {sim_b['squeeze']:.3f} N to
  {real_b['squeeze']:.3f} N); the sensors are noisier in the shifted domain. Result: {r(D)} removed,
  {v(D)} with a violation ({nv(D)} of {D['n']} seeds, deepest press {mx(D):.2f} mm), {D['time']:.1f} s
  per seed.

The comparison of C and D shows that the bounds were what blocked the robot. Which bound
mattered most was not separated. The sideways bound also grew, and it widens the jaw opening,
which by itself reduces misses.

So the simulation-calibrated safe set did not transfer as it was. It transferred after an
on-site calibration with four parts: force-noise check, stiffness probe, surface
re-referencing and new conformal bounds. Even then the result ({r(D)} removed) is below the
nominal simulation ({r(ref)}).

Cautions:

- **The probe itself touches the fruit.** Over {probe['n']} probed berries the probe presses went
  {probe['med']:.2f} mm deep (median) and at most {probe['mx']:.2f} mm; on {pct(probe['over'])} of the
  berries they went past the 0.6 mm limit. The fitted stiffness was off by {pct(probe['k_err'])}
  (median).
- **The new bounds rest on little data.** They come from 4 berries ({n_scores['surface']} seeds;
  {n_scores['indent']} for indentation and {n_scores['squeeze']} for squeeze), collected while the robot
  was still missing most seeds. The indentation bound came out slightly negative
  ({real_b['indent'] * 1e3:.2f} mm), which means the robot's indentation estimate was above the truth
  in at least 90% of those cases.
- **The one violation in D was a deep one** ({mx(D):.2f} mm).
- **Simulation against simulation.** A real sim-to-real gap has effects that are not modelled
  here (see the limits below)."""


def main():
    cfg = Config()
    safe = cfg.safe
    num = json.loads((REP / "screenshots" / "numbers.json").read_text())
    F.FIG.mkdir(parents=True, exist_ok=True)
    t1 = F.fig_e1()
    t2, bounds_table = F.fig_e2()
    t3 = F.fig_e3()
    t4, probe, bounds4, st4 = F.fig_e4b()
    t1b, t2b, stress, sm = F.fig_stress()
    R = lambda name: json.loads((ROOT / "results" / f"{name}.json").read_text())["results"]
    e1, e2, e3, e4 = R("e1_filter_vs_baseline"), R("e2_uncertainty"), R("e3_shared_control"), R("e4_sim_to_real")
    P = F.pooled
    s_open, s_touch, s_filt = P(e1, "open_loop"), P(e1, "touch"), P(e1, "filtered")
    s_nm = P(e2, "no margin")
    s_al = {a: P(e2, f"alpha={a}") for a in ("0.5", "0.2", "0.1", "0.05", "0.01")}
    s_tel, s_telf, s_sh, s_auto = (P(e3, t) for t in ("teleop", "teleop_filtered", "shared_filtered", "filtered"))
    s_ref, s_A, s_B, s_C, s_D = st4
    m0, m50, m90, m99, m3 = sm
    e4b = json.loads((ROOT / "results" / "e4b_transfer_fix.json").read_text())
    from seedsafe.runner import collect_scores
    cal_runs = [r for r in e4b["results"] if r["tag"] == "recalibration runs (D)"]
    e4b_scores = {k: len(x) for k, x in collect_scores(cal_runs).items()}
    stress_variants = json.loads((ROOT / "results" / "e1b_stress.json").read_text())["variants"]
    cal = num["calibration"]; per = num["perception"]; press = num["press"]; single = num["single"]
    conf = num.get("conformal", {})
    b90 = conf["bounds"]["0.1"]
    n_berries = s_filt["berries"]
    tests_line = num.get("tests", {}).get("last", "")

    def rv(s):
        return f"{pct(s['removal'][0])} removed, {pct(s['violation'][0])} with a violation, {s['time']:.1f} s per seed"

    md = TEMPLATE.format(**{k: v for k, v in dict(
        max_indent=safe.max_indent * 1e3, max_force=safe.max_tip_force, max_shear=safe.max_shear,
        max_squeeze=safe.max_squeeze, max_slip=safe.max_slip * 1e3, vis_indent=safe.visible_indent * 1e3,
        vis_force=safe.visible_force,
        nq=num["model"]["nq"], nu=num["model"]["nu"],
        c_closed=cal["c_closed"], gap_rms=cal["gap_rms_mm"], gauge_err=cal["gauge_err"], cal_max_sq=cal["max_squeeze"],
        shift_max=cal["shift_max_mm"],
        per_n=per["n"], per_det=per["detected"], lat_mean=per["lat_mean"], lat_p90=per["lat_p90"],
        hgt_mean=per["hgt_mean"], hgt_p90=per["hgt_p90"], ang_mean=per["ang_mean"],
        press_off_ind=press["off_indent_mm"], press_on_ind=press["on_indent_mm"], press_off_f=press["off_force"],
        press_on_f=press["on_force"],
        single_time=single["time"], single_ind=single["indent_mm"], single_f=single["force"], single_sq=single["squeeze"],
        conf_n=conf["n"], conf_berries=conf["berries"],
        b_surface=b90["surface"] * 1e3, b_indent=b90["indent"] * 1e3, b_lateral=b90["lateral"] * 1e3,
        b_force=b90["force"], b_squeeze=b90["squeeze"],
        n_berries=n_berries, n_seeds=s_filt["n"],
        t1=t1, t1b=t1b, t2=t2, t2b=t2b, bounds_table=bounds_table, t3=t3, t4=t4, bounds4=bounds4,
        open_rv=rv(s_open), touch_rv=rv(s_touch), filt_rv=rv(s_filt),
        open_viol=pct(s_open["violation"][0]), open_rem=pct(s_open["removal"][0]), open_vis=s_open["visible"],
        touch_viol=pct(s_touch["violation"][0]), touch_rem=pct(s_touch["removal"][0]),
        filt_viol=pct(s_filt["violation"][0]), filt_rem=pct(s_filt["removal"][0]),
        filt_time=s_filt["time"], touch_time=s_touch["time"], open_time=s_open["time"],
        filt_interv=pct(s_filt["intervention"]),
        st_off_viol=pct(stress[0]["violation"][0]), st_on_viol=pct(stress[1]["violation"][0]),
        st_off_rem=pct(stress[0]["removal"][0]), st_on_rem=pct(stress[1]["removal"][0]),
        st_on_nviol=int(round(stress[1]["violation"][0] * stress[1]["n"])), st_on_n=stress[1]["n"],
        st_off_vis=stress[0]["visible"], st_on_vis=stress[1]["visible"],
        st_off_ind=float(np.max(stress[0]["indent"])), st_on_ind=float(np.max(stress[1]["indent"])),
        st_off_time=stress[0]["time"], st_on_time=stress[1]["time"], st_on_interv=pct(stress[1]["intervention"]),
        m0_rv=rv(m0), m50_rv=rv(m50), m90_rv=rv(m90), m99_rv=rv(m99), m3_rv=rv(m3),
        q3=q3_text(m0, m50, m90, m99, m3, pct),
        m0_viol=pct(m0["violation"][0]), m99_viol=pct(m99["violation"][0]),
        m3_force=stress_variants["margins: 3 x 99%"]["force"],
        m50_lat=stress_variants["margins: 50%"]["lateral"] * 1e3, m99_lat=stress_variants["margins: 99%"]["lateral"] * 1e3, q5=q5_text(s_A, s_B, s_C, s_D, pct),
        e4_reading=e4_text(s_ref, s_A, s_B, s_C, s_D, probe, pct, e4 + e4b["results"], e4b["sim_bounds"],
                           e4b["real_bounds"], e4b_scores),
        nm_rv=rv(s_nm), a50_rv=rv(s_al["0.5"]), a90_rv=rv(s_al["0.1"]), a99_rv=rv(s_al["0.01"]),
        a50_time=s_al["0.5"]["time"], a99_time=s_al["0.01"]["time"], nm_time=s_nm["time"],
        a99_rem=pct(s_al["0.01"]["removal"][0]), a50_rem=pct(s_al["0.5"]["removal"][0]), nm_viol=pct(s_nm["violation"][0]),
        tel_rv=rv(s_tel), telf_rv=rv(s_telf), sh_rv=rv(s_sh), auto_rv=rv(s_auto),
        tel_viol=pct(s_tel["violation"][0]), tel_vis=s_tel["visible"], telf_rem=pct(s_telf["removal"][0]),
        telf_viol=pct(s_telf["violation"][0]), telf_vis=s_telf["visible"],
        telf_miss_ind=float(np.median([q["indent"] * 1e3 for r in e3 if r["tag"] == "teleop_filtered"
                                       for q in r["records"] if q["result"] == "miss"])),
        telf_nviol=int(round(s_telf["violation"][0] * s_telf["n"])), telf_maxf=float(np.max(s_telf["force"])),
        tel_rem=pct(s_tel["removal"][0]), sh_rem=pct(s_sh["removal"][0]),
        telf_auth=pct(s_telf["authority"]), sh_auth=pct(s_sh["authority"]), telf_time=s_telf["time"], sh_time=s_sh["time"],
        ref_rv=rv(s_ref) + f" ({s_ref['n']} seeds)", A_rv=rv(s_A), B_rv=rv(s_B), C_rv=rv(s_C), D_rv=rv(s_D),
        tests_line=tests_line,
    ).items()})
    (REP / "REPORT.md").write_text(md)
    print("wrote report/REPORT.md")


TEMPLATE = (pathlib.Path(__file__).with_name("report_template.md")).read_text() if pathlib.Path(__file__).with_name("report_template.md").exists() else ""

if __name__ == "__main__":
    main()
