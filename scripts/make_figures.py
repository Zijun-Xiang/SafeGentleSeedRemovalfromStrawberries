"""Turns results/*.json into the figures and tables of the report (report/figures/)."""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from seedsafe.config import Config

RES, FIG = ROOT / "results", ROOT / "report" / "figures"
BLUE, ORANGE, GREEN, RED, GREY, PURPLE = "#2f6db5", "#e08a1e", "#2e8b57", "#c0392b", "#7f8c8d", "#7d5ba6"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 130, "savefig.bbox": "tight"})
SAFE = Config().safe


def wilson(k, n, z=1.96):
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return p, max(0.0, c - h), min(1.0, c + h)


def pooled(results, tag):
    recs = [r for res in results if res["tag"] == tag and res["summary"] for r in res["records"]]
    n = len(recs)
    out = dict(n=n, berries=sum(1 for res in results if res["tag"] == tag and res["summary"]))
    out["removal"] = wilson(sum(r["collected"] > 0 for r in recs), n)
    out["violation"] = wilson(sum(bool(r["any_viol"]) for r in recs), n)
    for k in ("indent", "force", "shear", "squeeze", "slip"):
        out["viol_" + k] = sum(bool(r["viol"][k]) for r in recs) / max(n, 1)
    out["time"] = float(np.mean([r["time"] for r in recs])) if recs else 0.0
    out["time_sd"] = float(np.std([r["time"] for r in recs])) if recs else 0.0
    out["attempts"] = float(np.mean([r["attempts"] for r in recs])) if recs else 0.0
    out["visible"] = int(sum(r["visible"] for r in recs))
    out["indent"] = np.array([r["indent"] for r in recs]) * 1e3
    out["force"] = np.array([r["force"] for r in recs])
    out["squeeze"] = np.array([r["squeeze"] for r in recs])
    cmd = sum(r["cmd"] for r in recs)
    out["intervention"] = sum(r["interv"] for r in recs) / cmd if cmd > 0 else 0.0
    hc = sum(r.get("human_cmd", 0) for r in recs)
    out["authority"] = 1 - sum(r.get("human_dev", 0) for r in recs) / hc if hc > 0 else None
    return out


def rate_bars(ax, stats, key, labels, colors, title, ylabel="share of seeds"):
    x = np.arange(len(stats))
    p = [s[key][0] for s in stats]
    lo = [s[key][0] - s[key][1] for s in stats]
    hi = [s[key][2] - s[key][0] for s in stats]
    ax.bar(x, p, yerr=[lo, hi], color=colors, capsize=3, width=0.65, error_kw=dict(lw=1))
    for xi, pi in zip(x, p):
        ax.text(xi, min(pi + 0.05, 1.02), f"{100 * pi:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, 1.12); ax.set_title(title, fontsize=10); ax.set_ylabel(ylabel)


def table(stats, labels, extra=()):
    head = ["condition", "seeds", "removed", "any violation", "indent", "force", "shear", "squeeze", "slip",
            "visible damage", "time / seed [s]", "attempts"] + [e[0] for e in extra]
    rows = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for s, lab in zip(stats, labels):
        row = [lab, str(s["n"]), f"{100 * s['removal'][0]:.0f}%", f"{100 * s['violation'][0]:.0f}%"]
        row += [f"{100 * s['viol_' + k]:.0f}%" for k in ("indent", "force", "shear", "squeeze", "slip")]
        row += [str(s["visible"]), f"{s['time']:.1f}", f"{s['attempts']:.2f}"]
        row += [e[1](s) for e in extra]
        rows.append("| " + " | ".join(row) + " |")
    return "\n".join(rows)


def strip(ax, stats, key, labels, colors, limit, ylabel, visible=None):
    rng = np.random.default_rng(0)
    for i, s in enumerate(stats):
        v = s[key]
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(v)), v, s=9, color=colors[i], alpha=0.55, linewidths=0)
    ax.axhline(limit, color=RED, lw=1.2, ls="--"); ax.text(len(stats) - 0.45, limit, " limit", color=RED, va="bottom", fontsize=8)
    if visible:
        ax.axhline(visible, color="k", lw=0.8, ls=":"); ax.text(len(stats) - 0.45, visible, " visible", va="bottom", fontsize=8)
    ax.set_xticks(range(len(stats))); ax.set_xticklabels(labels, fontsize=8.5); ax.set_ylabel(ylabel)


def fig_e1():
    d = json.loads((RES / "e1_filter_vs_baseline.json").read_text())
    tags = ["open_loop", "touch", "filtered"]
    labels = ["camera depth only\n(no filter)", "touch-referenced\n(no filter)", "touch + safety filter\n(ours)"]
    cols = [GREY, ORANGE, BLUE]
    st = [pooled(d["results"], t) for t in tags]
    fig, ax = plt.subplots(1, 5, figsize=(16, 3.5))
    rate_bars(ax[0], st, "removal", labels, cols, "seeds removed and placed in the tray")
    rate_bars(ax[1], st, "violation", labels, cols, "seeds with a safe-set violation")
    strip(ax[2], st, "indent", labels, cols, SAFE.max_indent * 1e3, "peak indentation [mm]", SAFE.visible_indent * 1e3)
    ax[2].set_title("deepest tip indentation per seed", fontsize=10)
    strip(ax[3], st, "force", labels, cols, SAFE.max_tip_force, "peak tip force [N]", SAFE.visible_force)
    ax[3].set_title("largest tip force per seed", fontsize=10)
    ax[4].bar(range(3), [s["time"] for s in st], yerr=[s["time_sd"] for s in st], color=cols, capsize=3, width=0.65)
    ax[4].set_xticks(range(3)); ax[4].set_xticklabels(labels, fontsize=8.5); ax[4].set_ylabel("time per seed [s]")
    ax[4].set_title("time per seed", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e1_filter_vs_baseline.png"); plt.close(fig)
    return table(st, [l.replace("\n", " ") for l in labels],
                 [("filter intervention", lambda s: f"{100 * s['intervention']:.0f}%")])


def fig_e2():
    d = json.loads((RES / "e2_uncertainty.json").read_text())
    alphas = ["0.5", "0.2", "0.1", "0.05", "0.01"]
    tags = ["no margin"] + [f"alpha={a}" for a in alphas]
    labels = ["no\nmargin"] + [f"{100 * (1 - float(a)):.0f}%" for a in alphas]
    st = [pooled(d["results"], t) for t in tags]
    cols = [GREY] + [BLUE] * 5
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.5))
    rate_bars(ax[0], st, "removal", labels, cols, "seeds removed")
    ax[0].set_xlabel("coverage level of the conformal bounds")
    rate_bars(ax[1], st, "violation", labels, cols, "seeds with a safe-set violation")
    ax[1].set_xlabel("coverage level of the conformal bounds")
    ax[2].bar(range(len(st)), [s["time"] for s in st], color=cols, width=0.65)
    ax[2].set_xticks(range(len(st))); ax[2].set_xticklabels(labels, fontsize=8.5); ax[2].set_ylabel("time per seed [s]")
    ax[2].set_title("time per seed", fontsize=10); ax[2].set_xlabel("coverage level of the conformal bounds")
    b = d["bounds"]
    xs = [1 - float(a) for a in alphas]
    for key, lab, c, scale in (("surface", "surface height before touch [mm]", BLUE, 1e3),
                               ("indent", "indentation after touch [mm]", GREEN, 1e3),
                               ("lateral", "seed position, sideways [mm]", PURPLE, 1e3),
                               ("force", "tip force [N x 10]", ORANGE, 10), ("squeeze", "jaw squeeze [N x 10]", RED, 10)):
        ax[3].plot(range(5), [b[a][key] * scale for a in alphas], "o-", color=c, label=lab, ms=4)
    ax[3].set_xticks(range(5)); ax[3].set_xticklabels([f"{100 * x:.0f}%" for x in xs])
    ax[3].set_title("size of the conformal bounds", fontsize=10); ax[3].set_xlabel("coverage level")
    ax[3].legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "e2_uncertainty.png"); plt.close(fig)
    # coverage check
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.plot([0.45, 1], [0.45, 1], color="k", lw=0.8, ls=":", label="target")
    for key, c in (("surface", BLUE), ("indent", GREEN), ("lateral", PURPLE), ("force", ORANGE), ("squeeze", RED)):
        ys = [d["coverage"][a].get(key, np.nan) for a in alphas]
        ax.plot(xs, ys, "o-", color=c, label=key, ms=4)
    ax.set_xlabel("target coverage (1 - alpha)"); ax.set_ylabel("coverage on held-out berries")
    ax.set_title("do the bounds hold on new berries?", fontsize=10); ax.legend(fontsize=7, frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "e2_coverage.png"); plt.close(fig)
    tb = table(st, [l.replace("\n", " ") for l in labels], [("filter intervention", lambda s: f"{100 * s['intervention']:.0f}%")])
    bt = ["| coverage | surface [mm] | indentation [mm] | sideways [mm] | force [N] | squeeze [N] |", "|---|---|---|---|---|---|"]
    for a in alphas:
        bt.append(f"| {100 * (1 - float(a)):.0f}% | {b[a]['surface'] * 1e3:.2f} | {b[a]['indent'] * 1e3:.2f} | "
                  f"{b[a]['lateral'] * 1e3:.2f} | {b[a]['force']:.3f} | {b[a]['squeeze']:.3f} |")
    return tb, "\n".join(bt)


def fig_e3():
    d = json.loads((RES / "e3_shared_control.json").read_text())
    tags = ["teleop", "teleop_filtered", "shared_filtered", "filtered"]
    labels = ["person steers\n(no filter)", "person steers\n+ filter", "person: seed + sideways\nrobot: depth, + filter", "fully automatic\n+ filter"]
    cols = [GREY, ORANGE, BLUE, GREEN]
    st = [pooled(d["results"], t) for t in tags]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    rate_bars(ax[0], st, "removal", labels, cols, "seeds removed")
    rate_bars(ax[1], st, "violation", labels, cols, "seeds with a safe-set violation")
    ax[2].bar(range(4), [s["time"] for s in st], yerr=[s["time_sd"] for s in st], color=cols, capsize=3, width=0.65)
    ax[2].set_xticks(range(4)); ax[2].set_xticklabels(labels, fontsize=8.5); ax[2].set_ylabel("time per seed [s]")
    ax[2].set_title("time per seed", fontsize=10)
    au = [s["authority"] if s["authority"] is not None else 0 for s in st]
    ax[3].bar(range(4), au, color=cols, width=0.65)
    for i, v in enumerate(au):
        ax[3].text(i, v + 0.03, f"{100 * v:.0f}%" if st[i]["authority"] is not None else "n/a", ha="center", fontsize=9)
    ax[3].set_xticks(range(4)); ax[3].set_xticklabels(labels, fontsize=8.5); ax[3].set_ylim(0, 1.12)
    ax[3].set_ylabel("share of the person's command that is executed"); ax[3].set_title("operator authority", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e3_shared_control.png"); plt.close(fig)
    return table(st, [l.replace("\n", " ") for l in labels],
                 [("operator authority", lambda s: "n/a" if s["authority"] is None else f"{100 * s['authority']:.0f}%")])


def fig_e4():
    d = json.loads((RES / "e4_sim_to_real.json").read_text())
    tags = ["simulation (reference)", "A: sim-calibrated", "B: + on-site probes", "C: + probes + new bounds"]
    labels = ["nominal simulation\n(reference)", "shifted domain\nA: sim calibration", "shifted domain\nB: + on-site probes",
              "shifted domain\nC: + probes + new bounds"]
    cols = [GREY, RED, ORANGE, BLUE]
    st = [pooled(d["results"], t) for t in tags]
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    rate_bars(ax[0], st, "removal", labels, cols, "seeds removed")
    rate_bars(ax[1], st, "violation", labels, cols, "seeds with a safe-set violation")
    strip(ax[2], st, "indent", labels, cols, SAFE.max_indent * 1e3, "peak indentation [mm]", SAFE.visible_indent * 1e3)
    ax[2].set_title("deepest tip indentation per seed", fontsize=10)
    pr = [r for r in d["results"] if r.get("k_hat")]
    ax[3].scatter([r["k_true"] for r in pr], [r["k_hat"] for r in pr], s=18, color=BLUE)
    lim = [0, max([r["k_true"] for r in pr] + [r["k_hat"] for r in pr]) * 1.1]
    ax[3].plot(lim, lim, color="k", lw=0.8, ls=":")
    ax[3].axhline(Config().berry.hertz_k, color=GREY, lw=1); ax[3].text(lim[0], Config().berry.hertz_k, " simulation value", fontsize=8, va="bottom", color=GREY)
    ax[3].set_xlabel("true flesh stiffness [N/m^1.5]"); ax[3].set_ylabel("stiffness from the probe press")
    ax[3].set_title("on-site stiffness probe", fontsize=10)
    fig.tight_layout(); fig.savefig(FIG / "e4_sim_to_real.png"); plt.close(fig)
    probe = [r["probe_peak"] for r in d["results"] if r.get("k_hat") and r.get("probe_peak")]
    note = ""
    if probe:
        pi = np.array([p["indent"] for p in probe]) * 1e3
        note = (f"Probe presses themselves: peak indentation median {np.median(pi):.2f} mm, max {pi.max():.2f} mm; "
                f"{100 * np.mean(pi > SAFE.max_indent * 1e3):.0f}% of the probed berries exceeded the 0.6 mm limit during the probe.")
    sb, rb = d["sim_bounds"], d["real_bounds"]
    bt = ["| bound | from simulation | recalibrated in the shifted domain |", "|---|---|---|"]
    for k, u, s in (("surface", "mm", 1e3), ("indent", "mm", 1e3), ("lateral", "mm", 1e3), ("force", "N", 1), ("squeeze", "N", 1)):
        bt.append(f"| {k} [{u}] | {sb[k] * s:.3f} | {rb[k] * s:.3f} |")
    return table(st, [l.replace("\n", " ") for l in labels]), note, "\n".join(bt)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, fn in (("e1", fig_e1), ("e2", fig_e2), ("e3", fig_e3), ("e4", fig_e4)):
        try:
            out[name] = fn()
            print("figure", name, "ok")
        except FileNotFoundError:
            print("no results yet for", name)
    (FIG / "tables.json").write_text(json.dumps(out, indent=1))
