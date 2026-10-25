"""Make every results graph and summary table from the raw run outputs.

    python scripts/plot_results.py          # -> results/figures/*.png, results/summary/*.csv

Reads runs/*/done.json, runs/*/eval_log.csv and results/** only; safe to re-run at any time
(experiments that have not finished yet are simply left out, and the number of seeds behind every
point is written next to it).  Confidence intervals: t-distribution 95 % over the seeds.
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
FIG = os.path.join(ROOT, "results", "figures")
SUM = os.path.join(ROOT, "results", "summary")
T975 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776, 6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262}
WLS = ["uniform", "hc80-20", "hc90-10", "zipf1.0", "shift90-10"]
ALGOS = ["dqn", "ddqn", "dueling", "reinforce"]
NAME = {"greedy": "greedy", "cb": "cost-benefit", "fifo": "FIFO", "random": "random", "d8": "d-choices(8)",
        "cand-random": "random among candidates", "dqn": "DQN", "ddqn": "Double DQN", "dueling": "Dueling DQN",
        "reinforce": "REINFORCE+baseline"}
COL = {"greedy": "#2f5d8a", "cb": "#2e7d5b", "fifo": "#9a6a1f", "random": "#a64040", "d8": "#7a7368",
       "cand-random": "#8a8580", "dqn": "#b07a1f", "ddqn": "#1f6b75", "dueling": "#2b2b2b", "reinforce": "#b03a3a"}


def ci(v):
    n = len(v)
    m = st.mean(v)
    return (m, T975.get(n, 1.96) * st.stdev(v) / math.sqrt(n)) if n > 1 else (m, 0.0)


def rows(path):
    return list(csv.DictReader(open(path))) if os.path.exists(path) else []


def run_mean(rid, key="best_waf"):
    p = os.path.join(ROOT, "runs", rid, "done.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    v = d.get(key) or d["final_waf"]
    return {wl: st.mean(x) for wl, x in v.items()}


def baseline(name, wl, op=0.14):
    if name == "cand-random":
        return [float(r["waf"]) for r in rows(os.path.join(ROOT, "results/baselines/cand.csv"))
                if r["policy"] == name and r["workload"] == wl]
    return [float(r["waf"]) for r in rows(os.path.join(ROOT, "results/baselines/e1.csv"))
            if r["policy"] == name and r["workload"] == wl and abs(float(r["op"]) - op) < 1e-9]


def e2(algo, wl, key="best_waf"):
    out = []
    for s in range(5):
        r = run_mean(f"e2-{algo}-{wl}-s{s}", key)
        if r is not None:
            out.append(r[wl])
    return out


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_e1():
    pols = ["greedy", "cb", "fifo", "d8", "random"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=False)
    for ax, op in zip(axes, (0.07, 0.14, 0.28)):
        w = 0.16
        for i, p in enumerate(pols):
            ms, hs = [], []
            for wl in WLS:
                m, h = ci(baseline(p, wl, op))
                ms.append(m)
                hs.append(h)
            ax.bar([k + (i - 2) * w for k in range(len(WLS))], ms, w, yerr=hs, color=COL[p], label=NAME[p], capsize=2)
        ax.set_xticks(range(len(WLS)))
        ax.set_xticklabels(WLS)
        ax.set_title(f"rule policies, OP {int(op * 100)}%")
        ax.set_ylabel("WAF (lower is better)")
    axes[0].legend(fontsize=8)
    save(fig, "e1_rules.png")


RULES = ["beta0", "beta0.5", "beta1", "beta2", "random-beta", "alt0-1", "alt0-2"]
RNAME = {"beta0": "beta 0 (greedy)", "beta0.5": "beta 0.5", "beta1": "beta 1 (cost-benefit)", "beta2": "beta 2",
         "random-beta": "random beta every 64", "alt0-1": "alternate 0/1 every 64", "alt0-2": "alternate 0/2 every 64",
         "dqn": "rule DQN", "reinforce": "rule REINFORCE", "dqn-pool": "rule DQN (pool of 5)",
         "reinforce-pool": "rule REINFORCE (pool of 5)"}
RCOL = {"beta0": "#2f5d8a", "beta0.5": "#1f6b75", "beta1": "#2e7d5b", "beta2": "#9a6a1f", "random-beta": "#8a8580",
        "alt0-1": "#c08a2a", "alt0-2": "#d9b36b", "dqn": "#1f6b75", "reinforce": "#b03a3a", "dqn-pool": "#2b2b2b",
        "reinforce-pool": "#7a2a2a"}


def rule_base(policy, wl):
    return [float(r["waf"]) for r in rows(os.path.join(ROOT, "results/baselines/rule.csv"))
            if r["policy"] == policy and r["workload"] == wl]


def r2(key, wl, which="best_waf"):
    kind = key.split("-")[1] if key.endswith("-pool") else wl
    algo = key.split("-")[0]
    out = []
    for s in range(5):
        r = run_mean(f"r2-{algo}-{kind if key.endswith('-pool') else wl}-s{s}", which)
        if r is not None and wl in r:
            out.append(r[wl])
    return out


def plot_e2(which="best_waf"):
    """Final design: rule-choosing agents vs fixed rules (OP 14%)."""
    keys = RULES + ["dqn", "reinforce", "dqn-pool", "reinforce-pool"]
    os.makedirs(SUM, exist_ok=True)
    with open(os.path.join(SUM, f"e2_rule_{which}.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["workload", "policy", "n", "mean", "ci95_half"])
        fig, ax = plt.subplots(figsize=(12, 5.5))
        width = 0.8 / len(keys)
        for i, wl in enumerate(WLS):
            for j, k in enumerate(keys):
                v = rule_base(k, wl) if k in RULES else r2(k, wl, which)
                if not v:
                    continue
                m, h = ci(v)
                w.writerow([wl, k, len(v), round(m, 4), round(h, 4)])
                ax.errorbar(i - 0.4 + (j + 0.5) * width, m, yerr=h, fmt="s" if k in RULES else "o", color=RCOL[k],
                            mfc="white" if k in RULES else RCOL[k], label=RNAME[k] if i == 0 else None, capsize=2)
        ax.set_xticks(range(len(WLS)))
        ax.set_xticklabels(WLS)
        ax.set_ylabel("WAF at OP 14% (lower is better, 95% CI)")
        ax.set_title("E2 - rule-choosing agents vs fixed rules (agents: 5 training seeds)")
        ax.legend(fontsize=7, ncol=2)
        save(fig, f"e2_rule_{which}.png")


def plot_direct():
    """First design (choose a block directly): continuing task, gamma 0.999, 2 seeds per workload."""
    keys = ["greedy", "cb", "cand-random", "reinforce", "ddqn"]
    os.makedirs(SUM, exist_ok=True)
    with open(os.path.join(SUM, "e2a_direct.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["workload", "policy", "n", "mean", "ci95_half"])
        fig, ax = plt.subplots(figsize=(10, 4.5))
        for i, wl in enumerate(WLS):
            for j, k in enumerate(keys):
                if k in ("greedy", "cb", "cand-random"):
                    v = baseline(k, wl)
                else:
                    v = [run_mean(f"d2-{k}-g0.999-{wl}-s{s}")[wl] for s in range(5) if run_mean(f"d2-{k}-g0.999-{wl}-s{s}")]
                if not v:
                    continue
                m, h = ci(v)
                w.writerow([wl, k, len(v), round(m, 4), round(h, 4)])
                ax.errorbar(i - 0.3 + j * 0.15, m, yerr=h, fmt="s" if j < 3 else "o", color=COL[k],
                            mfc="white" if j < 3 else COL[k], label=NAME[k] if i == 0 else None, capsize=2)
        ax.set_xticks(range(len(WLS)))
        ax.set_xticklabels(WLS)
        ax.set_ylabel("WAF at OP 14%")
        ax.set_title("E2a - first design (choose a block among 8 candidates) does not beat the rules")
        ax.legend(fontsize=8)
        save(fig, "e2a_direct.png")


def plot_curves():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for ax, wl in zip(axes.flat, WLS):
        for a in ("dqn", "reinforce"):
            cs = []
            for s in range(5):
                p = os.path.join(ROOT, "runs", f"r2-{a}-{wl}-s{s}", "eval_log.csv")
                if os.path.exists(p):
                    cs.append([(int(r["decisions"]), float(r["eval_waf"])) for r in rows(p)])
            if not cs:
                continue
            n = min(len(c) for c in cs)
            xs = [cs[0][i][0] for i in range(n)]
            ys = [st.mean(c[i][1] for c in cs) for i in range(n)]
            lo = [min(c[i][1] for c in cs) for i in range(n)]
            hi = [max(c[i][1] for c in cs) for i in range(n)]
            ax.plot(xs, ys, color=RCOL[a], label=f"{RNAME[a]} (n={len(cs)})")
            ax.fill_between(xs, lo, hi, color=RCOL[a], alpha=0.12)
        for b in ("beta0", "beta1", "alt0-1"):
            v = rule_base(b, wl)
            if v:
                ax.axhline(st.mean(v), color=RCOL[b], ls="--", lw=1, label=RNAME[b])
        ax.set_title(wl)
        ax.set_xlabel("GC decisions")
        ax.set_ylabel("validation WAF")
    axes.flat[-1].axis("off")
    h, lab = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(h, lab, loc="center")
    save(fig, "learning_curves.png")


def pool_mean(rid):
    p = os.path.join(ROOT, "runs", rid, "done.json")
    if not os.path.exists(p):
        return None
    d = json.load(open(p))
    return st.mean(st.mean(v) for v in d["best_waf"].values())


def plot_group(prefix, title, fname):
    """E3/E4: rule DQN trained on the pool of five, one knob changed at a time (mean WAF over the five)."""
    g = defaultdict(list)
    for p in glob.glob(os.path.join(ROOT, "runs", f"{prefix}-*-pool-s*", "done.json")):
        rid = os.path.basename(os.path.dirname(p))
        g[rid[len(prefix) + 1:].rsplit("-pool-s", 1)[0]].append(pool_mean(rid))
    ref = [v for v in (pool_mean(f"r2-dqn-pool-s{s}") for s in range(5)) if v is not None]
    if ref:
        g["main setting"] = ref
    if len(g) < 2:
        return
    names = sorted(g, key=lambda k: (k != "main setting", k))
    os.makedirs(SUM, exist_ok=True)
    with open(os.path.join(SUM, fname.replace(".png", ".csv")), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "n", "mean", "ci95_half"])
        fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.5))
        for i, k in enumerate(names):
            m, h = ci(g[k])
            w.writerow([k, len(g[k]), round(m, 4), round(h, 4)])
            ax.errorbar(m, -i, xerr=h, fmt="o", color="#b03a3a" if k == "main setting" else "#444", capsize=2)
            ax.annotate(f"{m:.3f} (n={len(g[k])})", (m + h + 0.005, -i), va="center", fontsize=8)
        for b in ("beta1", "alt0-1"):
            ax.axvline(st.mean(st.mean(rule_base(b, wl)) for wl in WLS), color=RCOL[b], ls="--", lw=1, label=RNAME[b])
        ax.set_yticks([-i for i in range(len(names))])
        ax.set_yticklabels(names)
        ax.set_xlabel("mean WAF over the five workloads, OP 14% (95% CI) - lower is better")
        ax.set_title(title)
        ax.legend(fontsize=8)
        save(fig, fname)


def plot_timing():
    r = rows(os.path.join(ROOT, "results/timing/rules.csv"))
    if not r:
        return
    g = defaultdict(list)
    for x in r:
        g[x["rule"]].append(x)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, met in zip(axes, ("lat_mean_ms", "lat_p99_ms")):
        xs = [st.mean(float(x["waf"]) for x in v) for v in g.values()]
        ys = [st.mean(float(x[met]) for x in v) for v in g.values()]
        ax.scatter(xs, ys, s=10, color="#999", label="rule (W, tau, W_e)")
        ag = defaultdict(list)
        for p in glob.glob(os.path.join(ROOT, "runs", "e6-*", "done.json")):
            d = json.load(open(p))
            if not d["exp_id"].split("-")[2].startswith("lam"):
                continue
            algo, lam = d["exp_id"].split("-")[1], d["exp_id"].split("-")[2]
            ag[(algo, lam)].append((st.mean(e["waf"] for e in d["eval"]), st.mean(e[met] for e in d["eval"])))
        for (algo, lam), v in sorted(ag.items()):
            mx, hx = ci([a for a, _ in v])
            my, hy = ci([b for _, b in v])
            ax.errorbar(mx, my, xerr=hx, yerr=hy, fmt="o" if algo == "tabq" else "^", capsize=2,
                        label=f"{algo} {lam} (n={len(v)})")
        ax.set_xlabel("WAF")
        ax.set_ylabel(met)
        ax.set_yscale("log")
        ax.set_title(f"GC timing: {met} vs WAF (OP 28%, hc80-20)")
    axes[1].legend(fontsize=7)
    save(fig, "e6_timing_pareto.png")


def main():
    plot_e1()
    plot_direct()
    plot_e2("best_waf")
    plot_e2("final_waf")
    plot_curves()
    plot_group("r3-dqn", "E3 - rule DQN, one knob at a time", "e3_rule.png")
    plot_group("r4-dqn", "E4 - ablation (rule DQN)", "e4_rule.png")
    plot_timing()
    print("figures ->", FIG)


if __name__ == "__main__":
    main()
