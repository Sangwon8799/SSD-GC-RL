"""Record what a trained rule-choosing agent does over time (one evaluation run, test seed 100).

For every macro step (64 GCs) after the warm-up: the chosen beta, the mean valid ratio of its victims, the
mean valid ratio of the closed blocks and the fraction of closed blocks that are GC blocks.
    python scripts/trace_rule.py runs/r2-dqn-hc90-10-s0 [n_steps] -> results/analysis/trace_<run>.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ssdgc.ftl import CLOSED, FTL, GCW  # noqa: E402
from ssdgc.rule_env import BETAS, RuleAgentPolicy  # noqa: E402
from ssdgc.workloads import make_workload  # noqa: E402
from train_rule import build  # noqa: E402


def main():
    d = sys.argv[1]
    n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 120
    cfg = json.load(open(os.path.join(d, "config.json")))
    ag = build(cfg)
    ag.load_state_dict(torch.load(os.path.join(d, "best.pt"), weights_only=False))
    pol = RuleAgentPolicy(ag, cfg["M"], stochastic=bool(cfg["eval_stochastic"]), acts=cfg["acts"], drop=cfg["drop"])
    f = FTL(256, 64, cfg["op"])
    per_gc = []

    def wrapped(ftl):
        v = pol(ftl)
        per_gc.append((pol.beta, ftl.valid[v] / ftl.P, int(ftl.origin[v] == GCW), ftl.now - ftl.close_time[v]))
        return v

    f.victim_fn = wrapped
    wl = make_workload(cfg["workload"] if not cfg["pool"] else "hc90-10", f.U, 100)
    for lpn in range(f.U):
        f.write(lpn)
    for _ in range(3 * f.U):
        f.write(wl.next())
    rows = []
    start = len(pol.choices)
    while len(pol.choices) < start + n_steps:
        c0, k0 = f.gc_copies, len(pol.choices)
        while len(pol.choices) == k0:
            f.write(wl.next())
        # a new macro step has just started: summarise the one before it
        closed = f.state == CLOSED
        rows.append([len(rows), BETAS[pol.choices[-2]] if len(pol.choices) > 1 else BETAS[pol.choices[-1]],
                     round(pol.obs.victim_u_last, 4), round(float(f.valid[closed].mean() / f.P), 4),
                     round(float((f.origin[closed] == GCW).mean()), 4)])
    os.makedirs("results/analysis", exist_ok=True)
    out = f"results/analysis/trace_{os.path.basename(d.rstrip('/'))}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["step", "beta", "victim_u", "closed_mean_u", "gc_block_frac"])
        w.writerows(rows)
    g = np.array(per_gc[len(per_gc) // 2:], dtype=float)          # second half: steady state under the agent
    with open(out.replace("trace_", "pergc_"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["beta", "n_gc", "mean_victim_u", "gc_block_share", "median_age"])
        for beta in BETAS:
            m = g[:, 0] == beta
            if m.any():
                w.writerow([beta, int(m.sum()), round(float(g[m, 1].mean()), 4), round(float(g[m, 2].mean()), 4),
                            round(float(np.median(g[m, 3])), 1)])
    b = np.array([r[1] for r in rows])
    print(out, {float(x): int((b == x).sum()) for x in BETAS}, "first 30:", " ".join(f"{x:g}" for x in b[:30]))


if __name__ == "__main__":
    main()
