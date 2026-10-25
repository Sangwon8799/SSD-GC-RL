"""What does a trained victim policy actually choose?  (diagnostic for section 7)

For each run directory given, load the selected model (best.pt if present, else model.pt), play one
evaluation run (test seed 100) and record, at every GC, which of the k candidates it took:
  * same block as greedy would take?        * same block as cost-benefit would take?
  * valid ratio u and age (rank) of the chosen block
Writes results/analysis/choices.csv (one row per run).

    python scripts/analyze_choices.py runs/e2-reinforce-hc90-10-s0 runs/e2-ddqn-hc90-10-s0 ...
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ssdgc.env import AgentPolicy, CandidateRandom, run_policy  # noqa: E402
from ssdgc.policies import cb_score, eligible  # noqa: E402

OUT = os.environ.get("CHOICES_OUT", "results/analysis/choices.csv")


def build(run_dir):
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
    from train import build_agent  # noqa: E402
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    agent = build_agent(cfg)
    p = os.path.join(run_dir, "best.pt")
    if not os.path.exists(p):
        p = os.path.join(run_dir, "model.pt")
    agent.load_state_dict(torch.load(p, weights_only=False))
    return cfg, agent


class Recorder:
    """Wraps a victim policy and records each choice against greedy / cost-benefit."""

    def __init__(self, inner):
        self.inner = inner
        self.rows = []

    def __call__(self, ftl):
        b = int(self.inner(ftl))
        cand = eligible(ftl)
        v = ftl.valid[cand]
        g = int(cand[np.lexsort((ftl.close_time[cand], v))][0])
        c = int(cand[np.argmax(cb_score(ftl, cand))])
        order_u = np.argsort(np.argsort(v, kind="stable"), kind="stable")
        rank_u = int(order_u[np.flatnonzero(cand == b)[0]])
        self.rows.append((b == g, b == c, ftl.valid[b] / ftl.P, rank_u))
        return b


def main():
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for d in sys.argv[1:]:
        if d == "cand-random":
            name, wl, pol = "cand-random", "hc90-10", CandidateRandom(8, 4, seed=100)
        else:
            cfg, agent = build(d)
            name, wl, pol = os.path.basename(d.rstrip("/")), cfg["workload"], AgentPolicy(agent, cfg["k"], cfg["n_greedy"], cfg["drop"])
        rec = Recorder(pol)
        r = run_policy(rec, wl, 100, op=0.14)
        a = np.array(rec.rows[len(rec.rows) // 2:], dtype=float)   # second half = measured window
        rows.append([name, wl, round(r["waf"], 4), round(a[:, 0].mean(), 4), round(a[:, 1].mean(), 4),
                     round(a[:, 2].mean(), 4), round(float(np.median(a[:, 3])), 1)])
        print(rows[-1], flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run", "workload", "waf", "same_as_greedy", "same_as_cb", "mean_u", "median_u_rank"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
