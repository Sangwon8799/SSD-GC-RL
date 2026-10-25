"""What does a trained victim policy actually choose?  (diagnostic for section 7)

For each run directory given, load the selected model (best.pt if present, else model.pt), play one
evaluation run (test seed 100) and record, at every GC, which of the k candidates it took:
  * a greedy-equivalent block (as few valid pages as any)?   * the very block cost-benefit would take?
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

from ssdgc.env import AgentPolicy, CandidateRandom, candidate_set, run_policy  # noqa: E402
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
        same_g = bool(ftl.valid[b] == v.min())                  # greedy-equivalent: as few valid pages as possible
        c = int(cand[np.argmax(cb_score(ftl, cand))])
        rank_u = int((v < ftl.valid[b]).sum())                   # blocks with strictly fewer valid pages
        blocks = candidate_set(ftl, 8, 4)[0]
        hit = np.flatnonzero(blocks == b)
        slot = int(hit[0]) if len(hit) else -1                   # 0-3 greedy ranks 1-4, 4-7 cost-benefit ranks 1-4
        self.rows.append((same_g, b == c, ftl.valid[b] / ftl.P, rank_u, slot))
        return b


def main():
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    rows = []
    for d in sys.argv[1:]:
        if d.startswith("cand-random"):
            wl = d.split(":", 1)[1] if ":" in d else "hc90-10"
            name, pol = f"cand-random-{wl}", CandidateRandom(8, 4, seed=100)
        else:
            cfg, agent = build(d)
            name, wl, pol = os.path.basename(d.rstrip("/")), cfg["workload"], AgentPolicy(agent, cfg["k"], cfg["n_greedy"], cfg["drop"])
        rec = Recorder(pol)
        r = run_policy(rec, wl, 100, op=0.14)
        a = np.array(rec.rows[len(rec.rows) // 2:], dtype=float)   # second half = measured window
        g, c = a[:, 0] > 0, a[:, 1] > 0
        rows.append([name, wl, round(r["waf"], 4), round(a[:, 0].mean(), 4), round(a[:, 1].mean(), 4),
                     round(float((g & c).mean()), 4), round(float((g & ~c).mean()), 4), round(float((~g & c).mean()), 4),
                     round(float((~g & ~c).mean()), 4), round(a[:, 2].mean(), 4), round(float(np.median(a[:, 3])), 1)]
                    + [round(float((a[:, 4] == k).mean()), 4) for k in range(8)])
        print(rows[-1], flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["run", "workload", "waf", "same_as_greedy", "same_as_cb", "both", "greedy_only", "cb_only",
                        "neither", "mean_u", "median_u_rank"] + [f"slot{k}" for k in range(8)])
        w.writerows(rows)


if __name__ == "__main__":
    main()
