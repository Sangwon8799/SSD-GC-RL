"""E5 — does a learned victim policy carry over to what it was not trained on?

Evaluates, on the test seeds 100-104:
  * the pool-trained REINFORCE runs (e5-reinforce-pool-s*) and the hc90-10 specialists
    (e2-reinforce-hc90-10-s*), using each run's selected checkpoint (best.pt)
  * on workloads never used for training (zipf0.8, hc95-5) and on hc90-10 at other OP (7 %, 28 %)
  * plus the rule baselines and "random among candidates" on the same cases
Writes results/transfer/transfer.csv (idempotent: finished rows are skipped).
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

from ssdgc.env import AgentPolicy, CandidateRandom, run_policy  # noqa: E402
from ssdgc.policies import make_policy  # noqa: E402
from train import build_agent  # noqa: E402

CASES = [("zipf0.8", 0.14), ("hc95-5", 0.14), ("hc90-10", 0.07), ("hc90-10", 0.28), ("uniform", 0.14)]
SEEDS = [100, 101, 102, 103, 104]
OUT = "results/transfer/transfer.csv"


def load(run_dir):
    cfg = json.load(open(os.path.join(run_dir, "config.json")))
    agent = build_agent(cfg)
    agent.load_state_dict(torch.load(os.path.join(run_dir, "best.pt"), weights_only=False))
    return cfg, agent


def main():
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add((r["policy"], r["workload"], r["op"], r["seed"]))
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["policy", "trained_on", "workload", "op", "seed", "waf"])
    pols = []
    for s in range(5):
        for kind, rid in (("pool", f"e5-reinforce-pool-s{s}"), ("hc90-10", f"e2-reinforce-hc90-10-s{s}")):
            d = os.path.join("runs", rid)
            if os.path.exists(os.path.join(d, "done.json")) and os.path.exists(os.path.join(d, "best.pt")):
                pols.append((rid, kind, d))
    for wl, op in CASES:
        for i, seed in enumerate(SEEDS):
            for name in ("greedy", "cb", "cand-random"):
                key = (name, wl, str(op), str(seed))
                if key in done:
                    continue
                pol = CandidateRandom(8, 4, seed=seed) if name == "cand-random" else make_policy(name, np.random.default_rng(seed))
                r = run_policy(pol, wl, seed, op=op)
                w.writerow([name, "-", wl, op, seed, round(r["waf"], 5)])
                fh.flush()
            for rid, kind, d in pols:
                key = (rid, wl, str(op), str(seed))
                if key in done:
                    continue
                cfg, agent = load(d)
                r = run_policy(AgentPolicy(agent, cfg["k"], cfg["n_greedy"], cfg["drop"]), wl, seed, op=op)
                w.writerow([rid, kind, wl, op, seed, round(r["waf"], 5)])
                fh.flush()
                print(rid, wl, op, seed, round(r["waf"], 4), flush=True)


if __name__ == "__main__":
    main()
