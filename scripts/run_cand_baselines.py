"""Two baselines that need the agents' candidate set (OP 14%, 5 workloads, test seeds 100-104):

  cand-random  — uniform choice among the 8 candidates (what the candidate set alone gives)
  untrained    — a freshly initialised scorer network, argmax (what "learning" starts from;
                 network seeds 0-4, one per test seed)
Writes results/baselines/cand.csv (idempotent: finished rows are skipped).
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch  # noqa: E402

from ssdgc.agents.reinforce import ReinforceAgent  # noqa: E402
from ssdgc.env import N_CAND_FEAT, N_GLOB_FEAT, AgentPolicy, CandidateRandom, run_policy  # noqa: E402

WORKLOADS = ["uniform", "hc80-20", "hc90-10", "zipf1.0", "shift90-10"]
SEEDS = [100, 101, 102, 103, 104]
OUT = "results/baselines/cand.csv"


def main():
    torch.set_num_threads(1)
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add((r["policy"], r["workload"], r["seed"]))
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["policy", "workload", "op", "seed", "waf"])
    for wl in WORKLOADS:
        for i, s in enumerate(SEEDS):
            for name in ("cand-random", "untrained"):
                if (name, wl, str(s)) in done:
                    continue
                if name == "cand-random":
                    pol = CandidateRandom(8, 4, seed=s)
                else:
                    ag = ReinforceAgent(8, N_CAND_FEAT, N_GLOB_FEAT, seed=i)
                    pol = AgentPolicy(ag, 8, 4)
                r = run_policy(pol, wl, s, op=0.14)
                w.writerow([name, wl, 0.14, s, round(r["waf"], 5)])
                fh.flush()
                print(name, wl, s, round(r["waf"], 4), flush=True)


if __name__ == "__main__":
    main()
