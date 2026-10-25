"""Control for E2: same rule frequencies as the trained DQN, but drawn at random every 64 GCs.

For each workload, the probabilities are the mean choice fractions of r2-dqn-<wl>-s0..4 (selected
checkpoints, measured on test seed 100); evaluated on the test seeds 100-104.  -> results/baselines/mix.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402

from ssdgc.env import run_policy  # noqa: E402
from ssdgc.rule_env import MixBeta  # noqa: E402

WLS = ["uniform", "hc80-20", "hc90-10", "zipf1.0", "shift90-10"]
OUT = "results/baselines/mix.csv"


def main():
    rows = []
    for wl in WLS:
        fr = []
        for s in range(5):
            p = f"runs/r2-dqn-{wl}-s{s}/done.json"
            if os.path.exists(p):
                fr.append(json.load(open(p))["choices_best"][wl])
        if not fr:
            continue
        probs = np.mean(fr, axis=0)
        for seed in (100, 101, 102, 103, 104):
            w = run_policy(MixBeta(probs, 64, seed), wl, seed)["waf"]
            rows.append(["mix-dqn-freq", wl, seed, round(w, 5), " ".join(f"{x:.3f}" for x in probs), len(fr)])
            print(wl, seed, round(w, 4), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["policy", "workload", "seed", "waf", "probs", "n_agents"])
        wr.writerows(rows)


if __name__ == "__main__":
    main()
