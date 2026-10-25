"""Baselines for the rule-choosing formulation (test seeds 100-104, OP 14 % unless given):
fixed beta in (0, 0.5, 1, 2) and a uniformly random beta every 64 GCs.  -> results/baselines/rule.csv (idempotent)

    python scripts/run_rule_baselines.py [--op 0.14] [--workloads uniform,hc80-20,...]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from ssdgc.env import run_policy  # noqa: E402
from ssdgc.rule_env import BETAS, AlternateBeta, FixedBeta, RandomBeta  # noqa: E402

OUT = "results/baselines/rule.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--op", type=float, default=0.14)
    ap.add_argument("--workloads", default="uniform,hc80-20,hc90-10,zipf1.0,shift90-10")
    a = ap.parse_args()
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add((r["policy"], r["workload"], r["op"], r["seed"]))
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["policy", "workload", "op", "seed", "waf"])
    for wl in a.workloads.split(","):
        for seed in (100, 101, 102, 103, 104):
            for name in [f"beta{b:g}" for b in BETAS] + ["random-beta", "alt0-1", "alt0-2"]:
                if (name, wl, str(a.op), str(seed)) in done:
                    continue
                if name == "random-beta":
                    pol = RandomBeta(64, seed)
                elif name.startswith("alt"):
                    pol = AlternateBeta(tuple(float(x) for x in name[3:].split("-")), 64)
                else:
                    pol = FixedBeta(float(name[4:]))
                r = run_policy(pol, wl, seed, op=a.op)
                w.writerow([name, wl, a.op, seed, round(r["waf"], 5)])
                fh.flush()
        print(wl, "done", flush=True)


if __name__ == "__main__":
    main()
