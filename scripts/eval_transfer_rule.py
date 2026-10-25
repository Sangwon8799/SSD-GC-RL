"""E5 — does a rule-choosing agent carry over to what it was not trained on?

The agents trained on the pool of five workloads (r2-<algo>-pool-s*) and the single-workload agents of hc90-10
(r2-<algo>-hc90-10-s*) are evaluated (selected checkpoint, test seeds 100-104) on
  * workloads never used for training: zipf0.8, hc95-5
  * hc90-10 at other OP: 7 %, 28 %
together with the fixed rules and the random rule switch on the same cases.  -> results/transfer/transfer_rule.csv
"""
from __future__ import annotations

import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import torch  # noqa: E402

from ssdgc.env import run_policy  # noqa: E402
from ssdgc.rule_env import BETAS, FixedBeta, RandomBeta, RuleAgentPolicy  # noqa: E402
from train_rule import build  # noqa: E402

CASES = [("zipf0.8", 0.14), ("hc95-5", 0.14), ("hc90-10", 0.07), ("hc90-10", 0.28)]
SEEDS = [100, 101, 102, 103, 104]
OUT = "results/transfer/transfer_rule.csv"


def load(d):
    cfg = json.load(open(os.path.join(d, "config.json")))
    ag = build(cfg)
    ag.load_state_dict(torch.load(os.path.join(d, "best.pt"), weights_only=False))
    return cfg, ag


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
    runs = []
    for algo in ("dqn", "reinforce"):
        for s in range(5):
            for kind in ("pool", "hc90-10"):
                d = os.path.join("runs", f"r2-{algo}-{kind}-s{s}")
                if os.path.exists(os.path.join(d, "done.json")):
                    runs.append((os.path.basename(d), f"{algo}:{kind}", d))
    for wl, op in CASES:
        for seed in SEEDS:
            for name in [f"beta{b:g}" for b in BETAS] + ["random-beta"]:
                if (name, wl, str(op), str(seed)) in done:
                    continue
                pol = RandomBeta(64, seed) if name == "random-beta" else FixedBeta(float(name[4:]))
                w.writerow([name, "-", wl, op, seed, round(run_policy(pol, wl, seed, op=op)["waf"], 5)])
                fh.flush()
            for rid, kind, d in runs:
                if (rid, wl, str(op), str(seed)) in done:
                    continue
                cfg, ag = load(d)
                pol = RuleAgentPolicy(ag, cfg["M"], acts=cfg["acts"], drop=cfg["drop"])
                w.writerow([rid, kind, wl, op, seed, round(run_policy(pol, wl, seed, op=op)["waf"], 5)])
                fh.flush()
        print(wl, op, "done", flush=True)


if __name__ == "__main__":
    main()
