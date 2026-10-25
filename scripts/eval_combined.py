"""E7 — what to erase x when to erase, together (OP 28 %, hc80-20, the timing setting).

victim policy  : greedy | cost-benefit | learned REINFORCE (e7-reinforce-hc80-20-op28-s<i>, best.pt)
timing policy  : foreground only | rule (W=8, tau=2 ms, W_e=4) | learned tabular Q (e6-tabq-lam3000-s<i>)
Each of the 9 pairs is run on the test seeds 100-104 (the i-th seed uses the i-th trained model) for
60k requests; latency mean / p99 / p99.9 and WAF are written to results/combined/combined.csv.
"""
from __future__ import annotations

import csv
import json
import os
import pickle
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import torch  # noqa: E402

from ssdgc.agents.timing_agents import TabularQ  # noqa: E402
from ssdgc.env import AgentPolicy  # noqa: E402
from ssdgc.timing_env import RulePolicy, run_timing_policy  # noqa: E402
from train import build_agent  # noqa: E402

SEEDS = [100, 101, 102, 103, 104]
OUT = "results/combined/combined.csv"
KW = dict(workload="hc80-20", op=0.28)


def victim(name, i):
    if name in ("greedy", "cb"):
        return name
    d = f"runs/e7-reinforce-hc80-20-op28-s{i}"
    cfg = json.load(open(os.path.join(d, "config.json")))
    ag = build_agent(cfg)
    ag.load_state_dict(torch.load(os.path.join(d, "best.pt"), weights_only=False))
    return AgentPolicy(ag, cfg["k"], cfg["n_greedy"], cfg["drop"])


def timing(name, i):
    if name == "fg-only":
        return RulePolicy(0)
    if name == "rule":
        return RulePolicy(8, 2.0, 4)
    ag = TabularQ()
    ag.load_state_dict(pickle.load(open(f"runs/e6-tabq-lam3000-s{i}/model.pkl", "rb")))
    return lambda o: ag.act(o, greedy=True)


def main():
    torch.set_num_threads(1)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    done = set()
    if os.path.exists(OUT):
        for r in csv.DictReader(open(OUT)):
            done.add((r["victim"], r["timing"], r["seed"]))
    new = not os.path.exists(OUT)
    fh = open(OUT, "a", newline="")
    w = csv.writer(fh)
    if new:
        w.writerow(["victim", "timing", "seed", "lat_mean_ms", "lat_p99_ms", "lat_p999_ms", "waf", "bg_gcs"])
    for vn in ("greedy", "cb", "reinforce"):
        for tn in ("fg-only", "rule", "tabq"):
            for i, seed in enumerate(SEEDS):
                if (vn, tn, str(seed)) in done:
                    continue
                r = run_timing_policy(timing(tn, i), n_requests=60_000, seed=seed, lam=3000.0, victim=victim(vn, i), **KW)
                w.writerow([vn, tn, seed, round(r["lat_mean_ms"], 4), round(r["lat_p99_ms"], 3),
                            round(r["lat_p999_ms"], 3), round(r["waf"], 4), r["bg_gcs"]])
                fh.flush()
                print(vn, tn, seed, round(r["lat_mean_ms"], 3), round(r["lat_p99_ms"], 2), round(r["waf"], 3), flush=True)


if __name__ == "__main__":
    main()
