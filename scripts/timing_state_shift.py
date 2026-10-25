"""Why does the timing agent trained with greedy victims stop collecting with other victim policies (E7)?
Record, for each victim policy, the 'next victim u' feature the timing agent sees at its idle decisions
(foreground-only timing so that the agent's own choices do not change the state), and how often the
trained tabular agent (e6-tabq-lam3000-s0) would choose "GC now" in those states.
    -> results/combined/state_shift.csv
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
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ssdgc.agents.timing_agents import TabularQ  # noqa: E402
from ssdgc.rule_env import RuleAgentPolicy  # noqa: E402
from ssdgc.timing_env import run_timing_policy  # noqa: E402
from train_rule import build  # noqa: E402


def victim(name):
    if name in ("greedy", "cb"):
        return name
    d = "runs/r7-dqn-hc80-20-op28-s0"
    cfg = json.load(open(os.path.join(d, "config.json")))
    ag = build(cfg)
    ag.load_state_dict(torch.load(os.path.join(d, "best.pt"), weights_only=False))
    return RuleAgentPolicy(ag, cfg["M"], acts=cfg["acts"], drop=cfg["drop"])


def main():
    torch.set_num_threads(1)
    tq = TabularQ()
    tq.load_state_dict(pickle.load(open("runs/e6-tabq-lam3000-s0/model.pkl", "rb")))
    rows = []
    for vn in ("greedy", "cb", "rule-dqn"):
        us, gcs = [], []

        def pol(o):
            if o[1] > 0:                       # an idle decision
                us.append(float(o[3]))
                gcs.append(tq.act(o, greedy=True))
            return 0                           # foreground only: never collect in idle

        run_timing_policy(pol, n_requests=20_000, seed=100, lam=3000.0, victim=victim(vn), workload="hc80-20", op=0.28)
        u = np.array(us)
        rows.append([vn, len(u), round(float(np.median(u)), 4), round(float(np.percentile(u, 10)), 4),
                     round(float(np.percentile(u, 90)), 4), round(float(np.mean(gcs)), 4)])
        print(rows[-1], flush=True)
    os.makedirs("results/combined", exist_ok=True)
    with open("results/combined/state_shift.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["victim", "idle_decisions", "next_u_median", "next_u_p10", "next_u_p90", "tabq_gc_share"])
        w.writerows(rows)


if __name__ == "__main__":
    main()
