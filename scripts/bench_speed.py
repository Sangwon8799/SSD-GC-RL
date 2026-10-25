"""Wall-clock measurements that decide the size of the experiment table.

    python scripts/bench_speed.py --out results/bench_speed.json

Measures (single core, OMP_NUM_THREADS=1):
  * host page write cost of the FTL simulator for three over-provisioning ratios
  * environment reset (fill + warm-up)
  * one RL decision: simulator part vs agent part (DQN with learning)
  * one full evaluation run (fill + warm-up + measurement)
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from ssdgc.env import N_CAND_FEAT, N_GLOB_FEAT, AgentPolicy, VictimEnv, run_policy
from ssdgc.ftl import FTL
from ssdgc.policies import make_policy
from ssdgc.workloads import make_workload
from ssdgc.agents.dqn import DQNAgent


def host_write_cost(op, n_mult=4, B=256, P=64):
    f = FTL(B, P, op)
    f.victim_fn = make_policy("greedy", np.random.default_rng(0))
    w = make_workload("uniform", f.U, 1)
    for l in range(f.U):
        f.write(l)
    for _ in range(2 * f.U):
        f.write(w.next())
    n = n_mult * f.U
    t0 = time.perf_counter()
    for _ in range(n):
        f.write(w.next())
    return (time.perf_counter() - t0) / n * 1e6, f.U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bench_speed.json")
    ap.add_argument("--decisions", type=int, default=4000)
    a = ap.parse_args()
    torch.set_num_threads(1)
    out = {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__,
           "machine": platform.machine(), "cpu_count": os.cpu_count()}
    out["host_write_us"] = {}
    for op in (0.07, 0.14, 0.28):
        us, U = host_write_cost(op)
        out["host_write_us"][str(op)] = round(us, 2)
    env = VictimEnv("hc90-10", op=0.14, episode_len=10**9, seed=0)
    t0 = time.perf_counter()
    obs, _ = env.reset()
    out["env_reset_s"] = round(time.perf_counter() - t0, 3)
    out["U"] = env.U
    res = {}
    for name, kw in (("dqn_b128_te2", dict(batch=128, train_every=2)),
                     ("dqn_b64_te4", dict(batch=64, train_every=4))):
        ag = DQNAgent(8, N_CAND_FEAT, N_GLOB_FEAT, learn_start=500, eps_decay_steps=a.decisions, **kw)
        tsim = tag = 0.0
        hw = 0
        t0 = time.perf_counter()
        for _ in range(a.decisions):
            x0 = time.perf_counter()
            act = ag.act(obs)
            x1 = time.perf_counter()
            obs2, r, te, tr, info = env.step(act)
            x2 = time.perf_counter()
            ag.observe(obs, act, r, obs2, te)
            x3 = time.perf_counter()
            tsim += x2 - x1
            tag += (x1 - x0) + (x3 - x2)
            hw += info["host_writes"]
            obs = obs2
        tot = time.perf_counter() - t0
        res[name] = dict(ms_per_decision=round(tot / a.decisions * 1e3, 3),
                         sim_ms=round(tsim / a.decisions * 1e3, 3), agent_ms=round(tag / a.decisions * 1e3, 3),
                         host_writes_per_decision=round(hw / a.decisions, 2))
    out["decision"] = res
    ag = DQNAgent(8, N_CAND_FEAT, N_GLOB_FEAT)
    t0 = time.perf_counter()
    run_policy(AgentPolicy(ag), "hc90-10", seed=123, op=0.14)
    out["eval_run_agent_s"] = round(time.perf_counter() - t0, 2)
    t0 = time.perf_counter()
    run_policy(make_policy("greedy", np.random.default_rng(0)), "hc90-10", seed=123, op=0.14)
    out["eval_run_greedy_s"] = round(time.perf_counter() - t0, 2)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
