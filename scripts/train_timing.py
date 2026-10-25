"""Train one GC-timing agent (idempotent).  Same run-directory layout as scripts/train.py.

    python scripts/train_timing.py --exp-id e6-tabq-lam1-s0 --algo tabq --lam 1.0 --seed 0
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from ssdgc.agents.timing_agents import TabularQ, TimingDQN
from ssdgc.timing_env import TimingEnv, run_timing_policy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--algo", default="tabq", choices=["tabq", "dqn"])
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--reward-scale", type=float, default=0.01,
                    help="multiplies the reward; the tabular agent is invariant to it (Q starts at 0), the DQN is not: "
                         "its network has to reach |Q| ~ |r| / (1 - gamma) with a bounded step size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400_000)
    ap.add_argument("--episode-len", type=int, default=20_000)
    ap.add_argument("--gamma", type=float, default=0.995, help="per millisecond")
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument("--eps-frac", type=float, default=0.5)
    ap.add_argument("--workload", default="hc80-20")
    ap.add_argument("--op", type=float, default=0.28)
    ap.add_argument("--eval-requests", type=int, default=60_000)
    ap.add_argument("--final-eval-seeds", default="100,101,102,103,104")
    a = ap.parse_args()
    torch.set_num_threads(1)
    d = os.path.join(a.runs, a.exp_id)
    os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "done.json")):
        print(f"{a.exp_id}: done already")
        return
    cfg = vars(a).copy()
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"), indent=2)
    kw = dict(gamma=a.gamma, eps_decay_steps=int(a.eps_frac * a.steps), seed=a.seed)
    agent = TabularQ(alpha=a.alpha, **kw) if a.algo == "tabq" else TimingDQN(lr=a.lr, train_every=a.train_every, **kw)
    env = TimingEnv(workload=a.workload, op=a.op, seed=a.seed, lam=a.lam, episode_len=a.episode_len,
                    reward_scale=a.reward_scale)
    log = open(os.path.join(d, "train_log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["episode", "steps", "ep_return", "lat_mean_ms", "lat_p99_ms", "waf", "epsilon", "wall_s"])
    t0 = time.time()
    ep = 0
    while agent.steps < a.steps:
        obs, _ = env.reset(seed=a.seed * 1000 + ep)
        f = env.sim.f
        lat0, h0, c0 = len(env.sim.lat), f.host_writes, f.gc_copies
        ret = 0.0
        while True:
            act = agent.act(obs)
            obs2, r, term, trunc, info = env.step(act)
            agent.observe(obs, act, r, obs2, term, dt_ms=info["dt_ms"])
            ret += r
            obs = obs2
            if term or trunc or agent.steps >= a.steps:
                break
        ep += 1
        s = env.sim.summary(lat0)
        waf = 1.0 + (f.gc_copies - c0) / max(f.host_writes - h0, 1)
        w.writerow([ep, agent.steps, round(ret, 2), round(s["lat_mean_ms"], 4), round(s["lat_p99_ms"], 3),
                    round(waf, 4), round(agent.epsilon(), 4), round(time.time() - t0, 1)])
        log.flush()
    wall = time.time() - t0
    if a.algo == "tabq":
        pickle.dump(agent.state_dict(), open(os.path.join(d, "model.pkl"), "wb"))
    else:
        torch.save(agent.state_dict(), os.path.join(d, "model.pt"))
    seeds = [int(x) for x in a.final_eval_seeds.split(",")]
    res = [run_timing_policy(lambda o: agent.act(o, greedy=True), n_requests=a.eval_requests, seed=s,
                             lam=a.lam, workload=a.workload, op=a.op) for s in seeds]
    json.dump(dict(exp_id=a.exp_id, train_wall_s=round(wall, 1), steps=agent.steps, eval=res, eval_seeds=seeds),
              open(os.path.join(d, "done.json"), "w"), indent=2)
    m = {k: round(float(np.mean([r[k] for r in res])), 3) for k in ("lat_mean_ms", "lat_p99_ms", "lat_p999_ms", "waf")}
    print(f"{a.exp_id}: done in {wall:.0f}s {m}")


if __name__ == "__main__":
    main()
