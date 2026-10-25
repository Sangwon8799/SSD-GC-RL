"""Train one victim-selection agent (idempotent, resumable).

    python scripts/train.py --exp-id e2-dqn-hc90-10-s0 --algo dqn --workload hc90-10 --seed 0

Writes runs/<exp-id>/: config.json, train_log.csv (one row per episode), eval_log.csv
(periodic greedy evaluation), checkpoint.pt (every --ckpt-every episodes), model.pt and
done.json at the end.  If done.json exists the run is skipped; if checkpoint.pt exists the
run resumes from it (the replay buffer restarts empty — it refills within `learn_start` steps).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from ssdgc.agents.dqn import DQNAgent
from ssdgc.agents.reinforce import ReinforceAgent
from ssdgc.env import N_CAND_FEAT, N_GLOB_FEAT, AgentPolicy, VictimEnv, run_policy
from ssdgc.policies import make_policy


def build_agent(cfg):
    common = dict(gamma=cfg["gamma"], lr=cfg["lr"], hidden=cfg["hidden"], seed=cfg["seed"],
                  reward_shift=cfg["reward_shift"])
    if cfg["algo"] in ("dqn", "ddqn", "dueling"):
        return DQNAgent(cfg["k"], N_CAND_FEAT, N_GLOB_FEAT, double=(cfg["algo"] in ("ddqn", "dueling")),
                        dueling=(cfg["algo"] == "dueling"), nstep=cfg.get("nstep", 1),
                        batch=cfg["batch"], train_every=cfg["train_every"], target_every=cfg["target_every"],
                        learn_start=cfg["learn_start"], eps_decay_steps=int(cfg["eps_frac"] * cfg["decisions"]),
                        eps_end=cfg["eps_end"], buffer=cfg["buffer"], **common)
    if cfg["algo"] == "reinforce":
        return ReinforceAgent(cfg["k"], N_CAND_FEAT, N_GLOB_FEAT, lr_v=cfg["lr_v"], entropy=cfg["entropy"], **common)
    raise KeyError(cfg["algo"])


def evaluate(agent, cfg, workload, seeds, warm=3.0, meas=4.0, op=None):
    out = []
    for s in seeds:
        pol = AgentPolicy(agent, cfg["k"], cfg["n_greedy"], cfg["drop"])
        r = run_policy(pol, workload, s, op=op if op is not None else cfg["op"], warmup=warm, measure=meas)
        out.append(r["waf"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--algo", default="dqn", choices=["dqn", "ddqn", "dueling", "reinforce"])
    ap.add_argument("--workload", default="hc90-10")
    ap.add_argument("--pool", default="", help="comma list: train on a random workload per episode")
    ap.add_argument("--op", type=float, default=0.14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decisions", type=int, default=300_000)
    ap.add_argument("--episode-len", type=int, default=2000)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-v", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--n-greedy", type=int, default=-1, help="-1 = k/2 (mixed candidates); k = greedy-only candidates")
    ap.add_argument("--drop", default="", help="comma list of candidate features to zero (ablation)")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train-every", type=int, default=4)
    ap.add_argument("--target-every", type=int, default=500)
    ap.add_argument("--learn-start", type=int, default=2000)
    ap.add_argument("--buffer", type=int, default=100_000)
    ap.add_argument("--eps-frac", type=float, default=0.3)
    ap.add_argument("--eps-end", type=float, default=0.02)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--nstep", type=int, default=1)
    ap.add_argument("--eval-every", type=int, default=25, help="episodes between quick evaluations")
    ap.add_argument("--ckpt-every", type=int, default=10)
    ap.add_argument("--final-eval-seeds", default="100,101,102,103,104")
    a = ap.parse_args()
    torch.set_num_threads(1)

    run_dir = os.path.join(a.runs, a.exp_id)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "done.json")):
        print(f"{a.exp_id}: done already, skip")
        return
    cfg = dict(exp_id=a.exp_id, algo=a.algo, workload=a.workload, pool=[p for p in a.pool.split(",") if p],
               op=a.op, seed=a.seed, decisions=a.decisions, episode_len=a.episode_len, gamma=a.gamma, lr=a.lr,
               lr_v=a.lr_v, hidden=a.hidden, k=a.k, n_greedy=(a.k // 2 if a.n_greedy < 0 else a.n_greedy),
               drop=[d for d in a.drop.split(",") if d], batch=a.batch, train_every=a.train_every,
               target_every=a.target_every, learn_start=a.learn_start, buffer=a.buffer, eps_frac=a.eps_frac,
               eps_end=a.eps_end, entropy=a.entropy, nstep=a.nstep)
    # reward centring: subtract the mean victim valid-ratio of greedy on this workload (a constant —
    # it does not change the optimal policy, it only keeps Q values small)
    probe = VictimEnv(a.workload if not cfg["pool"] else cfg["pool"][0], op=a.op, seed=10_000 + a.seed)
    probe.reset()
    us = []
    for _ in range(300):
        _, r, _, _, _ = probe.step(0)
        us.append(-r)
    cfg["reward_shift"] = -float(np.mean(us))
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)

    agent = build_agent(cfg)
    env = VictimEnv(a.workload, op=a.op, k=a.k, n_greedy=cfg["n_greedy"], episode_len=a.episode_len,
                    seed=a.seed, workload_pool=cfg["pool"] or None, drop=cfg["drop"])
    start_ep = 0
    wall0 = 0.0
    ck = os.path.join(run_dir, "checkpoint.pt")
    if os.path.exists(ck):
        sd = torch.load(ck, weights_only=False)
        agent.load_state_dict(sd["agent"])
        start_ep, wall0 = sd["episode"], sd["wall"]
        env.rng = np.random.default_rng(sd["env_seed"])
        print(f"{a.exp_id}: resume at episode {start_ep}")
    def trim(path, upto):
        if not os.path.exists(path):
            return
        rows = list(csv.reader(open(path)))
        keep = [rows[0]] + [r for r in rows[1:] if r and int(r[0]) <= upto] if rows else []
        with open(path, "w", newline="") as fh:
            csv.writer(fh).writerows(keep)
    if start_ep > 0:
        trim(os.path.join(run_dir, "train_log.csv"), start_ep)
        trim(os.path.join(run_dir, "eval_log.csv"), start_ep)
    tlog = open(os.path.join(run_dir, "train_log.csv"), "a", newline="")
    tw = csv.writer(tlog)
    if start_ep == 0:
        tlog.truncate(0)
        tw.writerow(["episode", "decisions", "workload", "ep_waf", "ep_return", "epsilon", "loss", "wall_s"])
    elog = open(os.path.join(run_dir, "eval_log.csv"), "a", newline="")
    ew = csv.writer(elog)
    if start_ep == 0:
        elog.truncate(0)
        ew.writerow(["episode", "decisions", "eval_waf", "wall_s"])
    n_episodes = int(np.ceil(a.decisions / a.episode_len))
    t0 = time.time() - wall0
    for ep in range(start_ep, n_episodes):
        obs, info = env.reset(seed=a.seed * 100_000 + ep)
        ret = 0.0
        while True:
            act = agent.act(obs)
            obs2, r, term, trunc, _ = env.step(act)
            if hasattr(agent, "nstep"):
                agent.observe(obs, act, r, obs2, term, truncated=trunc)
            else:
                agent.observe(obs, act, r, obs2, term)
            ret += r
            obs = obs2
            if term or trunc:
                break
        if hasattr(agent, "end_episode"):
            agent.end_episode()
        eps = agent.epsilon() if hasattr(agent, "epsilon") else float("nan")
        tw.writerow([ep + 1, agent.steps, info["workload"], round(env.episode_waf(), 5), round(ret, 3),
                     round(eps, 4), round(agent.last_loss, 6), round(time.time() - t0, 1)])
        tlog.flush()
        if (ep + 1) % a.eval_every == 0 or ep + 1 == n_episodes:
            wl = a.workload if not cfg["pool"] else cfg["pool"][(ep // a.eval_every) % len(cfg["pool"])]
            w = evaluate(agent, cfg, wl, [9_000 + a.seed], warm=2.0, meas=2.0)
            ew.writerow([ep + 1, agent.steps, round(w[0], 5), round(time.time() - t0, 1)])
            elog.flush()
        if (ep + 1) % a.ckpt_every == 0:
            tmp = ck + ".tmp"
            torch.save({"agent": agent.state_dict(), "episode": ep + 1, "wall": time.time() - t0,
                        "env_seed": int(env.rng.integers(1 << 31))}, tmp)
            os.replace(tmp, ck)
    train_wall = time.time() - t0
    torch.save(agent.state_dict(), os.path.join(run_dir, "model.pt"))
    # final evaluation: frozen greedy policy, full protocol, evaluation seeds (never used in training)
    seeds = [int(s) for s in a.final_eval_seeds.split(",")]
    final = {}
    wls = cfg["pool"] or [a.workload]
    for wl in wls:
        final[wl] = evaluate(agent, cfg, wl, seeds)
    done = dict(exp_id=a.exp_id, train_wall_s=round(train_wall, 1), decisions=agent.steps, final_waf=final,
                eval_seeds=seeds)
    json.dump(done, open(os.path.join(run_dir, "done.json"), "w"), indent=2)
    if os.path.exists(ck):
        os.remove(ck)
    print(f"{a.exp_id}: done in {train_wall:.0f}s  final={ {k: round(float(np.mean(v)), 4) for k, v in final.items()} }")


if __name__ == "__main__":
    main()
