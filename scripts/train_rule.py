"""Train one rule-choosing agent (idempotent).  Same run-directory layout as scripts/train.py.

    python scripts/train_rule.py --exp-id r2-dqn-hc80-20-s0 --algo dqn --workload hc80-20 --seed 0

Budget is counted in GC decisions (like train.py); one agent step = M decisions with the chosen rule.
Validation (two seeds, never used for training or testing) every --eval-every episodes keeps the best
checkpoint; the final score is the test seeds 100-104 (fill, 3U warm-up, 4U measurement under the policy).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402
import torch  # noqa: E402

from ssdgc.agents.rule_agents import RuleDQN, RuleReinforce  # noqa: E402
from ssdgc.env import run_policy  # noqa: E402
from ssdgc.rule_env import BETAS, N_RULE_FEAT, FixedBeta, RuleAgentPolicy, RuleEnv  # noqa: E402


def build(cfg):
    kw = dict(gamma=cfg["gamma"], lr=cfg["lr"], hidden=cfg["hidden"], seed=cfg["seed"], reward_shift=cfg["reward_shift"])
    n_act = len(cfg["acts"])
    if cfg["algo"] == "dqn":
        return RuleDQN(N_RULE_FEAT, n_act, eps_decay_steps=int(cfg["eps_frac"] * cfg["n_steps"]), **kw)
    return RuleReinforce(N_RULE_FEAT, n_act, entropy=cfg["entropy"], **kw)


def evaluate(agent, cfg, workload, seeds, warm=3.0, meas=4.0, op=None, record=None):
    out = []
    for s in seeds:
        pol = RuleAgentPolicy(agent, cfg["M"], stochastic=bool(cfg["eval_stochastic"]), acts=cfg["acts"], drop=cfg["drop"])
        r = run_policy(pol, workload, s, op=op if op is not None else cfg["op"], warmup=warm, measure=meas)
        out.append(r["waf"])
        if record is not None:
            ch = np.array(pol.choices[len(pol.choices) // 2:])
            record.append([float((ch == k).mean()) for k in range(len(BETAS))])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--algo", default="dqn", choices=["dqn", "reinforce"])
    ap.add_argument("--workload", default="hc80-20")
    ap.add_argument("--pool", default="")
    ap.add_argument("--op", type=float, default=0.14)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--decisions", type=int, default=400_000)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--episode-len", type=int, default=50)
    ap.add_argument("--gamma", type=float, default=0.9)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--entropy", type=float, default=0.01)
    ap.add_argument("--eps-frac", type=float, default=0.3)
    ap.add_argument("--acts", default="0,1,2,3", help="indices into BETAS = (0, 0.5, 1, 2)")
    ap.add_argument("--drop", default="", help="observation features to zero (ablation)")
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--eval-stochastic", type=int, default=0)
    ap.add_argument("--final-eval-seeds", default="100,101,102,103,104")
    a = ap.parse_args()
    torch.set_num_threads(1)
    run_dir = os.path.join(a.runs, a.exp_id)
    os.makedirs(run_dir, exist_ok=True)
    if os.path.exists(os.path.join(run_dir, "done.json")):
        print(f"{a.exp_id}: done already")
        return
    acts = [int(x) for x in a.acts.split(",")]
    drop = [d for d in a.drop.split(",") if d]
    pool = [p for p in a.pool.split(",") if p]
    n_steps = a.decisions // a.M
    # reward centring: mean victim u of cost-benefit on this workload (a constant; optimal policy unchanged)
    probe = RuleEnv(a.workload if not pool else pool[0], op=a.op, M=a.M, seed=10_000 + a.seed, acts=[2])
    probe.reset()
    shift = -float(np.mean([probe.step(0)[1] for _ in range(20)]))
    cfg = dict(algo=a.algo, workload=a.workload, pool=pool, op=a.op, seed=a.seed, decisions=a.decisions, M=a.M,
               episode_len=a.episode_len, gamma=a.gamma, lr=a.lr, hidden=a.hidden, entropy=a.entropy,
               eps_frac=a.eps_frac, acts=acts, drop=drop, eval_stochastic=a.eval_stochastic, n_steps=n_steps,
               reward_shift=-shift, betas=[BETAS[k] for k in acts])
    json.dump(cfg, open(os.path.join(run_dir, "config.json"), "w"), indent=2)
    agent = build(cfg)
    env = RuleEnv(a.workload, op=a.op, M=a.M, episode_len=a.episode_len, seed=a.seed, workload_pool=pool or None,
                  acts=acts, drop=drop)
    val_wls = pool or [a.workload]
    val_seeds = [9_000 + a.seed] if pool else [9_000 + a.seed, 9_500 + a.seed]
    tlog = open(os.path.join(run_dir, "train_log.csv"), "w", newline="")
    tw = csv.writer(tlog)
    tw.writerow(["episode", "decisions", "workload", "ep_waf", "ep_return", "epsilon", "loss", "wall_s"] +
                [f"a{k}" for k in range(len(acts))])
    elog = open(os.path.join(run_dir, "eval_log.csv"), "w", newline="")
    ew = csv.writer(elog)
    ew.writerow(["episode", "decisions", "eval_waf", "wall_s"])
    best_val, best_ep = float("inf"), 0
    best_path = os.path.join(run_dir, "best.pt")
    t0 = time.time()
    ep = 0
    while agent.steps < n_steps:
        obs, info = env.reset(seed=a.seed * 100_000 + ep)
        ret, counts = 0.0, np.zeros(len(acts))
        while True:
            act = agent.act(obs)
            counts[act] += 1
            obs2, r, term, trunc, _ = env.step(act)
            agent.observe(obs, act, r, obs2, term, truncated=trunc)
            ret += r
            obs = obs2
            if term or trunc or agent.steps >= n_steps:
                break
        if hasattr(agent, "end_episode"):
            agent.end_episode(last_obs=obs)
        ep += 1
        eps = agent.epsilon() if hasattr(agent, "epsilon") else float("nan")
        tw.writerow([ep, agent.steps * a.M, info["workload"], round(env.episode_waf(), 5), round(ret, 4), round(eps, 4),
                     round(agent.last_loss, 6), round(time.time() - t0, 1)] + list((counts / counts.sum()).round(3)))
        tlog.flush()
        if ep % a.eval_every == 0 or agent.steps >= n_steps:
            w = [x for wl in val_wls for x in evaluate(agent, cfg, wl, val_seeds, warm=2.0, meas=2.0)]
            val = float(np.mean(w))
            ew.writerow([ep, agent.steps * a.M, round(val, 5), round(time.time() - t0, 1)])
            elog.flush()
            if val < best_val:
                best_val, best_ep = val, ep
                torch.save(agent.state_dict(), best_path)
    wall = time.time() - t0
    trained_steps = agent.steps
    torch.save(agent.state_dict(), os.path.join(run_dir, "model.pt"))
    seeds = [int(x) for x in a.final_eval_seeds.split(",")]
    wls = pool or [a.workload]
    final = {wl: evaluate(agent, cfg, wl, seeds) for wl in wls}
    other = {}
    if a.algo == "reinforce":                       # the other way of acting, for the record
        cfg2 = dict(cfg, eval_stochastic=0 if cfg["eval_stochastic"] else 1)
        other = {wl: evaluate(agent, cfg2, wl, seeds) for wl in wls}
    agent.load_state_dict(torch.load(best_path, weights_only=False))
    choices = {}
    best = {}
    for wl in wls:
        rec = []
        best[wl] = evaluate(agent, cfg, wl, seeds, record=rec)
        choices[wl] = [round(float(x), 4) for x in np.mean(rec, axis=0)]
    done = dict(exp_id=a.exp_id, train_wall_s=round(wall, 1), decisions=trained_steps * a.M, best_decisions=agent.steps * a.M,
                final_waf=final, best_waf=best,
                best_val=round(best_val, 5), best_ep=best_ep, val_seeds=val_seeds, eval_seeds=seeds,
                choices_best=choices, betas=list(BETAS), final_waf_other_mode=other,
                eval_mode="stochastic" if cfg["eval_stochastic"] else "argmax")
    json.dump(done, open(os.path.join(run_dir, "done.json"), "w"), indent=2)
    print(f"{a.exp_id}: done in {wall:.0f}s  final={ {k: round(float(np.mean(v)), 4) for k, v in final.items()} }"
          f"  best={ {k: round(float(np.mean(v)), 4) for k, v in best.items()} } (ep {best_ep})  choices={choices}")


if __name__ == "__main__":
    _ = FixedBeta
    main()
