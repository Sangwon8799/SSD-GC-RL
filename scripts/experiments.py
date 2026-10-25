"""The experiment table as code: writes queues/<name>.json for scripts/run_queue.py.

    python scripts/experiments.py            # writes every queue
Each job = {"id": ..., "args": [...train.py arguments...]}.  IDs are stable, so re-running a
queue skips finished runs (runs/<id>/done.json).
"""
from __future__ import annotations

import json
import os

WORKLOADS = ["uniform", "hc80-20", "hc90-10", "zipf1.0", "shift90-10"]
SEEDS = [0, 1, 2, 3, 4]
DQN_COMMON = ["--lr", "1e-4"]
ALGOS = {  # name -> extra args
    "dqn": ["--algo", "dqn"] + DQN_COMMON,
    "ddqn": ["--algo", "ddqn"] + DQN_COMMON,
    "dueling": ["--algo", "dueling"] + DQN_COMMON,
    "reinforce": ["--algo", "reinforce", "--lr", "3e-4"],
}
MAIN_DECISIONS = 150_000


EVAL = ["--eval-every", "10"]   # validation every 10 episodes = 20k decisions (also the checkpoint choice)


def job(i, args):
    return {"id": i, "args": [str(a) for a in args] + EVAL}


def e2(nstep=1):
    out = []
    for seed in SEEDS:
        for wl in WORKLOADS:
            for name, extra in ALGOS.items():
                ns = ["--nstep", str(nstep)] if (nstep > 1 and name != "reinforce") else []
                out.append(job(f"e2-{name}-{wl}-s{seed}",
                               extra + ns + ["--workload", wl, "--seed", seed, "--decisions", MAIN_DECISIONS]))
    return out


def e3(main="reinforce", grid=None, seeds=(0, 1, 2)):
    """Hyper-parameter comparison on hc90-10: one knob changed at a time from the main setting."""
    base = ALGOS[main]
    out = []
    grid = grid or [("gamma", "0"), ("gamma", "0.9"), ("gamma", "0.999"),
                    ("lr", "1e-4"), ("lr", "1e-3"), ("k", "4"), ("k", "16")]
    for seed in seeds:
        for key, val in grid:
            args = list(base)
            if key == "lr":
                j = args.index("--lr")
                args[j + 1] = val
            else:
                args += [f"--{key}", val]
            out.append(job(f"e3-{main}-{key.replace('-', '')}{val}-hc90-10-s{seed}",
                           args + ["--workload", "hc90-10", "--seed", seed, "--decisions", MAIN_DECISIONS]))
    return out


E3_DDQN = [("gamma", "0.9"), ("gamma", "0.999"), ("lr", "3e-5"), ("lr", "3e-4"), ("target-every", "2000")]


def e4(main="reinforce"):
    base = ALGOS[main]
    out = []
    variants = [("noage", ["--drop", "age"]), ("noinval", ["--drop", "since_inval,inval_rate"]),
                ("noorigin", ["--drop", "gc_origin"]), ("norank", ["--drop", "greedy_rank,cb_rank"]),
                ("greedycand", ["--n-greedy", "8"])]
    for seed in (0, 1, 2):
        for name, extra in variants:
            out.append(job(f"e4-{main}-{name}-hc90-10-s{seed}",
                           base + extra + ["--workload", "hc90-10", "--seed", seed, "--decisions", MAIN_DECISIONS]))
    return out


def e5(main="reinforce"):
    out = []
    for seed in SEEDS:
        out.append(job(f"e5-{main}-pool-s{seed}",
                       ALGOS[main] + ["--workload", "hc90-10", "--pool", ",".join(WORKLOADS), "--seed", seed,
                                      "--decisions", 2 * MAIN_DECISIONS]))
    return out


def e7_victim(main="reinforce"):
    return [job(f"e7-{main}-hc80-20-op28-s{seed}",
                ALGOS[main] + ["--workload", "hc80-20", "--op", "0.28", "--seed", seed, "--decisions", MAIN_DECISIONS])
            for seed in SEEDS]


LAMS = ["300", "1000", "3000", "10000"]


def e6():
    """GC timing (gamma 0.9995 per ms, OP 28 %, hc80-20): tabular Q 2M steps x 4 lambdas x 5 seeds,
    then DQN 1M steps (one update every 8 steps) x 2 lambdas x 5 seeds."""
    out = []
    for lam in LAMS:
        for seed in SEEDS:
            out.append({"id": f"e6-tabq-lam{lam}-s{seed}", "script": "train_timing.py",
                        "args": ["--algo", "tabq", "--lam", lam, "--gamma", "0.9995", "--seed", str(seed),
                                 "--steps", "2000000"]})
    for lam in ("300", "3000"):
        for seed in SEEDS:
            out.append({"id": f"e6-dqn-lam{lam}-s{seed}", "script": "train_timing.py",
                        "args": ["--algo", "dqn", "--lam", lam, "--gamma", "0.9995", "--seed", str(seed),
                                 "--steps", "1000000", "--train-every", "8"]})
    return out


REST_WORKLOADS = ("hc90-10", "zipf1.0")   # DQN and Dueling DQN only here (wall-clock; see the page's log 05:35)


# ---- direct block choice, continuing task (the formulation that did not learn well: evidence for section 7)
DIRECT = {
    "reinforce": ["--algo", "reinforce", "--lr", "1e-3", "--episode-len", "500", "--bootstrap", "1",
                  "--eval-stochastic", "1", "--eval-every", "40"],
    "ddqn": ["--algo", "ddqn", "--lr", "1e-4", "--eval-every", "10"],
}


def e2a():
    """One training seed per workload (a diagnostic of the first design, not a main result); hc80-20 gets
    a second seed because the discount-factor comparison (e3a) uses it."""
    out = []
    for seed in (0, 1):
        for wl in (WORKLOADS if seed == 0 else ("hc80-20",)):
            for name, extra in DIRECT.items():
                out.append({"id": f"d2-{name}-g0.999-{wl}-s{seed}",
                            "args": extra + ["--continuing", "1", "--gamma", "0.999", "--workload", wl,
                                             "--seed", str(seed), "--decisions", str(MAIN_DECISIONS)]})
    return out


def e3a():
    out = []
    for seed in (0, 1, 2):
        for name, gammas in (("reinforce", ("0", "0.99")), ("ddqn", ("0.99",))):
            for g in gammas:
                out.append({"id": f"d3-{name}-g{g}-hc80-20-s{seed}",
                            "args": DIRECT[name] + ["--continuing", "1", "--gamma", g, "--workload", "hc80-20",
                                                    "--seed", str(seed), "--decisions", str(MAIN_DECISIONS)]})
    return out


# ---- rule choice (the second formulation): the main experiment from 06:40
RULE_DECISIONS = 400_000
POOL = ",".join(WORKLOADS)


def rjob(i, args):
    return {"id": i, "script": "train_rule.py", "args": [str(a) for a in args]}


def e2b():
    out = []
    for seed in SEEDS:
        for algo in ("dqn", "reinforce"):
            ev = ["--eval-stochastic", "1"] if algo == "reinforce" else []   # a policy-gradient agent's policy is the distribution
            for wl in WORKLOADS:
                out.append(rjob(f"r2-{algo}-{wl}-s{seed}", ["--algo", algo, "--workload", wl, "--seed", seed,
                                                            "--decisions", RULE_DECISIONS] + ev))
            out.append(rjob(f"r2-{algo}-pool-s{seed}", ["--algo", algo, "--workload", WORKLOADS[0], "--pool", POOL,
                                                       "--seed", seed, "--decisions", RULE_DECISIONS] + ev))
    return out


def e3b():
    out = []
    for seed in (0, 1, 2):
        for name, extra in (("gamma0", ["--gamma", "0"]), ("gamma0.5", ["--gamma", "0.5"]), ("gamma0.99", ["--gamma", "0.99"]),
                            ("M16", ["--M", "16", "--episode-len", "200"]), ("M256", ["--M", "256", "--episode-len", "12"])):
            out.append(rjob(f"r3-dqn-{name}-pool-s{seed}", ["--algo", "dqn", "--workload", WORKLOADS[0], "--pool", POOL,
                                                           "--seed", seed, "--decisions", RULE_DECISIONS] + extra))
    return out


def e4b():
    out = []
    for seed in (0, 1, 2):
        for name, extra in (("nostate", ["--drop", ",".join(["mean_u", "std_u", "full_frac", "gc_block_frac", "victim_u_last",
                                                              "waf_last", "young_inval_share", "prev_a0", "prev_a1",
                                                              "prev_a2", "prev_a3"])]),
                            ("noudist", ["--drop", "mean_u,std_u,full_frac"]),
                            ("noinval", ["--drop", "young_inval_share"]),
                            ("prevonly", ["--drop", "mean_u,std_u,full_frac,gc_block_frac,victim_u_last,waf_last,young_inval_share"]),
                            ("two-rules", ["--acts", "0,2"])):
            out.append(rjob(f"r4-dqn-{name}-pool-s{seed}", ["--algo", "dqn", "--workload", WORKLOADS[0], "--pool", POOL,
                                                           "--seed", seed, "--decisions", RULE_DECISIONS] + extra))
    return out


def e7b():
    return [rjob(f"r7-dqn-hc80-20-op28-s{seed}", ["--algo", "dqn", "--workload", "hc80-20", "--op", "0.28", "--seed", seed,
                                                  "--decisions", RULE_DECISIONS]) for seed in SEEDS]


def ordered():
    """One queue, most important first: if time runs out, what is finished is still a result."""
    e2_all = e2()
    core = [j for j in e2_all if "-reinforce-" in j["id"] or "-ddqn-" in j["id"]]
    rest = [j for j in e2_all if j not in core and any(f"-{w}-s" in j["id"] for w in REST_WORKLOADS)]
    return core + e4() + e3() + e5() + e7_victim() + e3("ddqn", E3_DDQN) + rest


def main():
    os.makedirs("queues", exist_ok=True)
    q = ordered()
    ids = [j["id"] for j in q]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    json.dump(q, open("queues/main.json", "w"), indent=1)
    json.dump(e6(), open("queues/timing.json", "w"), indent=1)
    rule = e2b() + e7b() + e4b() + e3b()
    json.dump(rule, open("queues/rule.json", "w"), indent=1)
    print("rule", len(rule))
    d = e3a() + e2a()
    json.dump(d, open("queues/direct.json", "w"), indent=1)
    print("direct", len(d))
    from collections import Counter
    print(len(q), dict(Counter(i.split("-")[0] for i in ids)))


if __name__ == "__main__":
    main()
