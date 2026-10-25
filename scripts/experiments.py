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
MAIN_DECISIONS = 200_000


def job(i, args):
    return {"id": i, "args": [str(a) for a in args]}


def e2(nstep=1):
    out = []
    for seed in SEEDS:
        for wl in WORKLOADS:
            for name, extra in ALGOS.items():
                ns = ["--nstep", str(nstep)] if (nstep > 1 and name != "reinforce") else []
                out.append(job(f"e2-{name}-{wl}-s{seed}",
                               extra + ns + ["--workload", wl, "--seed", seed, "--decisions", MAIN_DECISIONS]))
    return out


def e3(main="reinforce"):
    base = ALGOS[main]
    out = []
    grid = [("gamma", "0"), ("gamma", "0.9"), ("gamma", "0.999"),
            ("lr", "1e-4"), ("lr", "1e-3"), ("k", "4"), ("k", "16")]
    for seed in (0, 1, 2):
        for key, val in grid:
            args = [a for a in base]
            if key == "lr":
                args = [a for a in args]
                j = args.index("--lr")
                args[j + 1] = val
            else:
                args += [f"--{key}", val]
            out.append(job(f"e3-{main}-{key}{val}-hc90-10-s{seed}",
                           args + ["--workload", "hc90-10", "--seed", seed, "--decisions", MAIN_DECISIONS]))
    return out


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


def main():
    os.makedirs("queues", exist_ok=True)
    qs = {"e2": e2(), "e3": e3(), "e4": e4(), "e5": e5(), "e7v": e7_victim()}
    for k, v in qs.items():
        json.dump(v, open(f"queues/{k}.json", "w"), indent=1)
        print(k, len(v))


if __name__ == "__main__":
    main()
