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


def ordered():
    """One queue, most important first: if time runs out, what is finished is still a result."""
    e2_all = e2()
    core = [j for j in e2_all if "-reinforce-" in j["id"] or "-ddqn-" in j["id"]]
    rest = [j for j in e2_all if j not in core]
    return core + e4() + e3() + e5() + e3("ddqn", E3_DDQN) + rest + e7_victim()


def main():
    os.makedirs("queues", exist_ok=True)
    q = ordered()
    ids = [j["id"] for j in q]
    assert len(ids) == len(set(ids)), "duplicate job ids"
    json.dump(q, open("queues/main.json", "w"), indent=1)
    from collections import Counter
    print(len(q), dict(Counter(i.split("-")[0] for i in ids)))


if __name__ == "__main__":
    main()
