"""E1 — the rule policies on every workload and over-provisioning ratio.

    python scripts/run_baselines.py --out results/baselines/e1.csv [--workers 2]

Idempotent: rows already present in the CSV (same key) are skipped, new rows are appended.
Evaluation protocol (same as for the agents): fill, warm up 3 x U writes under the policy,
measure WAF over the next 4 x U writes.  Seeds 100..104 are evaluation seeds (training uses others).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from ssdgc.env import run_policy
from ssdgc.policies import make_policy

POLICIES = ["greedy", "cb", "fifo", "random", "d8"]
WORKLOADS = ["uniform", "hc80-20", "hc90-10", "zipf1.0", "shift90-10"]
OPS = [0.07, 0.14, 0.28]
SEEDS = [100, 101, 102, 103, 104]
FIELDS = ["policy", "workload", "op", "seed", "waf", "gc", "copies", "host", "erase_mean", "erase_std", "erase_max", "secs"]


def job(args):
    pol, wl, op, seed = args
    t0 = time.perf_counter()
    r = run_policy(make_policy(pol, np.random.default_rng(seed + 5000)), wl, seed, op=op)
    r.update(policy=pol, workload=wl, op=op, seed=seed, secs=round(time.perf_counter() - t0, 2))
    r["waf"] = round(r["waf"], 5)
    r["erase_std"] = round(r["erase_std"], 4)
    r["erase_mean"] = round(r["erase_mean"], 4)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/baselines/e1.csv")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--policies", default=",".join(POLICIES))
    ap.add_argument("--workloads", default=",".join(WORKLOADS))
    ap.add_argument("--ops", default=",".join(map(str, OPS)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        for r in csv.DictReader(open(a.out)):
            done.add((r["policy"], r["workload"], float(r["op"]), int(r["seed"])))
    jobs = [(p, w, float(o), int(s)) for p in a.policies.split(",") for w in a.workloads.split(",")
            for o in a.ops.split(",") for s in a.seeds.split(",")]
    jobs = [j for j in jobs if j not in done]
    print(f"{len(done)} rows present, {len(jobs)} to run", flush=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        t0 = time.time()
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            w.writerow({k: r[k] for k in FIELDS})
            fh.flush()
            if i % 25 == 0:
                print(f"{i + 1}/{len(jobs)} {r['policy']} {r['workload']} op={r['op']} waf={r['waf']} ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
