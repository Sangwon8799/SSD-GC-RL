"""E0 — does the simulator behave correctly?  (run before any learning)

    python scripts/validate_sim.py --out results/validation [--workers 2] [--quick]

V1  hand-worked unit tests (tests/test_ftl.py)
V2  replication of a published setting (Xiang & Kurkoski, arXiv:1110.4245, Table I):
    N_p = 256 pages/block, U = 1024 user blocks, rho in {0.15 .. 1.0}, single write frontier,
    uniform random single-page writes.  greedy  vs their simulation column;
    FIFO vs the closed form u = exp(-alpha(1-u));  pure random vs WAF = alpha/(alpha-1).
V3  this project's device (256 blocks x 64 pages, separate GC frontier) at OP 7/14/28 %:
    FIFO and pure random against the same closed forms using the effective capacity.
V4  block-size sensitivity: same total pages, pages per block 32..256 (greedy, FIFO).
V5  long runs with the full consistency check (mapping <-> reverse map <-> valid counts <-> free list).
"""
from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from ssdgc import theory
from ssdgc.ftl import FTL
from ssdgc.policies import make_policy
from ssdgc.workloads import make_workload

XK_SIM = {0.15: 3.97, 0.20: 3.17, 0.25: 2.67, 0.30: 2.35, 1.00: 1.25}   # Table I "Simulation" column


def run(job):
    (tag, n_blocks, ppb, user_pages, op, separate, policy, workload, seed, warm, meas) = job
    f = FTL(n_blocks, ppb, op, separate_gc_frontier=separate, user_pages=user_pages)
    f.victim_fn = make_policy(policy, np.random.default_rng(seed + 1000))
    wl = make_workload(workload, f.U, seed)
    for lpn in range(f.U):
        f.write(lpn)
    for _ in range(int(warm * f.U)):
        f.write(wl.next())
    c0 = f.counters()
    n = int(meas * f.U)
    free_sum = 0
    samples = 0
    t0 = time.perf_counter()
    for i in range(n):
        f.write(wl.next())
        if (i & 63) == 0:
            free_sum += len(f.free)
            samples += 1
    dt = time.perf_counter() - t0
    c1 = f.counters()
    mean_free = free_sum / samples
    n_open = 2 if separate else 1
    alpha_nom = f.T / f.U
    alpha_eff = (f.B - mean_free - n_open / 2) * f.P / f.U
    waf = 1.0 + (c1["copies"] - c0["copies"]) / n
    return dict(tag=tag, n_blocks=n_blocks, ppb=ppb, U=f.U, op=op, separate=int(separate), policy=policy,
                workload=workload, seed=seed, waf=round(waf, 5), alpha_nom=round(alpha_nom, 5),
                alpha_eff=round(alpha_eff, 5), mean_free=round(mean_free, 3), gc=c1["gc"] - c0["gc"],
                secs=round(dt, 2))


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validation")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    root = os.path.join(os.path.dirname(__file__), "..")

    # V1
    r = subprocess.run([sys.executable, "-m", "tests.test_ftl"], cwd=root, capture_output=True, text=True)
    open(os.path.join(a.out, "v1_unit_tests.txt"), "w").write(r.stdout + r.stderr)
    print(r.stdout.strip().splitlines()[-1])
    if r.returncode != 0:
        sys.exit("unit tests failed")

    seeds = [0] if a.quick else [0, 1, 2]
    jobs = []
    # V2  Xiang & Kurkoski setting
    rhos = [0.15, 0.20, 0.25, 0.30, 0.50, 1.00]
    for rho in rhos:
        n_blocks = int(round(1024 * (1 + rho))) + 1
        for pol in ("greedy", "fifo", "pure-random"):
            for s in (seeds if pol != "pure-random" else seeds[:2]):
                jobs.append(("V2", n_blocks, 256, 1024 * 256, rho, False, pol, "uniform", s, 2.0, 2.0))
    # V3  our device
    for op in (0.07, 0.14, 0.28):
        for pol in ("greedy", "fifo", "pure-random"):
            for s in seeds:
                jobs.append(("V3", 256, 64, None, op, True, pol, "uniform", s, 3.0, 4.0))
    # V4  block size sensitivity (same 16384 physical pages, OP 14 %)
    for B, P in ((512, 32), (256, 64), (128, 128), (64, 256)):
        for pol in ("greedy", "fifo"):
            for s in seeds:
                jobs.append(("V4", B, P, None, 0.14, True, pol, "uniform", s, 3.0, 4.0))
    t0 = time.time()
    with Pool(a.workers) as pool:
        rows = []
        for i, row in enumerate(pool.imap_unordered(run, jobs)):
            rows.append(row)
            if i % 10 == 0:
                print(f"{i + 1}/{len(jobs)} {row['tag']} {row['policy']} op={row['op']} waf={row['waf']} ({time.time() - t0:.0f}s)", flush=True)
    rows.sort(key=lambda x: (x["tag"], x["op"], x["ppb"], x["policy"], x["seed"]))
    for row in rows:
        row["theory_fifo_nom"] = round(theory.fifo_waf(row["alpha_nom"]), 4)
        row["theory_fifo_eff"] = round(theory.fifo_waf(row["alpha_eff"]), 4)
        row["theory_random_eff"] = round(theory.random_waf(row["alpha_eff"]), 4)
        row["xk_sim"] = XK_SIM.get(round(row["op"], 2), "") if row["tag"] == "V2" else ""
    write_csv(os.path.join(a.out, "e0_runs.csv"), rows)

    # V5 long-run invariants
    lines = []
    for wl in ("uniform", "hc90-10", "zipf1.0", "shift90-10"):
        for pol in ("greedy", "cb", "fifo", "random", "d8"):
            f = FTL(64, 32, 0.14)
            f.victim_fn = make_policy(pol, np.random.default_rng(7))
            w = make_workload(wl, f.U, 7)
            for lpn in range(f.U):
                f.write(lpn)
            checks = 0
            for i in range(30 * f.U):
                f.write(w.next())
                if i % (2 * f.U) == 0:
                    f.check_invariants()
                    checks += 1
            f.check_invariants()
            lines.append(f"ok {wl} {pol} writes={30 * f.U} checks={checks + 1} waf={f.waf():.4f}")
    open(os.path.join(a.out, "v5_invariants.txt"), "w").write("\n".join(lines) + "\n")
    print(f"V5: {len(lines)} long runs passed every consistency check")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
