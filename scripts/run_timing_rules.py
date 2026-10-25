"""E6 rule baselines for GC timing: foreground-only, idle-watermark rules (W, tau) and two-level rules.

    python scripts/run_timing_rules.py --out results/timing/rules.csv [--workers 2]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ssdgc.timing_env import RulePolicy, run_timing_policy

SEEDS = [100, 101, 102, 103, 104]
FIELDS = ["rule", "W", "tau_ms", "W_emerg", "seed", "lat_mean_ms", "lat_p50_ms", "lat_p99_ms", "lat_p999_ms",
          "lat_max_ms", "fg_frac", "waf", "bg_gcs", "secs"]


def grid():
    out = [(0, 0.0, 0)]
    for W in (3, 4, 6, 8, 12, 16):
        for tau in (0.0, 0.5, 1.0, 2.0, 4.0, 8.0):
            out.append((W, tau, 0))
    for W in (6, 8, 12):
        for tau in (1.0, 2.0, 4.0):
            for We in (3, 4, 5):
                out.append((W, tau, We))
    return out


def job(x):
    (W, tau, We), seed = x
    t0 = time.perf_counter()
    pol = RulePolicy(W, tau, We)
    r = run_timing_policy(pol, n_requests=60000, seed=seed)
    r.update(rule=pol.name, W=W, tau_ms=tau, W_emerg=We, seed=seed, secs=round(time.perf_counter() - t0, 2))
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/timing/rules.csv")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        done = {(r["rule"], int(r["seed"])) for r in csv.DictReader(open(a.out))}
    jobs = [(g, s) for g in grid() for s in SEEDS if (RulePolicy(*g).name, s) not in done]
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as fh, Pool(a.workers) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, r in enumerate(pool.imap_unordered(job, jobs)):
            w.writerow({k: (round(r[k], 4) if isinstance(r[k], float) else r[k]) for k in FIELDS})
            fh.flush()
            if i % 20 == 0:
                print(f"{i + 1}/{len(jobs)} {r['rule']} p99={r['lat_p99_ms']:.2f} waf={r['waf']:.3f}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
