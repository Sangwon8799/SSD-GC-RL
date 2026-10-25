"""Victim statistics of the fixed rules (same protocol as trace_rule.py: test seed 100, fill, 3U warm-up,
second half of 4U measured).  -> results/analysis/pergc_rules_<workload>.csv"""
from __future__ import annotations

import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import numpy as np  # noqa: E402

from ssdgc.ftl import FTL, GCW  # noqa: E402
from ssdgc.rule_env import AlternateBeta, FixedBeta  # noqa: E402
from ssdgc.workloads import make_workload  # noqa: E402


def stats(pol, wl_name, seed=100, op=0.14):
    f = FTL(256, 64, op)
    rec = []

    def wrapped(ftl):
        v = pol(ftl)
        rec.append((ftl.valid[v] / ftl.P, int(ftl.origin[v] == GCW), ftl.now - ftl.close_time[v]))
        return v

    f.victim_fn = wrapped
    wl = make_workload(wl_name, f.U, seed)
    for lpn in range(f.U):
        f.write(lpn)
    for _ in range(3 * f.U):
        f.write(wl.next())
    rec.clear()
    c0, h0 = f.gc_copies, f.host_writes
    for _ in range(4 * f.U):
        f.write(wl.next())
    g = np.array(rec[len(rec) // 2:], dtype=float)
    return [len(rec), round(float(g[:, 0].mean()), 4), round(float(g[:, 1].mean()), 4), round(float(np.median(g[:, 2])), 1),
            round(1 + (f.gc_copies - c0) / (f.host_writes - h0), 4)]


def main():
    wl = sys.argv[1] if len(sys.argv) > 1 else "hc90-10"
    os.makedirs("results/analysis", exist_ok=True)
    out = f"results/analysis/pergc_rules_{wl}.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["policy", "n_gc", "mean_victim_u", "gc_block_share", "median_age", "waf"])
        for name, pol in (("beta0", FixedBeta(0.0)), ("beta1", FixedBeta(1.0)), ("beta2", FixedBeta(2.0)),
                          ("alt0-1", AlternateBeta((0.0, 1.0), 64))):
            r = stats(pol, wl)
            w.writerow([name] + r)
            print(wl, name, r, flush=True)


if __name__ == "__main__":
    main()
