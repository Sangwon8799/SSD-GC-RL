"""V6 — the time model against queueing theory (M/D/1, Pollaczek-Khinchine), GC switched off.

    python scripts/validate_timing.py --out results/validation/v6_md1.csv
A fresh, empty device (1024 blocks) serves 50,000 single-page writes arriving as one Poisson
stream; no GC can happen, so each request needs exactly t_w = 100 us and the mean waiting time
must match  W_q = rho * t_w / (2 (1 - rho)).
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from ssdgc.timing import TimedSim, md1_wait_us


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/validation/v6_md1.csv")
    a = ap.parse_args()
    rows = []
    for rho in (0.2, 0.4, 0.6, 0.8):
        gap = 100.0 / rho
        for seed in (0, 1, 2):
            sim = TimedSim("uniform", n_blocks=1024, precondition=False, gap_us=gap, burst_writes=10 ** 9,
                           off_ms=0.0, seed=seed)
            waits = []
            for _ in range(50_000):
                arr = sim.next_arr
                start = max(arr, sim.server_free)
                sim.serve_next()
                waits.append(start - arr)
            assert sim.f.gc_count == 0
            w = np.array(waits[5000:])
            rows.append(dict(rho=rho, seed=seed, sim_wait_us=round(float(w.mean()), 3),
                             theory_wait_us=round(md1_wait_us(1.0 / gap, 100.0), 3),
                             sim_p99_us=round(float(np.percentile(w, 99)), 2)))
            print(rows[-1], flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)


if __name__ == "__main__":
    main()
