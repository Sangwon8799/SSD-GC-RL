"""How long does a victim-policy change take to pay off?  (diagnostic for the discount factor)

Run greedy to steady state, then switch to cost-benefit (and the reverse) and record the valid ratio u of
every victim after the switch.  The number of GC decisions until the running mean of u reaches its new
steady level is the horizon the agent must see (1 / (1 - gamma) should be at least of that order).

    python scripts/switch_horizon.py [workload] -> results/analysis/switch_<workload>.csv
"""
from __future__ import annotations

import csv
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np  # noqa: E402

from ssdgc.ftl import FTL  # noqa: E402
from ssdgc.policies import make_policy  # noqa: E402
from ssdgc.workloads import make_workload  # noqa: E402


def trace(wl_name, first, second, seed=100, warm=4.0, after_decisions=4000, op=0.14):
    f = FTL(256, 64, op)
    rng = np.random.default_rng(seed)
    us = []
    f.victim_fn = make_policy(first, rng)
    wl = make_workload(wl_name, f.U, seed)
    for lpn in range(f.U):
        f.write(lpn)
    for _ in range(int(warm * f.U)):
        f.write(wl.next())
    pol2 = make_policy(second, rng)

    def rec(ftl):
        b = pol2(ftl)
        us.append(ftl.valid[b] / ftl.P)
        return b
    f.victim_fn = rec
    while len(us) < after_decisions:
        f.write(wl.next())
    return np.array(us[:after_decisions])


def main():
    wl = sys.argv[1] if len(sys.argv) > 1 else "hc80-20"
    os.makedirs("results/analysis", exist_ok=True)
    out = []
    for a, b in (("greedy", "cb"), ("cb", "greedy")):
        u = np.mean([trace(wl, a, b, seed=s) for s in (100, 101, 102)], axis=0)
        w = 50
        ma = np.convolve(u, np.ones(w) / w, mode="valid")
        final = ma[-1000:].mean()
        start = ma[:50].mean()
        # decisions until 90 % of the way from start to final
        target = start + 0.9 * (final - start)
        hit = np.argmax((ma - target) * np.sign(final - start) >= 0) + w
        print(f"{wl} {a}->{b}: u {start:.3f} -> {final:.3f}, 90% reached after ~{hit} decisions")
        for i in range(0, len(ma), 25):
            out.append([f"{a}->{b}", i + w, round(float(ma[i]), 5)])
    with open(f"results/analysis/switch_{wl}.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["switch", "decision", "u_ma50"])
        wr.writerows(out)


if __name__ == "__main__":
    main()
