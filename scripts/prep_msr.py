"""Turn an MSR Cambridge block trace (CSV) into the LPN array that ssdgc.workloads.Replay plays.

    python scripts/prep_msr.py <trace.csv> <out.npy> [--page 4096] [--max-writes N]

MSR Cambridge rows: Timestamp,Hostname,DiskNumber,Type,Offset,Size,ResponseTime (offset / size in
bytes).  Only Write rows are kept; a write of `size` bytes at `offset` becomes the 4 KiB pages it
touches, in order.  Page numbers are then renumbered by first appearance (0, 1, 2 ...) so the
trace's footprint is dense; Replay folds them onto the device with `lpn % U`.
The traces themselves are NOT included in this repository (the download needs a registration);
the README says where they come from.  This project's experiments did not use them.
"""
from __future__ import annotations

import argparse
import csv

import numpy as np


def convert(rows, page=4096, max_writes=None):
    remap: dict[int, int] = {}
    out = []
    for r in rows:
        if len(r) < 6 or r[3].strip().lower() != "write":
            continue
        off, size = int(r[4]), int(r[5])
        if size <= 0:
            continue
        for p in range(off // page, (off + size - 1) // page + 1):
            out.append(remap.setdefault(p, len(remap)))
            if max_writes and len(out) >= max_writes:
                return np.array(out, dtype=np.int64), len(remap)
    return np.array(out, dtype=np.int64), len(remap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("out")
    ap.add_argument("--page", type=int, default=4096)
    ap.add_argument("--max-writes", type=int, default=0)
    a = ap.parse_args()
    with open(a.trace, newline="") as fh:
        lpns, footprint = convert(csv.reader(fh), a.page, a.max_writes or None)
    np.save(a.out, lpns)
    print(f"{len(lpns)} page writes, footprint {footprint} pages -> {a.out}")


if __name__ == "__main__":
    main()
