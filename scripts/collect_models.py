"""Copy the selected checkpoints of the reported runs into models/ (small; committed with the code).

For every run of the final design (r2-*, r3-*, r4-*, r7-*), the timing agents (e6-*) and the first design
(d2-*, d3-*): config.json, done.json, eval_log.csv and the selected model (best.pt, or model.pkl for the
tabular timing agent).  Also writes results/summary/runs.csv (one row per run).
    python scripts/collect_models.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import statistics as st

PREFIXES = ("r2-", "r3-", "r4-", "r7-", "e6-", "d2-", "d3-")
KEEP = ("config.json", "done.json", "eval_log.csv", "best.pt", "model.pkl")


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    rows = []
    n = 0
    for d in sorted(glob.glob("runs/*/")):
        rid = os.path.basename(d.rstrip("/"))
        if not rid.startswith(PREFIXES) or not os.path.exists(os.path.join(d, "done.json")):
            continue
        out = os.path.join("models", rid)
        os.makedirs(out, exist_ok=True)
        for f in KEEP:
            if os.path.exists(os.path.join(d, f)):
                shutil.copy2(os.path.join(d, f), os.path.join(out, f))
        if not any(os.path.exists(os.path.join(d, f)) for f in ("best.pt", "model.pkl")) and os.path.exists(os.path.join(d, "model.pt")):
            shutil.copy2(os.path.join(d, "model.pt"), os.path.join(out, "model.pt"))   # timing DQN keeps its last model
        n += 1
        dn = json.load(open(os.path.join(d, "done.json")))
        if "best_waf" in dn and dn["best_waf"]:
            for wl, v in dn["best_waf"].items():
                rows.append([rid, wl, round(st.mean(v), 5), round(st.mean(dn["final_waf"][wl]), 5), dn.get("train_wall_s")])
        elif "eval" in dn:
            ev = dn["eval"]
            rows.append([rid, "hc80-20(OP 28%)", round(st.mean(e["waf"] for e in ev), 5), "", dn.get("train_wall_s")])
    with open("results/summary/runs.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["run", "workload", "test_waf_selected", "test_waf_last", "train_wall_s"])
        w.writerows(rows)
    print("models:", n, "runs; summary rows:", len(rows))


if __name__ == "__main__":
    main()
