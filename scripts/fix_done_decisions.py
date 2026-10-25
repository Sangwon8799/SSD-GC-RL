"""One-off repair (2026-10-26): done.json files written before the fix recorded the step count of the *best*
checkpoint as `decisions` (load_state_dict restores the counter).  Take the trained count from the last row of
train_log.csv, keep the old value as `best_decisions`.  Idempotent."""
import csv
import glob
import json
import os

n = 0
for p in sorted(glob.glob("runs/*/done.json")) + sorted(glob.glob("pilot/**/done.json", recursive=True)):
    d = json.load(open(p))
    if "best_decisions" in d:
        continue
    tl = os.path.join(os.path.dirname(p), "train_log.csv")
    if not os.path.exists(tl):
        continue
    rows = list(csv.DictReader(open(tl)))
    if not rows or "decisions" not in rows[-1]:
        continue
    d["best_decisions"] = d.get("decisions")
    d["decisions"] = int(rows[-1]["decisions"])
    json.dump(d, open(p, "w"), indent=2)
    n += 1
print("repaired", n)
