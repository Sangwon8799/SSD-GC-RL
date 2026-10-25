"""Run a list of training jobs, at most --parallel at a time, politely.

    python scripts/run_queue.py --queue queues/main.json [--parallel 2] [--max-load 8]

* each job is `python scripts/<script> --exp-id ID <args...>` in its own process (script defaults to
  train.py; GC-timing jobs use train_timing.py)
  (nice 19, OMP_NUM_THREADS=1, one log file per job under runs/<ID>/stdout.log)
* a job whose runs/<ID>/done.json exists is skipped (idempotent); a half-finished job resumes
  from its checkpoint
* a new job is started only while the 1-minute load average is below --max-load and no site
  build (`astro build`) is running on the machine
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")


def site_build_running() -> bool:
    try:
        out = subprocess.run(["ps", "-eo", "args"], capture_output=True, text=True).stdout
    except Exception:
        return False
    return any(("astro" in ln and " build" in ln and "ps -eo" not in ln) for ln in out.splitlines())


def live_ids() -> set[str]:
    """exp-ids of training processes already running (so a restarted runner never starts a twin)."""
    out = set()
    for p in os.listdir("/proc"):
        if not p.isdigit():
            continue
        try:
            c = open(f"/proc/{p}/cmdline", "rb").read().split(b"\0")
        except OSError:
            continue
        args = [x.decode(errors="replace") for x in c]
        if any(x.endswith(("train.py", "train_timing.py")) for x in args) and "--exp-id" in args:
            i = args.index("--exp-id")
            if i + 1 < len(args):
                out.add(args[i + 1])
    return out


def load_jobs(path, runs, seen):
    """(Re-)read the queue file: jobs not seen before and not finished are appended in file order."""
    new = []
    for j in json.load(open(path)):
        if j["id"] in seen:
            continue
        seen.add(j["id"])
        if not os.path.exists(os.path.join(ROOT, runs, j["id"], "done.json")):
            new.append(j)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=8.0)
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--ignore-build", action="store_true",
                    help="start jobs even while a site build runs (only when the orchestrator allows it)")
    ap.add_argument("--parallel-file", default="", help="if this file exists, its integer overrides --parallel "
                                                         "(re-read every 10 s, so the width can change while running)")
    a = ap.parse_args()
    seen: set[str] = set()
    pending = load_jobs(a.queue, a.runs, seen)
    print(f"[queue] {len(seen)} jobs, {len(pending)} pending", flush=True)
    running: list[tuple[dict, subprocess.Popen]] = []
    adopted: set[str] = set()
    while True:
        added = load_jobs(a.queue, a.runs, seen)
        if added:
            pending += added
            print(f"[queue] {time.strftime('%H:%M:%S')} +{len(added)} jobs from {a.queue}", flush=True)
        alive = live_ids()
        adopted &= alive
        if not (pending or running or adopted):
            break
        for j, p in list(running):
            if p.poll() is not None:
                running.remove((j, p))
                ok = os.path.exists(os.path.join(ROOT, a.runs, j["id"], "done.json"))
                print(f"[queue] {time.strftime('%H:%M:%S')} finished {j['id']} rc={p.returncode} done={ok}", flush=True)
                if not ok and j.get("_tries", 0) < 2:
                    j["_tries"] = j.get("_tries", 0) + 1
                    pending.append(j)
        width = a.parallel
        if a.parallel_file and os.path.exists(a.parallel_file):
            try:
                width = max(0, min(2, int(open(a.parallel_file).read().strip())))
            except ValueError:
                pass
        while pending and len(running) < width:
            load = os.getloadavg()[0]
            if load >= a.max_load or (not a.ignore_build and site_build_running()):
                break
            if len(running) + len(adopted) >= width:
                break
            j = pending.pop(0)
            if j["id"] in alive:          # started by an earlier runner and still going: wait for it
                pending.append(j)
                if j["id"] in adopted:
                    break
                adopted.add(j["id"])
                continue
            if os.path.exists(os.path.join(ROOT, a.runs, j["id"], "done.json")):
                continue
            d = os.path.join(ROOT, a.runs, j["id"])
            os.makedirs(d, exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            cmd = ["nice", "-n", "19", sys.executable, os.path.join(HERE, j.get("script", "train.py")), "--exp-id",
                   j["id"], "--runs", os.path.join(ROOT, a.runs)] + j["args"]
            log = open(os.path.join(d, "stdout.log"), "a")
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running.append((j, p))
            print(f"[queue] {time.strftime('%H:%M:%S')} start {j['id']} pid={p.pid} load={load:.1f}", flush=True)
            time.sleep(2)
        time.sleep(10)
    print(f"[queue] {time.strftime('%H:%M:%S')} all done", flush=True)


if __name__ == "__main__":
    main()
