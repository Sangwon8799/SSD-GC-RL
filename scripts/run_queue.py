"""Run a list of training jobs, at most --parallel at a time, politely.

    python scripts/run_queue.py --queue queues/main.json [--parallel 2] [--max-load 8]

* each job is `python scripts/train.py --exp-id ID <args...>` in its own process
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue", required=True)
    ap.add_argument("--parallel", type=int, default=2)
    ap.add_argument("--max-load", type=float, default=8.0)
    ap.add_argument("--runs", default="runs")
    a = ap.parse_args()
    jobs = json.load(open(a.queue))
    pending = [j for j in jobs if not os.path.exists(os.path.join(ROOT, a.runs, j["id"], "done.json"))]
    print(f"[queue] {len(jobs)} jobs, {len(pending)} pending", flush=True)
    running: list[tuple[dict, subprocess.Popen]] = []
    while pending or running:
        for j, p in list(running):
            if p.poll() is not None:
                running.remove((j, p))
                ok = os.path.exists(os.path.join(ROOT, a.runs, j["id"], "done.json"))
                print(f"[queue] {time.strftime('%H:%M:%S')} finished {j['id']} rc={p.returncode} done={ok}", flush=True)
                if not ok and j.get("_tries", 0) < 2:
                    j["_tries"] = j.get("_tries", 0) + 1
                    pending.append(j)
        while pending and len(running) < a.parallel:
            load = os.getloadavg()[0]
            if load >= a.max_load or site_build_running():
                break
            j = pending.pop(0)
            d = os.path.join(ROOT, a.runs, j["id"])
            os.makedirs(d, exist_ok=True)
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            cmd = ["nice", "-n", "19", sys.executable, os.path.join(HERE, "train.py"), "--exp-id", j["id"],
                   "--runs", os.path.join(ROOT, a.runs)] + j["args"]
            log = open(os.path.join(d, "stdout.log"), "a")
            p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            running.append((j, p))
            print(f"[queue] {time.strftime('%H:%M:%S')} start {j['id']} pid={p.pid} load={load:.1f}", flush=True)
            time.sleep(2)
        time.sleep(10)
    print(f"[queue] {time.strftime('%H:%M:%S')} all done", flush=True)


if __name__ == "__main__":
    main()
