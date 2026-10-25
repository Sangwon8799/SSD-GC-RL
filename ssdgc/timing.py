"""Time model for the "when to collect" problem.

One server (the SSD) serves single-page host writes in arrival order.  Service times are a
relative model, chosen only to keep the well-known proportion "one GC is tens of writes long":

    host page write     t_w     = 100 us
    GC copy of a page   t_copy  = 150 us   (read + program)
    block erase         t_erase = 2000 us

Arrivals come in ON/OFF bursts: during ON, a Poisson stream with mean gap `gap_us`; ON lasts
`burst_writes` writes on average (geometric), OFF lasts an exponential time with mean `off_ms`.

Foreground GC (inside the FTL, forced when the free pool drops below the threshold) adds its
time to the write that triggered it.  Background GC is a non-preemptive job the controller can
start while the device is idle; a request that arrives meanwhile waits.

`TimedSim` is shared by the rule policies and the RL environment (`TimingEnv`).
"""
from __future__ import annotations

import math

import numpy as np

from .ftl import FTL
from .policies import make_policy
from .workloads import make_workload

T_W, T_COPY, T_ERASE = 100.0, 150.0, 2000.0


class Arrivals:
    def __init__(self, seed: int, gap_us=250.0, burst_writes=800, off_ms=150.0):
        self.rng = np.random.default_rng(seed)
        self.gap, self.burst, self.off = gap_us, burst_writes, off_ms * 1000.0
        self.t = 0.0
        self.left = 0

    def next(self) -> float:
        if self.left <= 0:
            self.t += self.rng.exponential(self.off)
            self.left = int(self.rng.geometric(1.0 / self.burst))
        else:
            self.t += self.rng.exponential(self.gap)
        self.left -= 1
        return self.t


class TimedSim:
    def __init__(self, workload="hc80-20", n_blocks=256, ppb=64, op=0.14, seed=0, victim="greedy",
                 gap_us=250.0, burst_writes=800, off_ms=150.0, gc_threshold=2, precondition=True,
                 t_w=T_W, t_copy=T_COPY, t_erase=T_ERASE):
        self.f = FTL(n_blocks, ppb, op, gc_threshold=gc_threshold)
        self.f.victim_fn = make_policy(victim, np.random.default_rng(seed + 11)) if isinstance(victim, str) else victim
        self.wl = make_workload(workload, self.f.U, seed)
        self.arr = Arrivals(seed + 101, gap_us, burst_writes, off_ms)
        self.t_w, self.t_copy, self.t_erase = t_w, t_copy, t_erase
        if precondition:
            for lpn in range(self.f.U):
                self.f.write(lpn)
            for _ in range(2 * self.f.U):
                self.f.write(self.wl.next())
        self.now = 0.0
        self.server_free = 0.0
        self.next_arr = self.arr.next()
        self.lat: list[float] = []          # latencies (us) of completed requests
        self.fg_hit: list[bool] = []
        self.bg_gcs = 0
        self.bg_copies = 0
        self.last_arr = 0.0
        self.gap_ema = gap_us

    # --------------------------------------------------------------- events
    def serve_next(self):
        """Serve the next arriving request (it may wait for the server)."""
        a = self.next_arr
        start = max(a, self.server_free)
        c0, g0 = self.f.gc_copies, self.f.gc_count
        self.f.write(self.wl.next())
        dc, dg = self.f.gc_copies - c0, self.f.gc_count - g0
        svc = self.t_w + dc * self.t_copy + dg * self.t_erase
        fin = start + svc
        self.server_free = fin
        self.lat.append(fin - a)
        self.fg_hit.append(dg > 0)
        self.gap_ema = 0.95 * self.gap_ema + 0.05 * (a - self.last_arr)
        self.last_arr = a
        self.next_arr = self.arr.next()
        self.now = fin
        return fin - a

    def background_gc(self, at: float, victim=None) -> int:
        """Start one background GC at time `at` (server must be idle).  Returns pages copied."""
        assert at >= self.server_free - 1e-9
        c0 = self.f.gc_copies
        self.f.collect_one(victim)
        dc = self.f.gc_copies - c0
        self.server_free = at + dc * self.t_copy + self.t_erase
        self.bg_gcs += 1
        self.bg_copies += dc
        return dc

    def idle_until(self) -> float:
        return self.next_arr

    # ------------------------------------------------------------- metrics
    def summary(self, since: int = 0) -> dict:
        lat = np.array(self.lat[since:]) / 1000.0
        fg = np.array(self.fg_hit[since:])
        return dict(n=len(lat), lat_mean_ms=float(lat.mean()), lat_p50_ms=float(np.percentile(lat, 50)),
                    lat_p99_ms=float(np.percentile(lat, 99)), lat_p999_ms=float(np.percentile(lat, 99.9)),
                    lat_max_ms=float(lat.max()), fg_frac=float(fg.mean()))


class RuleTiming:
    """Background GC whenever the device has been idle for >= tau_ms and the free pool < watermark."""

    def __init__(self, watermark: int, tau_ms: float = 0.0):
        self.W, self.tau = watermark, tau_ms * 1000.0
        self.name = f"idle(W={watermark},tau={tau_ms:g}ms)" if watermark > 0 else "foreground-only"


def run_rule(sim: TimedSim, rule: RuleTiming, n_requests: int) -> dict:
    """Drive a TimedSim with a rule for n_requests requests and return latency / WAF metrics.

    Before each request the device is idle on [server_free, next_arrival).  If the free pool is
    below the watermark, background GC starts `tau` after the idle start and continues back to back
    until the watermark is reached or the next request has arrived (GC is not pre-empted).
    """
    f = sim.f
    lat0 = len(sim.lat)
    h0, c0, b0 = f.host_writes, f.gc_copies, sim.bg_gcs
    for _ in range(n_requests):
        if rule.W > 0:
            t = sim.server_free + rule.tau
            while len(f.free) < rule.W and t < sim.next_arr:
                sim.background_gc(t)
                t = sim.server_free
        sim.serve_next()
    out = sim.summary(lat0)
    out["waf"] = 1.0 + (f.gc_copies - c0) / max(f.host_writes - h0, 1)
    out["bg_gcs"] = sim.bg_gcs - b0
    return out


def md1_wait_us(lam_per_us: float, svc_us: float) -> float:
    """Mean waiting time in queue of an M/D/1 queue (Pollaczek-Khinchine)."""
    rho = lam_per_us * svc_us
    return rho * svc_us / (2.0 * (1.0 - rho))


def p_of(x, q):
    return float(np.percentile(np.asarray(x), q))


_ = math  # keep import (used by callers via timing.math)
