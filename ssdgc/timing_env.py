"""RL environment for GC *timing* ("when to collect").

Decision epochs happen only while the device is idle (no request in service, none waiting):
at the moment it becomes idle, at fixed points of the idle time (0.5, 1, 2, 4 ... 256 ms after
the idle start) and right after a background GC finishes.  The agent never sees the next arrival.

Action 0 = wait, 1 = start one background GC now (victim chosen by the victim policy, greedy by
default).  Foreground GC is still forced by the FTL when the free pool drops below 2 blocks.

Reward between two epochs = -[ (sum of latencies of the requests completed in between, in ms)
                               + lambda * (pages copied by any GC in between) / pages_per_block ] * 0.01
(lambda = how many ms of summed latency one block's worth of copying is worth; the factor 0.01
only keeps the numbers small for the neural network.)
Epochs are unevenly spaced in time, so `info["dt_ms"]` is returned and the agents discount by
gamma ** dt_ms (semi-Markov discounting: one millisecond is one discount step).

Observation (5 numbers): free blocks, idle time so far, recent mean inter-arrival time,
valid ratio of the victim the next GC would take, and whether the last request found the pool
empty (foreground GC) recently.
"""
from __future__ import annotations

import math

import numpy as np

from .ftl import CLOSED
from .policies import eligible
from .timing import TimedSim

TICKS_MS = (0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256)
N_TIMING_FEAT = 5
TIMING_NAMES = ["free", "idle", "gap", "victim_u", "fg_recent"]


def timing_obs(sim: TimedSim, t: float, idle_start: float, fg_ema: float) -> np.ndarray:
    f = sim.f
    idle_ms = max(t - idle_start, 0.0) / 1000.0
    cand = eligible(f)
    u = float(f.valid[cand].min()) / f.P if len(cand) else 1.0
    return np.array([
        min(len(f.free), 32) / 16.0,
        math.log1p(idle_ms) / math.log1p(256.0),
        math.log1p(sim.gap_ema / 1000.0) / math.log1p(10.0),
        u,
        fg_ema,
    ], dtype=np.float32)


class TimingEnv:
    def __init__(self, workload="hc80-20", op=0.28, seed=0, lam=1.0, episode_len=6000,
                 gap_us=400.0, burst_writes=256, off_ms=300.0, victim="greedy", reward_scale=0.01):
        self.kw = dict(workload=workload, op=op, gap_us=gap_us, burst_writes=burst_writes, off_ms=off_ms,
                       victim=victim)
        self.lam = lam
        self.reward_scale = reward_scale
        self.episode_len = episode_len
        self.rng = np.random.default_rng(seed)

    def _serve_until_idle(self):
        """Serve requests until the device is idle before the next arrival; returns (lat_sum_ms, n)."""
        s = self.sim
        lat = 0.0
        n = 0
        while True:
            l_us = s.serve_next()
            lat += l_us / 1000.0
            n += 1
            self.fg_ema = 0.98 * self.fg_ema + 0.02 * float(s.fg_hit[-1])
            if s.server_free < s.next_arr:
                break
        self.idle_start = s.server_free
        self.t = s.server_free
        return lat, n

    def _obs(self):
        return timing_obs(self.sim, self.t, self.idle_start, self.fg_ema)

    def reset(self, seed=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        s = int(self.rng.integers(1 << 31))
        self.sim = TimedSim(seed=s, **self.kw)
        self.fg_ema = 0.0
        self.k = 0
        self.copies0 = self.sim.f.gc_copies
        self._serve_until_idle()
        self.t_prev = self.t
        self.ep_lat = []
        return self._obs(), {}

    def step(self, a: int):
        s = self.sim
        f = s.f
        c0 = f.gc_copies
        lat_sum = 0.0
        t_prev = self.t
        if a == 1 and len(eligible(f)) > 0:
            s.background_gc(self.t)
            if s.server_free < s.next_arr:
                self.t = s.server_free                      # still idle: decide again right away
            else:
                lat_sum, _ = self._serve_until_idle()
        else:
            idle_ms = (self.t - self.idle_start) / 1000.0
            nxt = next((x for x in TICKS_MS if x > idle_ms + 1e-9), None)
            t_next = self.idle_start + nxt * 1000.0 if nxt is not None else None
            if t_next is not None and t_next < s.next_arr:
                self.t = t_next
            else:
                lat_sum, _ = self._serve_until_idle()
        copies = f.gc_copies - c0
        r = -(lat_sum + self.lam * copies / f.P) * self.reward_scale
        self.k += 1
        dt_ms = (self.t - t_prev) / 1000.0
        trunc = self.k >= self.episode_len
        return self._obs(), r, False, trunc, {"dt_ms": dt_ms, "lat_sum": lat_sum, "copies": copies}


def run_timing_policy(policy, n_requests=60000, seed=100, lam=1.0, **kw):
    """Evaluate a frozen timing policy (callable obs -> action) for n_requests requests."""
    env = TimingEnv(seed=seed, lam=lam, episode_len=10 ** 12, **kw)
    obs, _ = env.reset(seed=seed)
    f = env.sim.f
    lat0 = len(env.sim.lat)
    h0, c0 = f.host_writes, f.gc_copies
    while len(env.sim.lat) - lat0 < n_requests:
        obs, r, _, _, _ = env.step(int(policy(obs)))
    out = env.sim.summary(lat0)
    out["waf"] = 1.0 + (f.gc_copies - c0) / max(f.host_writes - h0, 1)
    out["bg_gcs"] = env.sim.bg_gcs
    return out


class RulePolicy:
    """Rule expressed on the same observation: GC if free < W_emerg, or if free < W and idle >= tau."""

    def __init__(self, W: int, tau_ms: float = 0.0, W_emerg: int = 0):
        self.W, self.tau, self.We = W, tau_ms, W_emerg
        self.name = f"rule(W={W},tau={tau_ms:g}ms,We={W_emerg})"

    def __call__(self, obs) -> int:
        free = round(obs[0] * 16.0)
        idle_ms = math.expm1(obs[1] * math.log1p(256.0))
        if free < self.We:
            return 1
        if free < self.W and idle_ms >= self.tau - 1e-6:
            return 1
        return 0


_ = CLOSED
