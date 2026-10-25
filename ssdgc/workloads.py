"""Write workloads: streams of logical page numbers (LPNs).

All generators draw numbers in chunks (vectorised) and hand them out one by one.
The hot set of a skewed workload is a fixed random subset of the LPN space, so
"hot" pages are not physically adjacent (page-level mapping does not care anyway).
"""
from __future__ import annotations

import numpy as np


class Workload:
    name = "base"

    def __init__(self, U: int, seed: int):
        self.U = int(U)
        self.rng = np.random.default_rng(seed)
        self._buf = np.empty(0, dtype=np.int64)
        self._i = 0
        self.t = 0                      # writes handed out so far

    def _draw(self, n: int) -> np.ndarray:  # pragma: no cover - abstract
        raise NotImplementedError

    def next(self) -> int:
        if self._i >= len(self._buf):
            self._buf = self._draw(65536)
            self._i = 0
        x = int(self._buf[self._i])
        self._i += 1
        self.t += 1
        return x

    def take(self, n: int) -> np.ndarray:
        out = np.empty(n, dtype=np.int64)
        for k in range(n):
            out[k] = self.next()
        return out


class Uniform(Workload):
    name = "uniform"

    def _draw(self, n):
        return self.rng.integers(0, self.U, size=n)


class HotCold(Workload):
    """A fraction ``hot_frac`` of the LPNs receives ``hot_prob`` of the writes (e.g. 20 % / 80 %)."""

    def __init__(self, U, seed, hot_frac=0.2, hot_prob=0.8):
        super().__init__(U, seed)
        self.hot_frac, self.hot_prob = float(hot_frac), float(hot_prob)
        perm = np.random.default_rng(seed + 7919).permutation(self.U)
        nh = max(1, int(round(self.hot_frac * self.U)))
        self.hot, self.cold = perm[:nh], perm[nh:]
        self.name = f"hotcold-{int(round(hot_prob * 100))}/{int(round(hot_frac * 100))}"

    def _draw(self, n):
        is_hot = self.rng.random(n) < self.hot_prob
        h = self.hot[self.rng.integers(0, len(self.hot), size=n)]
        c = self.cold[self.rng.integers(0, len(self.cold), size=n)]
        return np.where(is_hot, h, c)


class Zipf(Workload):
    """Bounded Zipf over the LPN space: P(rank r) ~ 1 / r^s."""

    def __init__(self, U, seed, s=1.0):
        super().__init__(U, seed)
        self.s = float(s)
        w = 1.0 / np.arange(1, self.U + 1, dtype=np.float64) ** self.s
        self.cdf = np.cumsum(w) / w.sum()
        self.perm = np.random.default_rng(seed + 104729).permutation(self.U)
        self.name = f"zipf-{s:g}"

    def _draw(self, n):
        r = np.searchsorted(self.cdf, self.rng.random(n), side="right")
        r = np.minimum(r, self.U - 1)
        return self.perm[r]


class ShiftingHotCold(HotCold):
    """Hot/cold whose hot set moves to a different region every ``period`` writes."""

    def __init__(self, U, seed, hot_frac=0.1, hot_prob=0.9, period=None):
        super().__init__(U, seed, hot_frac, hot_prob)
        self.period = int(period or 2 * U)
        self._perm = np.random.default_rng(seed + 7919).permutation(self.U)
        self._phase = 0
        self.name = f"shifting-{int(round(hot_prob * 100))}/{int(round(hot_frac * 100))}"

    def _set_phase(self, k):
        nh = len(self.hot)
        rolled = np.roll(self._perm, -k * nh)
        self.hot, self.cold = rolled[:nh], rolled[nh:]

    def _draw(self, n):
        # draw in pieces so that a phase change happens exactly every `period` writes
        out = np.empty(n, dtype=np.int64)
        done = 0
        pos = self.t + (len(self._buf) - self._i)  # writes already drawn (buffer is empty when called)
        while done < n:
            phase = (pos + done) // self.period
            if phase != self._phase:
                self._phase = phase
                self._set_phase(phase)
            left_in_phase = self.period - (pos + done) % self.period
            m = min(n - done, left_in_phase)
            out[done:done + m] = HotCold._draw(self, m)
            done += m
        return out


class Replay(Workload):
    """Replays a pre-computed LPN array (e.g. a pre-processed block trace) in a loop."""

    def __init__(self, U, seed, lpns: np.ndarray, name="trace", offset=0):
        super().__init__(U, seed)
        self.lpns = np.asarray(lpns, dtype=np.int64) % self.U
        self.pos = int(offset) % len(self.lpns)
        self.name = name

    def _draw(self, n):
        idx = (self.pos + np.arange(n)) % len(self.lpns)
        self.pos = int((self.pos + n) % len(self.lpns))
        return self.lpns[idx]


def make_workload(spec: str, U: int, seed: int, trace_lpns=None):
    """``spec`` examples: uniform, hc80-20, hc90-10, zipf1.0, shift90-10."""
    if spec == "uniform":
        return Uniform(U, seed)
    if spec.startswith("hc"):
        p, f = spec[2:].split("-")
        return HotCold(U, seed, hot_frac=int(f) / 100, hot_prob=int(p) / 100)
    if spec.startswith("zipf"):
        return Zipf(U, seed, s=float(spec[4:]))
    if spec.startswith("shift"):
        p, f = spec[5:].split("-")
        return ShiftingHotCold(U, seed, hot_frac=int(f) / 100, hot_prob=int(p) / 100)
    if spec.startswith("trace") and trace_lpns is not None:
        return Replay(U, seed, trace_lpns, name=spec, offset=seed * 7919)
    raise KeyError(spec)
