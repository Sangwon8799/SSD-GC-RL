"""Victim selection by choosing a rule — temporally extended actions (the second formulation).

Choosing a block directly at every GC did not learn well: one decision changes the future too little, so its
signal is lost in the noise (see the page, section 7).  Here the agent picks a *rule* and the FTL then uses it
for the next M garbage collections:

    action a  -> beta = BETAS[a];   victim = argmax over closed blocks of  (1 - u) * age**beta / u
                 beta 0   = greedy (fewest valid pages)       beta 1 = cost-benefit (LFS 1992)
                 beta 0.5 / 2 = in between / more weight on age
    reward    = -(pages copied in those M GCs) / (M * pages_per_block) - shift      (= -mean victim u + const)
    discount  = gamma per macro step (one macro step = M GCs)

The device is never refilled between episodes (continuing task); an episode is only an update segment.
Observation (11 numbers) — statistics of the closed blocks and of the last macro step, plus the previous action.
"""
from __future__ import annotations

import numpy as np

from .ftl import CLOSED, FTL, GCW
from .policies import eligible, make_policy
from .workloads import make_workload

BETAS = (0.0, 0.5, 1.0, 2.0)
RULE_FEAT_NAMES = ["mean_u", "std_u", "full_frac", "gc_block_frac", "victim_u_last", "waf_last",
                   "young_inval_share", "prev_a0", "prev_a1", "prev_a2", "prev_a3"]
N_RULE_FEAT = len(RULE_FEAT_NAMES)


def beta_victim(ftl: FTL, beta: float) -> int:
    cand = eligible(ftl)
    u = ftl.valid[cand] / ftl.P
    if beta == 0.0:
        return int(cand[np.lexsort((ftl.close_time[cand], u))][0])     # greedy, older first on ties
    age = ftl.now - ftl.close_time[cand] + 1.0
    with np.errstate(divide="ignore"):
        s = (1.0 - u) * age ** beta / u
    return int(cand[np.argmax(s)])


class RuleObs:
    """Builds the observation; shared by the training env and the evaluation wrapper."""

    def __init__(self, drop=()):
        self.prev_a = 2                       # cost-benefit (index into BETAS)
        self.victim_u_last = 0.75
        self.waf_last = 3.0
        self.drop = [RULE_FEAT_NAMES.index(d) for d in drop]

    def __call__(self, f: FTL) -> np.ndarray:
        closed = f.state == CLOSED
        u = f.valid[closed] / f.P
        n = max(int(closed.sum()), 1)
        age = f.now - f.close_time[closed]
        inv = f.inval_since_close[closed].astype(np.float64)
        young = age <= np.median(age) if len(age) else np.zeros(0, bool)
        share = float(inv[young].sum() / inv.sum()) if inv.sum() > 0 else 0.5
        x = np.zeros(N_RULE_FEAT, dtype=np.float32)
        x[0] = u.mean() if len(u) else 0.0
        x[1] = u.std() * 4.0 if len(u) else 0.0
        x[2] = float((u >= 0.95).mean()) if len(u) else 0.0
        x[3] = float((f.origin[closed] == GCW).sum()) / n
        x[4] = self.victim_u_last
        x[5] = min(self.waf_last, 20.0) / 10.0
        x[6] = share
        x[7 + self.prev_a] = 1.0
        if self.drop:
            x[self.drop] = 0.0
        return x


def run_macro(f: FTL, wl, beta: float, M: int):
    """Write host pages and do the next M garbage collections with rule beta. Returns (copies, host_writes)."""
    copies = host = done = 0
    while done < M:
        while not f.need_gc():
            lpn = wl.next()
            f._invalidate(lpn)
            f._program_host(lpn)
            f.host_writes += 1
            f.now += 1.0
            host += 1
        copies += f.collect_one(beta_victim(f, beta))
        done += 1
    return copies, host


class RuleEnv:
    def __init__(self, workload="hc80-20", n_blocks=256, ppb=64, op=0.14, M=64, episode_len=50, seed=0,
                 workload_pool=None, warmup_writes=2.0, acts=(0, 1, 2, 3), drop=()):
        self.acts, self.drop = list(acts), list(drop)
        self.workload_spec = workload
        self.workload_pool = list(workload_pool) if workload_pool else None
        self.M, self.episode_len = M, episode_len
        self.rng = np.random.default_rng(seed)
        self.ftl = FTL(n_blocks, ppb, op)
        self.U, self.P = self.ftl.U, self.ftl.P
        self.warmup_writes = warmup_writes
        self.obs = RuleObs(self.drop)
        self._started = False

    def _pick_workload(self):
        if self.workload_pool:
            return self.workload_pool[int(self.rng.integers(len(self.workload_pool)))]
        return self.workload_spec

    def reset(self, seed=None):
        if self._started:
            spec = self._pick_workload()
            if spec != self.cur_workload:
                self.cur_workload = spec
                self.wl = make_workload(spec, self.U, int(self.rng.integers(1 << 31)))
        else:
            self._started = True
            if seed is not None:
                self.rng = np.random.default_rng(seed)
            spec = self._pick_workload()
            self.cur_workload = spec
            f = self.ftl
            f.reset()
            f.victim_fn = make_policy("greedy", np.random.default_rng(int(self.rng.integers(1 << 31))))
            self.wl = make_workload(spec, self.U, int(self.rng.integers(1 << 31)))
            for lpn in range(self.U):
                f.write(lpn)
            for _ in range(int(self.warmup_writes * self.U)):
                f.write(self.wl.next())
            f.victim_fn = None
            self.obs = RuleObs(self.drop)
        self.t = 0
        self.ep_copies = self.ep_host = 0
        return self.obs(self.ftl), {"workload": self.cur_workload}

    def step(self, a: int):
        b = self.acts[int(a)]
        copies, host = run_macro(self.ftl, self.wl, BETAS[b], self.M)
        self.ep_copies += copies
        self.ep_host += host
        mean_u = copies / (self.M * self.P)
        self.obs.prev_a = b
        self.obs.victim_u_last = mean_u
        self.obs.waf_last = 1.0 + copies / max(host, 1)
        self.t += 1
        trunc = self.t >= self.episode_len
        return self.obs(self.ftl), -mean_u, False, trunc, {"copies": copies, "host": host}

    def episode_waf(self):
        return 1.0 + self.ep_copies / max(self.ep_host, 1)


class RuleAgentPolicy:
    """A trained rule-choosing agent as an FTL victim policy: every M GCs it looks at the device and picks beta."""

    def __init__(self, agent, M=64, stochastic=False, acts=(0, 1, 2, 3), drop=()):
        self.agent, self.M, self.stochastic = agent, M, stochastic
        self.acts = list(acts)
        self.obs = RuleObs(drop)
        self.k = 0
        self.beta = BETAS[2]
        self.copies0 = self.host0 = 0
        self.us = []
        self.choices = []

    def __call__(self, f: FTL) -> int:
        if self.k % self.M == 0:
            if self.k > 0:
                dc, dh = f.gc_copies - self.copies0, f.host_writes - self.host0
                self.obs.victim_u_last = dc / (self.M * f.P)
                self.obs.waf_last = 1.0 + dc / max(dh, 1)
            b = self.acts[int(self.agent.act(self.obs(f), greedy=not self.stochastic))]
            self.obs.prev_a = b
            self.beta = BETAS[b]
            self.choices.append(b)
            self.copies0, self.host0 = f.gc_copies, f.host_writes
        self.k += 1
        return beta_victim(f, self.beta)


class FixedBeta:
    def __init__(self, beta):
        self.beta = beta
        self.name = f"beta{beta:g}"

    def __call__(self, f):
        return beta_victim(f, self.beta)


class RandomBeta:
    """Baseline: a uniformly random rule every M GCs (what the rule set alone gives)."""

    def __init__(self, M=64, seed=0):
        self.M, self.k = M, 0
        self.rng = np.random.default_rng(seed)
        self.beta = BETAS[2]

    def __call__(self, f):
        if self.k % self.M == 0:
            self.beta = BETAS[int(self.rng.integers(len(BETAS)))]
        self.k += 1
        return beta_victim(f, self.beta)
