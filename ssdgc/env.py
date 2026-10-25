"""RL environment for GC *victim selection* ("which block to erase").

Decision epoch = every moment the FTL needs a GC victim (the free-block pool fell below the
threshold).  Between two decisions the environment replays host writes from the workload.

Observation (dict of numpy arrays):
  cand  (k, F)  features of k candidate blocks
  mask  (k,)    1 = real candidate, 0 = padding
  glob  (G,)    device-wide features
Action:  index i in [0, k) of the candidate to collect.
Reward:  -(valid pages copied by this GC) / pages_per_block   (= -u of the victim)

In steady state WAF = 1 / (1 - mean u of victims), so maximising the long-run average
reward is the same as minimising write amplification.  A myopic agent (gamma = 0)
can at best rediscover the greedy policy; gamma > 0 is needed to trade a slightly more
expensive victim now for cheaper GCs later (the cost-benefit idea).

The interface mimics gymnasium: reset(seed) -> (obs, info), step(a) -> (obs, r, terminated,
truncated, info).  The task is continuing: `terminated` is always False and an episode is
cut (`truncated`) after `episode_len` decisions.
"""
from __future__ import annotations

import numpy as np

from .ftl import CLOSED, FTL, GCW
from .policies import Greedy, cb_score, eligible, make_policy
from .workloads import make_workload

N_CAND_FEAT = 8
N_GLOB_FEAT = 4
CAND_NAMES = ["u", "age", "since_inval", "inval_rate", "gc_origin", "wear", "greedy_rank", "cb_rank"]
GLOB_NAMES = ["free_ratio", "mean_u", "gc_block_frac", "recent_waf"]


def candidate_set(ftl: FTL, k: int, n_greedy: int):
    """k candidate blocks: the n_greedy fewest-valid blocks + the best cost-benefit blocks.

    Returns (blocks array, greedy_rank, cb_rank) with ranks normalised to [0, 1].
    """
    cand = eligible(ftl)
    if len(cand) == 0:
        raise RuntimeError("no eligible victim")
    v = ftl.valid[cand]
    order_g = cand[np.lexsort((ftl.close_time[cand], v))]      # fewest valid, older first on ties
    s = cb_score(ftl, cand)
    order_c = cand[np.argsort(-s, kind="stable")]
    chosen: list[int] = []
    for b in order_g[:n_greedy]:
        chosen.append(int(b))
    for b in order_c:
        if len(chosen) >= k:
            break
        if int(b) not in chosen:
            chosen.append(int(b))
    for b in order_g[n_greedy:]:
        if len(chosen) >= k:
            break
        if int(b) not in chosen:
            chosen.append(int(b))
    blocks = np.array(chosen, dtype=np.int64)
    rank_g = {int(b): i for i, b in enumerate(order_g)}
    rank_c = {int(b): i for i, b in enumerate(order_c)}
    gr = np.array([rank_g[int(b)] for b in blocks], dtype=np.float64)
    cr = np.array([rank_c[int(b)] for b in blocks], dtype=np.float64)
    return blocks, gr, cr


class ObsBuilder:
    """Builds the observation at a decision epoch.  Shared by the training env and evaluation
    so that the agent sees exactly the same features in both places.

    It also keeps the "recent WAF" feature: an EMA over decisions of
    1 + (pages copied by the previous GC) / (host writes since the previous decision).
    """

    def __init__(self, k: int = 8, n_greedy: int = 4, drop=()):
        self.k, self.n_greedy = k, n_greedy
        self.drop = [CAND_NAMES.index(d) for d in drop]
        self.reset(None)

    def reset(self, ftl):
        self.waf_ema = ftl.waf() if (ftl is not None and ftl.host_writes > 0) else 1.0
        self.last_host = ftl.host_writes if ftl is not None else 0
        self.last_copies = ftl.gc_copies if ftl is not None else 0

    def __call__(self, f: FTL):
        dh = f.host_writes - self.last_host
        dc = f.gc_copies - self.last_copies
        if dh > 0:
            self.waf_ema = 0.98 * self.waf_ema + 0.02 * (1.0 + dc / dh)
        self.last_host, self.last_copies = f.host_writes, f.gc_copies
        k = self.k
        blocks, gr, cr = candidate_set(f, k, self.n_greedy)
        n = len(blocks)
        U, P = float(f.U), float(f.P)
        cand = np.zeros((k, N_CAND_FEAT), dtype=np.float32)
        mask = np.zeros(k, dtype=np.float32)
        u = f.valid[blocks] / P
        age = np.maximum(f.now - f.close_time[blocks], 0.0)
        since = np.maximum(f.now - np.maximum(f.last_inval[blocks], f.close_time[blocks]), 0.0)
        inv_rate = f.inval_since_close[blocks] / np.maximum(age, 1.0) * U / P
        mean_e = f.erase_cnt.mean()
        wear = (f.erase_cnt[blocks] - mean_e) / (mean_e + 1.0)
        cand[:n, 0] = u
        cand[:n, 1] = np.minimum(age / U, 4.0) / 4.0
        cand[:n, 2] = np.minimum(since / U, 4.0) / 4.0
        cand[:n, 3] = np.minimum(inv_rate, 4.0) / 4.0
        cand[:n, 4] = (f.origin[blocks] == GCW).astype(np.float32)
        cand[:n, 5] = np.clip(wear, -1.0, 1.0)
        cand[:n, 6] = np.minimum(gr, k * 2) / (k * 2)
        cand[:n, 7] = np.minimum(cr, k * 2) / (k * 2)
        mask[:n] = 1.0
        if self.drop:
            cand[:, self.drop] = 0.0
        closed = f.state == CLOSED
        nc = max(int(closed.sum()), 1)
        glob = np.array([
            len(f.free) / max(f.gc_threshold, 1),
            f.valid[closed].sum() / (nc * P),
            float((f.origin[closed] == GCW).sum()) / nc,
            min(self.waf_ema, 20.0) / 20.0,
        ], dtype=np.float32)
        return {"cand": cand, "mask": mask, "glob": glob}, blocks


class VictimEnv:
    def __init__(self, workload: str = "hc80-20", n_blocks: int = 256, ppb: int = 64, op: float = 0.14,
                 k: int = 8, n_greedy: int = 4, episode_len: int = 2000, warmup_writes: float = 2.0,
                 warmup_policy: str = "greedy", seed: int = 0, workload_pool=None, drop=(), continuing: bool = False):
        """continuing=True: the device is filled and warmed up only once; later reset() calls only start a new
        episode (an update segment) on the running device, so the agent lives in the state its own policy makes."""
        self.continuing = continuing
        self._started = False
        self.workload_spec = workload
        self.workload_pool = list(workload_pool) if workload_pool else None
        self.B, self.P, self.op = n_blocks, ppb, op
        self.k, self.n_greedy = k, n_greedy
        self.episode_len = episode_len
        self.warmup_writes = warmup_writes
        self.warmup_policy = warmup_policy
        self.rng = np.random.default_rng(seed)
        self.ftl = FTL(n_blocks, ppb, op)
        self.U = self.ftl.U
        self.obs_builder = ObsBuilder(k, n_greedy, drop)

    def _advance(self):
        """Replay host writes until the FTL needs a victim."""
        f, w = self.ftl, self.wl
        writes = 0
        while not f.need_gc():
            lpn = w.next()
            f._invalidate(lpn)
            f._program_host(lpn)
            f.host_writes += 1
            f.now += 1.0
            writes += 1
        return writes

    def _obs(self):
        obs, self._blocks = self.obs_builder(self.ftl)
        return obs

    def reset(self, seed: int | None = None):
        if self.continuing and self._started:
            if self.workload_pool:
                spec = self.workload_pool[int(self.rng.integers(len(self.workload_pool)))]
                if spec != self.cur_workload:
                    self.cur_workload = spec
                    self.wl = make_workload(spec, self.U, int(self.rng.integers(1 << 31)))
            self.t = 0
            self.ep_host = 0 if self.ftl.need_gc() else self._advance()
            self.ep_copies = 0
            return self._obs(), {"workload": self.cur_workload}
        self._started = True
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        wseed = int(self.rng.integers(1 << 31))
        spec = self.workload_spec
        if self.workload_pool:
            spec = self.workload_pool[int(self.rng.integers(len(self.workload_pool)))]
        self.cur_workload = spec
        f = self.ftl
        f.reset()
        f.victim_fn = make_policy(self.warmup_policy, np.random.default_rng(wseed + 1))
        self.wl = make_workload(spec, self.U, wseed)
        for lpn in range(self.U):                  # sequential fill
            f.write(lpn)
        for _ in range(int(self.warmup_writes * self.U)):
            f.write(self.wl.next())
        f.victim_fn = None
        self.obs_builder.reset(f)
        self.t = 0
        self.ep_host = self._advance()
        self.ep_copies = 0
        return self._obs(), {"workload": spec}

    def step(self, action: int):
        a = int(action)
        if a < 0 or a >= len(self._blocks):
            raise ValueError("invalid action")
        victim = int(self._blocks[a])
        copied = self.ftl.collect_one(victim)
        self.ep_copies += copied
        r = -copied / self.P
        self.t += 1
        w = 0
        if not self.ftl.need_gc():
            w = self._advance()
            self.ep_host += w
        truncated = self.t >= self.episode_len
        return self._obs(), r, False, truncated, {"copied": copied, "host_writes": w}

    def episode_waf(self):
        return 1.0 + self.ep_copies / max(self.ep_host, 1)


class AgentPolicy:
    """Wraps a trained agent as an FTL victim policy (for full evaluation runs)."""

    def __init__(self, agent, k: int = 8, n_greedy: int = 4, drop=(), stochastic: bool = False):
        self.agent = agent
        self.stochastic = stochastic             # sample from a policy-gradient agent instead of taking its argmax
        self.ob = ObsBuilder(k, n_greedy, drop)
        self.name = getattr(agent, "name", "agent")
        self._started = False

    def __call__(self, ftl: FTL) -> int:
        if not self._started:
            self.ob.reset(ftl)
            self._started = True
        obs, blocks = self.ob(ftl)
        a = int(self.agent.act(obs, greedy=not self.stochastic))
        return int(blocks[a])


def run_policy(policy_fn, workload: str, seed: int, n_blocks=256, ppb=64, op=0.14,
               warmup=3.0, measure=4.0, trace_lpns=None) -> dict:
    """Full evaluation run: fill, warm up under the same policy, then measure WAF.

    ``policy_fn(ftl) -> block`` (baselines) or an object with ``.select(ftl)``.
    """
    f = FTL(n_blocks, ppb, op)
    rng = np.random.default_rng(seed)
    f.victim_fn = policy_fn
    wl = make_workload(workload, f.U, seed, trace_lpns=trace_lpns)
    for lpn in range(f.U):
        f.write(lpn)
    for _ in range(int(warmup * f.U)):
        f.write(wl.next())
    c0 = f.counters()
    e0 = f.erase_cnt.copy()
    n = int(measure * f.U)
    for _ in range(n):
        f.write(wl.next())
    c1 = f.counters()
    de = f.erase_cnt - e0
    return dict(waf=1.0 + (c1["copies"] - c0["copies"]) / n,
                gc=c1["gc"] - c0["gc"], copies=c1["copies"] - c0["copies"], host=n,
                erase_mean=float(de.mean()), erase_std=float(de.std()), erase_max=int(de.max()))


class CandidateRandom:
    """Baseline: pick uniformly among the k candidates the agents see (no learning at all).

    It separates what the candidate set alone buys from what the learned scorer adds."""

    def __init__(self, k: int = 8, n_greedy: int = 4, seed: int = 0):
        self.k, self.n_greedy = k, n_greedy
        self.rng = np.random.default_rng(seed)
        self.name = f"cand-random(k={k})"

    def __call__(self, ftl: FTL) -> int:
        blocks = candidate_set(ftl, self.k, self.n_greedy)[0]
        return int(blocks[self.rng.integers(len(blocks))])
