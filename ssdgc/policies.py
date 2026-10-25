"""Hand-written victim-selection policies (the baselines).

Every policy is a callable ``policy(ftl) -> block id`` that only looks at CLOSED blocks
that still contain at least one invalid page.
"""
from __future__ import annotations

import numpy as np

from .ftl import CLOSED, FTL


def eligible(ftl: FTL) -> np.ndarray:
    """CLOSED blocks that free at least one page when collected."""
    return np.flatnonzero((ftl.state == CLOSED) & (ftl.valid < ftl.P))


def cb_score(ftl: FTL, blocks: np.ndarray) -> np.ndarray:
    """Cost-benefit score (Rosenblum & Ousterhout, LFS 1992): (1-u)*age / (2u).

    u = valid ratio, age = write-time since the block was closed (its youngest data).
    A block with u = 0 gets +inf (it is free space already).
    """
    u = ftl.valid[blocks] / ftl.P
    age = ftl.now - ftl.close_time[blocks] + 1.0
    with np.errstate(divide="ignore"):
        s = (1.0 - u) * age / (2.0 * u)
    return s


class Greedy:
    name = "greedy"

    def __call__(self, ftl: FTL) -> int:
        cand = eligible(ftl)
        return int(cand[np.argmin(ftl.valid[cand])])


class CostBenefit:
    name = "cost-benefit"

    def __call__(self, ftl: FTL) -> int:
        cand = eligible(ftl)
        return int(cand[np.argmax(cb_score(ftl, cand))])


class FIFO:
    """Oldest closed block first (a.k.a. LRU cleaning)."""
    name = "fifo"

    def __call__(self, ftl: FTL) -> int:
        cand = eligible(ftl)
        return int(cand[np.argmin(ftl.close_time[cand])])


class RandomVictim:
    name = "random"

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, ftl: FTL) -> int:
        cand = eligible(ftl)
        return int(cand[self.rng.integers(len(cand))])


class PureRandom:
    """Uniform over *all* closed blocks, even fully valid ones (theory check only — wasteful)."""
    name = "pure-random"
    allow_full = True

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def __call__(self, ftl: FTL) -> int:
        cand = np.flatnonzero(ftl.state == CLOSED)
        return int(cand[self.rng.integers(len(cand))])


class DChoices:
    """Sample d eligible blocks uniformly, take the one with fewest valid pages (Van Houdt 2013)."""

    def __init__(self, d: int, rng: np.random.Generator):
        self.d = int(d)
        self.rng = rng
        self.name = f"d-choices(d={d})"

    def __call__(self, ftl: FTL) -> int:
        cand = eligible(ftl)
        pick = self.rng.choice(len(cand), size=min(self.d, len(cand)), replace=False)
        sub = cand[pick]
        return int(sub[np.argmin(ftl.valid[sub])])


def make_policy(name: str, rng: np.random.Generator):
    if name == "greedy":
        return Greedy()
    if name in ("cb", "cost-benefit"):
        return CostBenefit()
    if name == "fifo":
        return FIFO()
    if name == "random":
        return RandomVictim(rng)
    if name == "pure-random":
        return PureRandom(rng)
    if name.startswith("d"):
        return DChoices(int(name[1:]), rng)
    raise KeyError(name)
