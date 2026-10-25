"""Closed-form write amplification under uniform random single-page writes (mean-field, many pages per block).

alpha = (physical pages that can hold data) / (logical pages) = 1 + rho   (rho = over-provisioning factor)

FIFO (= LRU cleaning, Menon 1995; the same expression is derived for greedy-like cleaning by
Xiang & Kurkoski, arXiv:1110.4245, with the Lambert W function):
    a victim block was written T/A host writes ago; each of its pages survived with probability
    exp(-(T/A)/U), so  u = exp(-alpha (1 - u))  and  WAF = 1 / (1 - u).

Pure random victim (this project's own derivation, same assumptions): the age of a block at
collection is exponential with mean T/A, hence  u = E[exp(-X/U)] = A / (A + alpha)  and with
u = 1 - 1/A:   WAF = alpha / (alpha - 1).
"""
from __future__ import annotations

import math


def fifo_u(alpha: float) -> float:
    """Non-trivial root of u = exp(-alpha (1-u)) in (0, 1), alpha > 1 (bisection)."""
    if alpha <= 1.0:
        return 1.0
    lo, hi = 0.0, 1.0 - 1e-12
    g = lambda u: u - math.exp(-alpha * (1.0 - u))  # noqa: E731
    for _ in range(200):
        mid = (lo + hi) / 2
        if g(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def fifo_waf(alpha: float) -> float:
    return 1.0 / (1.0 - fifo_u(alpha))


def lambertw0(z: float) -> float:
    """Principal branch W0(z) for z in [-1/e, 0) via Halley iteration (used only to cross-check fifo_u)."""
    w = -0.5 if z < -0.25 else z
    for _ in range(100):
        ew = math.exp(w)
        f = w * ew - z
        wn = w - f / (ew * (w + 1) - (w + 2) * f / (2 * w + 2))
        if abs(wn - w) < 1e-15:
            return wn
        w = wn
    return w


def xiang_kurkoski_waf(rho: float) -> float:
    """A = (-1-rho) / (-1-rho - W0((-1-rho) e^(-1-rho)))  (arXiv:1110.4245, asymptotic form)."""
    a = -1.0 - rho
    return a / (a - lambertw0(a * math.exp(a)))


def random_waf(alpha: float) -> float:
    return alpha / (alpha - 1.0)


def agarwal_waf(rho: float) -> float:
    """Older approximation (1 + rho) / (2 rho) quoted in arXiv:1110.4245."""
    return 0.5 * (1.0 + rho) / rho
