"""Page-mapped FTL simulator (garbage collection + write amplification).

The model is deliberately small and explicit so every line can be checked by hand:

* NAND = ``n_blocks`` erase blocks x ``ppb`` pages.  A page is FREE, VALID or INVALID.
* Host writes one logical page (LPN) at a time.  The FTL never overwrites in place:
  the old physical page becomes INVALID and the data goes to the next free page of the
  *host* write frontier (an open block).
* Garbage collection (GC) picks a *victim* among CLOSED blocks, copies its VALID pages to
  the *GC* write frontier (a second open block) and erases the victim.
* Foreground GC runs inside a host write whenever the free-block pool drops below
  ``gc_threshold``.  Background GC (``collect_one``) can be called from outside.

Logical capacity follows the industry definition of over-provisioning:
``op = (physical - logical) / logical``  ->  ``user_pages = floor(physical / (1 + op))``.

The clock ``self.now`` counts host page writes ("write time").  Block metadata
(close time, last invalidation time, invalidations since close, origin) feeds the
victim policies and the RL state.
"""
from __future__ import annotations

from collections import deque

import numpy as np

FREE, OPEN, CLOSED = 0, 1, 2
HOST, GCW = 0, 1


class FTL:
    def __init__(self, n_blocks: int = 256, ppb: int = 64, op: float = 0.07,
                 gc_threshold: int = 2, separate_gc_frontier: bool = True, user_pages: int | None = None):
        if gc_threshold < 2:
            raise ValueError("gc_threshold must be >= 2 (one free block is reserved for the GC frontier)")
        self.B = int(n_blocks)
        self.P = int(ppb)
        self.op = float(op)
        self.gc_threshold = int(gc_threshold)
        self.separate = bool(separate_gc_frontier)
        self.T = self.B * self.P
        self.U = int(user_pages) if user_pages is not None else int(self.T / (1.0 + self.op))
        # The free pool may not be eaten by user data: at most (B - threshold - open blocks) blocks hold data.
        open_blocks = 2 if self.separate else 1
        if self.U > (self.B - self.gc_threshold - open_blocks) * self.P:
            raise ValueError("over-provisioning too small for gc_threshold and open blocks")
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self) -> None:
        B, P = self.B, self.P
        self.l2p = [-1] * self.U              # logical page -> physical page
        self.p2l = [-1] * self.T              # physical page -> logical page (-1 = not valid)
        self.valid = np.zeros(B, dtype=np.int64)
        self.state = np.zeros(B, dtype=np.int8)        # FREE / OPEN / CLOSED
        self.erase_cnt = np.zeros(B, dtype=np.int64)
        self.close_time = np.zeros(B, dtype=np.float64)
        self.open_time = np.zeros(B, dtype=np.float64)
        self.last_inval = np.zeros(B, dtype=np.float64)
        self.inval_since_close = np.zeros(B, dtype=np.int64)
        self.origin = np.zeros(B, dtype=np.int8)       # HOST or GCW frontier wrote the block
        self.free = deque(range(B))
        self.now = 0.0
        self.host_writes = 0
        self.gc_copies = 0
        self.erases = 0
        self.gc_count = 0
        self.fg_gc_count = 0
        self.victim_fn = None
        self.on_gc = None                     # optional callback(victim, copies, foreground)
        self.host_blk = self._alloc(HOST)
        self.host_ptr = 0
        if self.separate:
            self.gc_blk = self._alloc(GCW)
            self.gc_ptr = 0

    def _alloc(self, origin: int) -> int:
        b = self.free.popleft()
        self.state[b] = OPEN
        self.origin[b] = origin
        self.open_time[b] = self.now
        return b

    # ------------------------------------------------------------- primitives
    def _invalidate(self, lpn: int) -> None:
        old = self.l2p[lpn]
        if old >= 0:
            b = old // self.P
            self.p2l[old] = -1
            self.valid[b] -= 1
            self.last_inval[b] = self.now
            if self.state[b] == CLOSED:
                self.inval_since_close[b] += 1

    def _program_host(self, lpn: int) -> None:
        b = self.host_blk
        ppn = b * self.P + self.host_ptr
        self.l2p[lpn] = ppn
        self.p2l[ppn] = lpn
        self.valid[b] += 1
        self.host_ptr += 1
        if self.host_ptr == self.P:
            self._close(b)
            self.host_blk = self._alloc(HOST)
            self.host_ptr = 0

    def _program_gc(self, lpn: int) -> None:
        if not self.separate:
            self._program_host(lpn)
            return
        b = self.gc_blk
        ppn = b * self.P + self.gc_ptr
        self.l2p[lpn] = ppn
        self.p2l[ppn] = lpn
        self.valid[b] += 1
        self.gc_ptr += 1
        if self.gc_ptr == self.P:
            self._close(b)
            self.gc_blk = self._alloc(GCW)
            self.gc_ptr = 0

    def _close(self, b: int) -> None:
        self.state[b] = CLOSED
        self.close_time[b] = self.now
        self.inval_since_close[b] = 0

    # --------------------------------------------------------------- host I/O
    def write(self, lpn: int) -> int:
        """One host page write.  Returns the number of pages copied by foreground GC."""
        self._invalidate(lpn)
        self._program_host(lpn)
        self.host_writes += 1
        self.now += 1.0
        copied = 0
        while len(self.free) < self.gc_threshold:
            copied += self.collect_one(foreground=True)
        return copied

    def need_gc(self) -> bool:
        return len(self.free) < self.gc_threshold

    # ---------------------------------------------------------------- GC core
    def closed_blocks(self) -> np.ndarray:
        return np.flatnonzero(self.state == CLOSED)

    def collect_one(self, victim: int | None = None, foreground: bool = False) -> int:
        """Collect one victim block.  Returns the number of valid pages copied."""
        if victim is None:
            if self.victim_fn is None:
                raise RuntimeError("no victim policy set")
            victim = int(self.victim_fn(self))
        if self.state[victim] != CLOSED:
            raise ValueError(f"victim {victim} is not a closed block")
        if self.valid[victim] >= self.P and not getattr(self.victim_fn, "allow_full", False):
            raise ValueError(f"victim {victim} has no invalid page")
        P = self.P
        base = victim * P
        p2l = self.p2l
        copied = 0
        for off in range(P):
            lpn = p2l[base + off]
            if lpn >= 0:
                self._program_gc(lpn)
                copied += 1
        p2l[base:base + P] = [-1] * P
        self.valid[victim] = 0
        self.state[victim] = FREE
        self.erase_cnt[victim] += 1
        self.inval_since_close[victim] = 0
        self.free.append(victim)
        self.gc_copies += copied
        self.erases += 1
        self.gc_count += 1
        if foreground:
            self.fg_gc_count += 1
        if self.on_gc is not None:
            self.on_gc(victim, copied, foreground)
        return copied

    # ---------------------------------------------------------------- metrics
    def waf(self) -> float:
        if self.host_writes == 0:
            return float("nan")
        return (self.host_writes + self.gc_copies) / self.host_writes

    def counters(self) -> dict:
        return dict(host=self.host_writes, copies=self.gc_copies, erases=self.erases,
                    gc=self.gc_count, fg_gc=self.fg_gc_count)

    def check_invariants(self) -> None:
        """Expensive consistency check (tests / validation only)."""
        P = self.P
        seen = 0
        for lpn, ppn in enumerate(self.l2p):
            if ppn >= 0:
                assert self.p2l[ppn] == lpn, (lpn, ppn)
                seen += 1
        assert sum(1 for x in self.p2l if x >= 0) == seen
        counts = np.zeros(self.B, dtype=np.int64)
        for ppn, lpn in enumerate(self.p2l):
            if lpn >= 0:
                counts[ppn // P] += 1
        assert np.array_equal(counts, self.valid), "valid counts drifted"
        n_free = int(np.sum(self.state == FREE))
        assert n_free == len(self.free)
        assert all(self.state[b] == FREE for b in self.free)
        open_expected = 2 if self.separate else 1
        assert int(np.sum(self.state == OPEN)) == open_expected
        assert int(self.valid[self.state == FREE].sum()) == 0
