"""Hand-checked cases for the FTL simulator.  Run:  python -m tests.test_ftl  (from the repo root)

Every expected number below was worked out by hand on paper first (see the comments);
the test only replays the same steps and compares.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

from ssdgc.ftl import CLOSED, FREE, FTL, GCW, HOST, OPEN
from ssdgc.policies import CostBenefit, FIFO, Greedy, cb_score


def tiny():
    # 6 blocks x 4 pages = 24 physical pages, op = 1.0 -> 12 logical pages.
    # capacity check: 12 <= (6 - 2 threshold - 2 open) * 4 = 8 ?  -> no, so use op=2.0 -> U = 8
    return FTL(n_blocks=6, ppb=4, op=2.0, gc_threshold=2)


def test_layout_and_sequential_fill():
    f = tiny()
    assert f.U == 8                     # 24 / (1 + 2.0) = 8 logical pages
    assert list(f.free) == [2, 3, 4, 5]  # blocks 0 (host) and 1 (GC frontier) are open
    assert f.state[0] == OPEN and f.state[1] == OPEN
    f.victim_fn = Greedy()
    for lpn in range(4):                 # fills block 0 exactly -> closed, host moves to block 2
        f.write(lpn)
    assert f.state[0] == CLOSED and f.host_blk == 2 and f.host_ptr == 0
    assert f.l2p[:4] == [0, 1, 2, 3]
    assert f.valid[0] == 4
    for lpn in range(4, 8):              # block 2 full -> closed, host moves to block 3
        f.write(lpn)
    assert f.state[2] == CLOSED and f.host_blk == 3
    assert list(f.free) == [4, 5]        # 2 free blocks = threshold, no GC yet
    assert f.gc_count == 0
    f.check_invariants()


def test_overwrite_invalidates_old_page():
    f = tiny()
    f.victim_fn = Greedy()
    for lpn in range(8):
        f.write(lpn)
    # overwrite lpn 1 and 2: old pages 1, 2 (block 0) become invalid, new data in block 3 pages 0, 1
    f.write(1)
    f.write(2)
    assert f.p2l[1] == -1 and f.p2l[2] == -1
    assert f.l2p[1] == 3 * 4 + 0 and f.l2p[2] == 3 * 4 + 1
    assert f.valid[0] == 2 and f.valid[3] == 2
    assert f.inval_since_close[0] == 2
    f.check_invariants()


def test_greedy_gc_by_hand():
    """Worked example (also drawn on the study page, section 5).

    fill 0..7  -> block0 = [0 1 2 3], block2 = [4 5 6 7], host -> block3, free = [4, 5]
    write 1, 2, 3, 5  -> block3 = [1 2 3 5]; block0 valid = {0} (1 valid), block2 valid = {4, 6, 7} (3 valid)
                         block3 closes, host -> block4, free = [5]  -> 1 < threshold 2 -> GC
    greedy victim = block0 (1 valid).  Copy lpn 0 to GC frontier block1 page 0 -> 1 copy.
    erase block0 -> free = [5, 0].
    """
    f = tiny()
    f.victim_fn = Greedy()
    for lpn in range(8):
        f.write(lpn)
    copies = 0
    for lpn in (1, 2, 3):
        copies += f.write(lpn)
    assert f.gc_count == 0 and copies == 0
    copies += f.write(5)
    assert f.gc_count == 1 and copies == 1
    assert f.state[0] == FREE and f.erase_cnt[0] == 1
    assert f.l2p[0] == 1 * 4 + 0              # lpn 0 now in GC block 1, page 0
    assert f.origin[1] == GCW and f.valid[1] == 1
    assert list(f.free) == [5, 0]
    assert f.host_writes == 12 and f.gc_copies == 1
    assert abs(f.waf() - 13 / 12) < 1e-12        # (12 host + 1 copy) / 12
    f.check_invariants()


def test_cost_benefit_prefers_old_cold_block():
    """Two candidates: young block with u=0.25 vs old block with u=0.5.
    CB = (1-u)*age/(2u):  young (age 1): 0.75*1/0.5 = 1.5 ; old (age 9): 0.5*9/1.0 = 4.5 -> old wins.
    greedy would pick the young one (fewer valid pages)."""
    f = FTL(n_blocks=8, ppb=4, op=2.0, gc_threshold=2)
    f.state[:] = FREE
    f.free.clear()
    f.state[5] = CLOSED; f.valid[5] = 1; f.close_time[5] = 10.0   # young, u = 0.25
    f.state[6] = CLOSED; f.valid[6] = 2; f.close_time[6] = 2.0    # old,   u = 0.5
    f.now = 10.0
    s = cb_score(f, np.array([5, 6]))
    assert abs(s[0] - 1.5) < 1e-12 and abs(s[1] - 4.5) < 1e-12
    assert CostBenefit()(f) == 6
    assert Greedy()(f) == 5
    assert FIFO()(f) == 6


def test_random_long_run_invariants():
    rng = np.random.default_rng(3)
    f = FTL(n_blocks=32, ppb=8, op=0.25, gc_threshold=2)
    f.victim_fn = Greedy()
    for lpn in range(f.U):
        f.write(lpn)
    for i in range(20000):
        f.write(int(rng.integers(f.U)))
        if i % 5000 == 0:
            f.check_invariants()
    f.check_invariants()
    # page conservation: every NAND program is either a host write or a GC copy
    programmed = f.host_writes + f.gc_copies
    erased_pages = f.erases * f.P
    in_use = int((f.state != FREE).sum()) * f.P - (f.P - f.host_ptr) - (f.P - f.gc_ptr)
    assert programmed == erased_pages + in_use


def main():
    tests = [v for k, v in globals().items() if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok", t.__name__)
    print(f"ALL {len(tests)} TESTS PASSED")


if __name__ == "__main__":
    main()
