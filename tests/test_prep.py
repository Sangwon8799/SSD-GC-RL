"""The MSR trace converter on a hand-made 4-row trace (python -m tests.test_prep)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "scripts"))
from prep_msr import convert  # noqa: E402


def test_convert_by_hand():
    rows = [["1", "hm", "1", "Write", "8192", "8192", "1"],    # pages 2, 3       -> 0, 1
            ["2", "hm", "1", "Read", "0", "4096", "1"],       # read: ignored
            ["3", "hm", "1", "Write", "4096", "6000", "1"],   # bytes 4096..10095 -> pages 1, 2 -> 2, 0
            ["4", "hm", "1", "Write", "8192", "4096", "1"]]   # page 2            -> 0
    lpns, footprint = convert(rows)
    assert lpns.tolist() == [0, 1, 2, 0, 0]
    assert footprint == 3


if __name__ == "__main__":
    test_convert_by_hand()
    print("ok test_convert_by_hand")
