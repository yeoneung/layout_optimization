"""Tests for the trusted witness-carrying part of certified decoding."""

import sys

import numpy as np

from certified_decode import FirstFitWitness


def place(occ, iw, ih, item):
    i, r, c = item
    out = occ.copy()
    out[r:r + int(ih[i]), c:c + int(iw[i])] += 1
    return out


def main():
    iw = np.array([3, 3, 2, 2], dtype=np.int64)
    ih = np.array([2, 2, 2, 2], dtype=np.int64)
    cert = FirstFitWitness(iw, ih, GH=4, GW=6, gw=6, gh=4)
    empty = np.zeros((4, 6), dtype=np.int32)

    ok, generated = cert.certify(empty, list(range(4)))
    assert ok and cert.verify(empty, list(range(4)), generated)

    # This is a valid completion but not presented in generator order.  Commit
    # facilities in another order and carry the remainder without regeneration.
    witness = [(0, 0, 0), (1, 0, 3), (2, 2, 0), (3, 2, 2)]
    remaining = [0, 1, 2, 3]
    occ = empty
    assert cert.verify(occ, remaining, witness)
    for i in (2, 0, 3, 1):
        action = next(x for x in witness if x[0] == i)
        occ = place(occ, iw, ih, action)
        remaining.remove(i)
        witness = [x for x in witness if x[0] != i]
        assert cert.verify(occ, remaining, witness)

    invalid = [(0, 0, 0), (1, 0, 0), (2, 2, 0), (3, 2, 2)]
    assert not cert.verify(empty, list(range(4)), invalid)

    already_invalid = empty.copy()
    already_invalid[0, 0] = 2
    assert not cert.verify(already_invalid, list(range(4)), generated)
    print("CERTIFIED WITNESS CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
