"""Adversarial regression tests for the independent final verifier."""

import sys

import numpy as np

from layout_verifier import verify_complete_layout, verify_witness_completion


def main():
    iw = np.array([2, 2, 1, 1], dtype=np.int64)
    ih = np.array([2, 2, 2, 2], dtype=np.int64)
    good = [(0, 0, 0), (1, 0, 2), (2, 0, 4), (3, 0, 5)]

    # Legal edge contact is not overlap.
    assert verify_complete_layout(iw, ih, 6, 2, good).valid

    duplicate = [(0, 0, 0), (0, 0, 2), (2, 0, 4), (3, 0, 5)]
    assert verify_complete_layout(iw, ih, 6, 2, duplicate).reason == "duplicate facility"
    assert verify_complete_layout(iw, ih, 6, 2, good[:-1]).reason == \
        "missing or extra facility"

    outside = list(good)
    outside[-1] = (3, 0, 6)
    assert verify_complete_layout(iw, ih, 6, 2, outside).reason == "out of bounds"

    overlap = list(good)
    overlap[1] = (1, 0, 1)
    assert verify_complete_layout(iw, ih, 6, 2, overlap).reason == "overlap"

    # A witness generated before a changed commitment is rejected as stale.
    committed = [(0, 0, 0)]
    witness = good[1:]
    assert verify_witness_completion(iw, ih, 6, 2, committed, witness).valid
    stale_prefix = [(0, 0, 1)]
    assert not verify_complete_layout(iw, ih, 6, 2, good,
                                      prefix=stale_prefix).valid

    # An interrupted repair cannot pass by returning a partial list.
    incomplete_repair = committed + witness[:-1]
    assert not verify_complete_layout(iw, ih, 6, 2, incomplete_repair).valid
    print("INDEPENDENT LAYOUT VERIFIER CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

