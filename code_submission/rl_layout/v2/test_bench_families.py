"""Structural checks for the slicing and nonslicing benchmark families."""

import sys

import numpy as np

import bench


def area(rects):
    return sum(w * h for x, y, w, h in rects)


def no_overlap(rects):
    for i, (x, y, w, h) in enumerate(rects):
        for xx, yy, ww, hh in rects[i + 1:]:
            if min(x + w, xx + ww) > max(x, xx) and \
                    min(y + h, yy + hh) > max(y, yy):
                return False
    return True


def main():
    rng = np.random.default_rng(20260907)
    for n in (5, 16, 32, 64):
        rects = bench.nonslicing(n, 44, 44, rng)
        assert len(rects) == n
        assert area(rects) == 44 * 44
        assert no_overlap(rects)
        assert not bench._has_full_cut(rects, 44, 44)

    for geometry in ("guillotine", "nonslicing"):
        cell = bench.suite_cell("val", 32, 0.90, 8,
                                tag="family_test_" + geometry,
                                geometry=geometry)
        check = bench.verify_witness(cell)
        assert check["all_inside"] and check["max_overlap"] <= 1e-9
    print("BENCHMARK FAMILY CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

