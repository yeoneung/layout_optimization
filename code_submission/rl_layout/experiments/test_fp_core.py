"""
Chain of trust for the floorplanning objective.

The constructive machinery is only usable on these benchmarks if the incremental
form of HPWL is exact, in the same sense the adjacency objective's marginal was:
the marginals accumulated along a construction must reproduce the objective of
the layout they build, to floating-point precision, for any commit order and any
positions.

Two properties are checked.

  1. Sum of marginals equals total HPWL, up to the constant contributed by nets
     whose fixed terminals already span some distance before any block is placed.
     That constant is computed independently here rather than taken from the
     implementation under test.

  2. The marginal is order-independent in the sense that matters: building the
     same final layout by a different commit order gives the same total.
"""

import sys

import numpy as np

import fp_core as FP


def terminal_only_offset(inst):
    """HPWL contributed by fixed terminals before any block is placed."""
    tot = 0.0
    for f in inst.nets_fixed:
        if f is not None:
            tot += (f[1] - f[0]) + (f[3] - f[2])
    return tot


def build(inst, obj, order, cx, cy):
    """Accumulate marginals along `order`, placing block i at (cx[i], cy[i])."""
    box = obj.fresh()
    acc = 0.0
    for i in order:
        g = obj.marginal_grid(i, box, np.array([[cx[i]]]), np.array([[cy[i]]]))
        acc += float(g[0, 0])
        box = obj.commit(i, box, cx[i], cy[i])
    return acc


def main():
    rng = np.random.default_rng(0)
    ok = True
    for name in ("ami33", "ami49", "n100", "n200", "n300"):
        inst = FP.load(name)
        obj = FP.HPWL(inst)
        off = terminal_only_offset(inst)

        for trial in range(3):
            cx = rng.uniform(inst.w / 2, inst.W - inst.w / 2)
            cy = rng.uniform(inst.h / 2, inst.H - inst.h / 2)
            total = obj.total(cx, cy)

            o1 = rng.permutation(inst.n)
            o2 = rng.permutation(inst.n)
            a1 = build(inst, obj, o1, cx, cy) + off
            a2 = build(inst, obj, o2, cx, cy) + off
            e1 = abs(a1 - total)
            e2 = abs(a2 - total)
            rel = e1 / max(abs(total), 1.0)
            if trial == 0:
                print(f"{name:6s} total={total:12.2f} offset={off:10.2f} "
                      f"|sum-total|={e1:.3e} (rel {rel:.2e})  order-swap diff={abs(a1 - a2):.3e}")
            if rel > 1e-9 or abs(a1 - a2) > 1e-6:
                ok = False
                print(f"  FAIL on {name} trial {trial}: {e1:.6f} / {abs(a1 - a2):.6f}")

    print("\nMARGINAL IS EXACT" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
