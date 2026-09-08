"""
Correctness checks for the non-myopic constructive baselines.

Two properties have to hold before any number they produce is worth reporting:

  1. A beam of width 1 is exactly the joint room+position greedy rule.  If it is
     not, the beam bookkeeping is wrong and every wider beam is meaningless.
  2. Every layout produced is overlap-free, since all three procedures act inside
     the same mask.

Beam quality is deliberately *not* asserted to be monotone in the width.  Beam
search is not monotone in general: a wider beam can retain a high-scoring partial
layout whose successors crowd out the line a narrower beam would have followed.
We observed exactly that here (hospital, width 4 -> 8), so the paper must report
the width sweep rather than a single width, and must choose the width on
validation instances rather than on the test scenario.
"""

import sys

import numpy as np

import core
from constructive_search import BeamConstructor, regret_construct
from greedy_construct import GreedyConstructor

sys.path.insert(0, "..")
from layout_env import build_spec                                 # noqa: E402


def main():
    ok = True
    for scen in ("comb_high", "hospital"):
        spec = build_spec(scen, flip_adj=True)
        inst = core.from_spec(spec, 1)
        obj = core.BatchObjective(inst)

        g = GreedyConstructor(inst).solve(room_rule="greedy")
        b1 = BeamConstructor(inst, width=1).solve()
        d = abs(float(g["best"][0]) - b1["best"])
        print(f"{scen}: greedy={float(g['best'][0]):.6f}  beam1={b1['best']:.6f}  "
              f"|diff|={d:.2e}")
        if d > 1e-6:
            ok = False
            print("  FAIL: beam width 1 does not reproduce the greedy rule")

        for W in (1, 2, 4, 8, 16):
            r = BeamConstructor(inst, width=W).solve()
            ov = float(obj.evaluate(r["x"], r["y"], components=True)[1]["overlap_area"][0])
            print(f"  beam {W:3d}: J={r['best']:9.2f}  overlap={ov:.3g}  "
                  f"candidates={r['candidates']:,}")
            if ov > 1e-9:
                ok = False
                print("  FAIL: beam produced an overlapping layout")

        rg = regret_construct(inst, k=2)
        ovr = float(rg["overlap_area"][0])
        print(f"  regret-2: J={float(rg['best'][0]):9.2f}  overlap={ovr:.3g}")
        if ovr > 1e-9:
            ok = False
            print("  FAIL: regret insertion produced an overlapping layout")

    print("\nALL CHECKS PASSED" if ok else "\nCHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
