"""
The certificate question on real floorplanning benchmarks.

The generated suite of Section 7 is built so that a feasible packing provably
exists, which is what lets a failure be attributed to the method.  The MCNC and
GSRC benchmarks carry no such witness: if a constructor cannot fit ami33 into a
plate with 5% dead space, we cannot say whether that is the constructor's fault
or the instance's.

The metric is therefore changed rather than the claim weakened.  For each method
we sweep the dead space downward and record the tightest outline it can fill ---
its *packing frontier* --- and, at a dead space every method survives, the
wirelength it achieves.  The frontier needs no witness: it is a property of the
method on a fixed instance, and a method that packs the same real blocks into a
tighter outline is doing something the others cannot.

Fixed-outline floorplanning conventionally reports 15% dead space, so that is
where wirelength is compared; the frontier is where the certificates separate.
"""

import argparse
import json
import time

import numpy as np

import fp_core as FP
from fp_construct import FPConstructor

METHODS = [
    ("no certificate", dict(filter_mode="none")),
    ("first-fit", dict(filter_mode="pack")),
    ("rollout", dict(filter_mode="rollout")),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benches", nargs="+",
                   default=["ami33", "ami49", "n100", "n200", "n300"])
    p.add_argument("--dead", nargs="+", type=float,
                   default=[0.30, 0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06])
    p.add_argument("--compare-dead", dest="cmp_dead", type=float, default=0.15)
    p.add_argument("--rule", default="area", choices=["area", "greedy"])
    p.add_argument("--n-cand", dest="n_cand", type=int, default=8)
    p.add_argument("--rollout-max-n", dest="roll_max", type=int, default=100)
    p.add_argument("--out", default="fp_results.json")
    a = p.parse_args()

    rows = []
    for nm in a.benches:
        probe = FP.load(nm, dead_space=a.cmp_dead)
        print(f"\n=== {nm}: {probe.n} blocks, {len(probe.nets_blocks)} nets, "
              f"scale {probe.scale}, plate {probe.W}x{probe.H} at "
              f"{a.cmp_dead:.0%} dead space ===", flush=True)

        for label, kw in METHODS:
            if label == "rollout" and probe.n > a.roll_max:
                rows.append({"bench": nm, "method": label, "skipped": True})
                print(f"  {label:16s} SKIPPED (rollout too slow at n={probe.n})",
                      flush=True)
                continue

            frontier, hp_at_cmp, t_at_cmp = None, None, None
            for ds in a.dead:
                inst = FP.load(nm, dead_space=ds)
                t0 = time.time()
                r = FPConstructor(inst, room_rule=a.rule, n_cand=a.n_cand,
                                  **kw).solve()
                el = time.time() - t0
                if r["complete"]:
                    frontier = ds                      # tightest so far
                if abs(ds - a.cmp_dead) < 1e-9:
                    hp_at_cmp = r["hpwl"] if r["complete"] else None
                    t_at_cmp = el
                print(f"  {label:16s} dead={ds:.2f} complete={str(r['complete']):5s} "
                      f"HPWL={r['hpwl'] if r['complete'] else float('nan'):10.1f} "
                      f"fb={r['fallbacks']:3d} {el:7.1f}s", flush=True)

            rows.append({"bench": nm, "method": label, "skipped": False,
                         "frontier_dead_space": frontier,
                         "hpwl_at_compare": hp_at_cmp,
                         "seconds_at_compare": t_at_cmp,
                         "n_blocks": probe.n, "scale": probe.scale})
            with open(a.out, "w") as f:
                json.dump({"compare_dead": a.cmp_dead, "rows": rows}, f, indent=1)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
