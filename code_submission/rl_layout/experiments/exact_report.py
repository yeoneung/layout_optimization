"""
Join proven optima with heuristic results and print optimality gaps.

Heuristic gaps are summarized on instances for which CP-SAT proved the scaled
integer model optimal.  Those solutions are certified within twice the stored
quantization allowance for the stated real objective.  Instances left at
FEASIBLE contribute a real-objective upper bound, not a point optimum.
A method's gap on an instance where it produced an infeasible layout is
undefined and reported as such rather than imputed.
"""

import argparse
import json
from collections import defaultdict

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--exact", default="exact_results_audited.json")
    p.add_argument("--heur", default="exact_cells_heur.json")
    a = p.parse_args()

    with open(a.exact) as f:
        ex = json.load(f)["results"]
    with open(a.heur) as f:
        hr = json.load(f)["records"]

    opt = {}
    bound = {}
    n_opt = n_feas_only = 0
    for r in ex:
        key = (r["fill"], r["instance"])
        if r["status"] == "OPTIMAL":
            opt[key] = r["objective"]
            bound[key] = r["bound"]
            n_opt += 1
        elif r["status"] == "FEASIBLE":
            bound[key] = r["bound"]      # valid upper bound on the optimum
            n_feas_only += 1
    print(f"CP-SAT: {n_opt} quantized optima, {n_feas_only} feasible solutions "
          f"with bounds, of {len(ex)}")

    by = defaultdict(list)
    infeas = defaultdict(int)
    for r in hr:
        key = (r["fill"], r["instance"])
        if key not in opt:
            continue
        if not r["feasible"]:
            infeas[r["method"]] += 1
            continue
        by[r["method"]].append((opt[key] - r["J"]) / abs(opt[key]))

    print(f"\n{'method':28s} {'n':>4s} {'mean gap':>9s} {'median':>8s} "
          f"{'max':>8s} {'#opt':>5s} {'#infeas':>8s}")
    for m in sorted(by, key=lambda m: np.mean(by[m])):
        g = np.array(by[m])
        print(f"{m:28s} {len(g):4d} {100 * g.mean():8.2f}% "
              f"{100 * np.median(g):7.2f}% {100 * g.max():7.2f}% "
              f"{(g <= 1e-9).sum():5d} {infeas[m]:8d}")

    # per-fill means, for the manuscript table
    print()
    fills = sorted({r["fill"] for r in ex})
    for m in sorted(by, key=lambda m: np.mean(by[m])):
        row = []
        for f in fills:
            g = [(opt[(rf, ri)] - r["J"]) / abs(opt[(rf, ri)])
                 for r in hr
                 for (rf, ri) in [(r["fill"], r["instance"])]
                 if r["method"] == m and rf == f and (rf, ri) in opt
                 and r["feasible"]]
            row.append(f"{100 * np.mean(g):5.2f}%" if g else "  --  ")
        print(f"{m:28s} " + "  ".join(row))

    # Certified gap upper bound on every instance CP-SAT bounded, closed or not.
    # The audit has already added delta to the quantized bound.  For positive J,
    # (U - J)/U bounds the relative gap to the unknown real optimum.
    print("\nCERTIFIED GAP UPPER BOUND (all bounded instances, incl. unclosed)")
    byb = defaultdict(list)
    for r in hr:
        key = (r["fill"], r["instance"])
        if key in bound and r["feasible"]:
            byb[r["method"]].append((bound[key] - r["J"]) / abs(bound[key]))
    for m in sorted(byb, key=lambda m: np.mean(byb[m])):
        g = np.array(byb[m])
        print(f"{m:28s} n={len(g):3d}  mean<= {100 * g.mean():5.2f}%  "
              f"median<= {100 * np.median(g):5.2f}%  max<= {100 * g.max():5.2f}%")


if __name__ == "__main__":
    main()
