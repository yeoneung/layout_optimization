"""
Does the training-free constructor cover the policy's exclusive region?

Section 8.6 claims one region for the learned policy alone: high quality
(J > 3400) at moderate diversity (0.30-0.45).  The claim only holds if the greedy
heuristic cannot be tuned into that region, and the first sweep did not settle
it -- score jitter was swept from sigma=2 upward and every setting landed at
diversity 0.66-0.69, while a fully random commit order landed at 0.603.  Nothing
was measured between 0.0 and 0.60, which is exactly where the contested region
sits.

This closes the gap by sweeping both knobs finely at the low end: the fraction of
commit slots that get reshuffled, and small score jitter.  If any setting reaches
diversity 0.30-0.45 at J > 3400, the policy has no exclusive region and the
manuscript has to say so.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                              # noqa: E402

import core                                                    # noqa: E402
import evaluate2 as E                                          # noqa: E402
from greedy_construct import GreedyConstructor                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="comb_high")
    p.add_argument("--flip-adj", dest="flip_adj", action="store_true", default=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="qd_greedy_sweep.json")
    a = p.parse_args()

    spec = build_spec(a.scenario, flip_adj=a.flip_adj)
    K = a.samples
    inst = core.from_spec(spec, K)
    obj = core.BatchObjective(inst)
    rng = np.random.default_rng(a.seed)
    gc = GreedyConstructor(inst)
    rows = []

    def add(name, r, wall):
        tot, c = obj.evaluate(r["x"], r["y"], components=True)
        d = E.diversity(inst, r["x"], r["y"], rng=np.random.default_rng(a.seed + 99))
        row = {"method": name, "quality_mean": float(tot.mean()),
               "quality_std": float(tot.std()), "quality_max": float(tot.max()),
               "diversity": d,
               "feasible_rate": float(np.mean(c["overlap_area"] <= 1e-9)),
               "ms_per_sample": wall / K * 1000,
               "evals_per_sample": r["evals"] / K}
        rows.append(row)
        print(f"{name:44s} J={row['quality_mean']:7.1f} div={d:.3f} "
              f"feas={row['feasible_rate']:4.2f}")
        return row

    print("--- commit-order noise (fraction of slots reshuffled) ---")
    for f in (0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50, 0.70, 1.00):
        t = time.time()
        r = gc.solve(rng=np.random.default_rng(a.seed + 1), room_rule="random",
                     order_noise=f)
        add(f"greedy, order noise {f:.2f}", r, time.time() - t)

    print("\n--- small score jitter, area order ---")
    for s in (0.05, 0.1, 0.25, 0.5, 1.0):
        t = time.time()
        r = gc.solve(rng=np.random.default_rng(a.seed + 2), jitter=s,
                     room_rule="area")
        add(f"greedy, area order, jitter {s:g}", r, time.time() - t)

    # ---- the verdict, stated in the output rather than inferred later ------
    band = [r for r in rows if 0.28 <= r["diversity"] <= 0.47]
    print("\n--- rows landing in the contested band (diversity 0.28-0.47) ---")
    if not band:
        print("  none: no setting of either knob reaches that diversity range")
    for r in sorted(band, key=lambda r: -r["quality_mean"]):
        print(f"  div={r['diversity']:.3f}  J={r['quality_mean']:7.1f}  {r['method']}")
    best = max(band, key=lambda r: r["quality_mean"]) if band else None
    alone = not (best and best["quality_mean"] > 3400)
    print("\npolicy holds the region alone: " + ("YES" if alone else "NO"))
    if best:
        print(f"  best heuristic inside the band: J={best['quality_mean']:.1f} "
              f"at diversity {best['diversity']:.3f}  ({best['method']})")

    with open(a.out, "w") as f:
        json.dump({"scenario": a.scenario, "samples": K, "rows": rows}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
