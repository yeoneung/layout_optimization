"""
Add the training-free constructive heuristic to the quality/diversity front.

The diversity result of Track E rests on a specific mechanism: drawing the order
in which rooms are committed at random, so that the same masked action set is
traversed differently on every rollout.  Nothing about that mechanism is learned.
A greedy rule inside the same masked action set can be given a random commit
order too, and it can be given score jitter as a second source of variety.  If
the learned front does not dominate those, the diversity claim belongs to the
constructive formulation rather than to the policy -- which is exactly the
distinction the rest of the paper is about, so it has to be measured here as
well.

Rows are merged into the existing qd_*.json, on the same K instances and with
the same diversity metric, so the fronts are directly comparable.
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
    p.add_argument("--out", default="qd_comb_high.json")
    a = p.parse_args()

    spec = build_spec(a.scenario, flip_adj=a.flip_adj)
    K = a.samples
    inst = core.from_spec(spec, K)
    obj = core.BatchObjective(inst)
    rng = np.random.default_rng(a.seed)
    gc = GreedyConstructor(inst)

    rows = []

    def add(name, x, y, wall, evals, note=""):
        tot, c = obj.evaluate(x, y, components=True)
        d = E.diversity(inst, x, y, rng=rng)
        r = {"method": name, "quality_mean": float(tot.mean()),
             "quality_std": float(tot.std()), "quality_max": float(tot.max()),
             "diversity": d, "feasible_rate": float(np.mean(c["overlap_area"] <= 1e-9)),
             "ms_per_sample": wall / K * 1000, "evals_per_sample": evals / K,
             "note": note}
        rows.append(r)
        print(f"{name:42s} J={r['quality_mean']:7.1f}+-{r['quality_std']:5.1f} "
              f"div={d:.3f} feas={r['feasible_rate']:4.2f} "
              f"{r['ms_per_sample']:7.1f} ms/sample  {r['evals_per_sample']:8.0f} ev")

    # ---- the deterministic reference points (diversity 0 by construction) ---
    for label, rule in (("greedy constructive (area order, det.)", "area"),
                        ("greedy constructive (room+position, det.)", "greedy")):
        t = time.time()
        r = gc.solve(rng=np.random.default_rng(a.seed), room_rule=rule)
        add(label, r["x"], r["y"], time.time() - t, r["evals"],
            "deterministic: one layout per brief")

    # ---- the policy's own diversity mechanism, without the policy ----------
    t = time.time()
    r = gc.solve(rng=np.random.default_rng(a.seed + 1), room_rule="random")
    add("greedy constructive (random order)", r["x"], r["y"], time.time() - t,
        r["evals"], "the Track E mechanism with a greedy rule instead of a policy")

    # ---- score jitter as a second, independent source of variety -----------
    for s in (2.0, 5.0, 10.0, 20.0, 40.0):
        t = time.time()
        r = gc.solve(rng=np.random.default_rng(a.seed + 2), jitter=s,
                     room_rule="area")
        add(f"greedy constructive (area, jitter {s:g})", r["x"], r["y"],
            time.time() - t, r["evals"])

    # ---- both at once ------------------------------------------------------
    for s in (5.0, 20.0):
        t = time.time()
        r = gc.solve(rng=np.random.default_rng(a.seed + 3), jitter=s,
                     room_rule="random")
        add(f"greedy constructive (random order, jitter {s:g})", r["x"], r["y"],
            time.time() - t, r["evals"])

    if os.path.exists(a.out):
        d = json.load(open(a.out))
        keep = [r for r in d["rows"] if not r["method"].startswith("greedy constructive")]
        d["rows"] = keep + rows
        json.dump(d, open(a.out, "w"), indent=1)
        print(f"\nmerged {len(rows)} rows into {a.out}")

        print("\n--- quality/diversity front (all methods) ---")
        for r in sorted(d["rows"], key=lambda r: -r["diversity"]):
            print(f"  div={r['diversity']:.3f}  J={r['quality_mean']:7.1f}  "
                  f"feas={r['feasible_rate']:4.2f}  {r['method']}")


if __name__ == "__main__":
    main()
