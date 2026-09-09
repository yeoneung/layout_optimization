"""
Add the training-free masked greedy constructor to the transfer comparison.

`compare_transfer.py` puts the zero-shot policy against shelf packing and
annealing-from-scratch, but annealing is not the real competitor for "answer an
unseen brief immediately".  The real competitor is the constructive heuristic
that needs no training at all and shares the policy's masked action set.  Any
claim about amortization has to survive that row, so it belongs in the same
table and on the same instances.

The test sets are rebuilt from `compare_transfer.test_sets` with the identical
seed, so the rows merge into `transfer.json` without rerunning anything else.
"""

import argparse
import json
import os

import numpy as np

import evaluate2 as E
from compare_transfer import test_sets
from greedy_construct import GreedyConstructor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--instances", type=int, default=64)
    p.add_argument("--n-rooms", dest="n_rooms", type=int, default=32)
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--seed", type=int, default=7777)
    p.add_argument("--transfer", default="transfer.json")
    a = p.parse_args()

    N = a.instances
    new_rows = []

    for name, inst, obj, x0, y0 in test_sets(N, a.n_rooms, a.canvas, a.seed):
        print(f"\n=== {name} ===")
        gc = GreedyConstructor(inst, canvas=a.canvas)

        for label, rule in (("greedy constructive (area order)", "area"),
                            ("greedy constructive (room+position)", "greedy")):
            r = gc.solve(rng=np.random.default_rng(a.seed + 5), room_rule=rule)
            row = E.summarize(label, obj, r["x"], r["y"], r["evals"],
                              r["wall_time"], N)
            row["test_set"] = name
            new_rows.append(row)
            E.print_row(row)

    if os.path.exists(a.transfer):
        d = json.load(open(a.transfer))
        keep = [r for r in d["rows"] if not r["method"].startswith("greedy constructive")]
        # interleave: each test set keeps its own block order
        merged = []
        for name in dict.fromkeys(r["test_set"] for r in keep):
            merged += [r for r in keep if r["test_set"] == name]
            merged += [r for r in new_rows if r["test_set"] == name]
        d["rows"] = merged
        json.dump(d, open(a.transfer, "w"), indent=1)
        print(f"\nmerged {len(new_rows)} rows into {a.transfer}")


if __name__ == "__main__":
    main()
