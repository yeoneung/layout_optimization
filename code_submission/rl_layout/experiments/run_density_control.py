"""
The ordering control for the density experiment.

The density sweep shows that at high fill ratio the training-free procedures
which choose *which* facility to commit by immediate objective gain -- joint
greedy, free-order beam search -- lose feasibility rapidly, while the simple
largest-first rule does not.  That makes the commit order, not the position
choice, the variable that governs feasibility under density.

Any claim that a learned policy is more robust at density therefore has to be
measured against the best heuristic *ordering*, not merely against the greedy
one.  This runs largest-first order with beam search over positions, which is the
strongest training-free combination we know of for this regime: the classical
packing order, with search spent where it does not endanger feasibility.
"""

import argparse
import json
import time

import numpy as np

import bench
import core
import evaluate2 as E
from constructive_search import BeamConstructor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--rooms", type=int, default=32)
    p.add_argument("--fills", nargs="+", type=float,
                   default=[0.75, 0.80, 0.85, 0.90, 0.95])
    p.add_argument("--widths", nargs="+", type=int, default=[4, 16])
    p.add_argument("--bench", default="bench_density.json")
    a = p.parse_args()

    with open(a.bench) as f:
        d = json.load(f)
    label_prefix = "beam, area order, width"
    records = [r for r in d["records"] if not r["method"].startswith(label_prefix)]

    for fl in a.fills:
        cell = bench.suite_cell(a.split, a.rooms, fl, a.instances)
        inst = cell["inst"]
        obj = core.BatchObjective(inst)
        for W in a.widths:
            t = time.time()
            X = np.zeros((inst.B, a.rooms))
            Y = np.zeros((inst.B, a.rooms))
            cands = 0
            for i in range(inst.B):
                r = BeamConstructor(inst.take(np.array([i])), width=W,
                                    order="area").solve()
                X[i], Y[i] = r["x"][0], r["y"][0]
                cands += r["candidates"]
            wall = time.time() - t
            tot, c = obj.evaluate(X, Y, components=True)
            feas = c["overlap_area"] <= 1e-9
            m = f"{label_prefix} {W}"
            for i in range(inst.B):
                records.append({
                    "split": a.split, "n_rooms": a.rooms, "fill": fl,
                    "instance": i, "method": m, "J": float(tot[i]),
                    "feasible": bool(feas[i]), "obj_calls": float(W),
                    "candidates": float(cands / inst.B),
                    "ms_per_layout": float(wall / inst.B * 1000), "note": ""})
            print(f"fill={fl:.2f}  {m:28s} J={tot.mean():9.2f} "
                  f"feas={feas.mean():4.2f}  {wall / inst.B * 1000:8.1f} ms",
                  flush=True)

    d["records"] = records
    with open(a.bench, "w") as f:
        json.dump(d, f)
    print(f"\nwrote {a.bench}  ({len(records)} records)")


if __name__ == "__main__":
    main()
