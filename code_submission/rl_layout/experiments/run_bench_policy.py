"""
Evaluate the constructive policy zero-shot on the benchmark suite.

This is the distribution-shift test the two case studies cannot provide.  The
policy was trained on generated 32-room briefs at fill 0.45-0.68; the suite spans
8 to 64 rooms at fill 0.40 to 0.75, so three shifts are exercised at once:

  size shift      n = 8, 16 and n = 64 are outside the training range entirely;
  density shift   fill 0.75 is above anything seen in training;
  geometry shift  plate aspect ratios are redrawn per instance.

The architecture supports variable n because the room encoder carries no
per-room-index embedding (Remark "No learned room identity" in the manuscript),
and variable plates because the position head is fully convolutional.  Whether
that *works* is exactly what is measured here, and a failure at n = 64 would be a
finding rather than a bug.

Records are appended to the same file `run_bench.py` writes, keyed identically by
(split, n_rooms, fill, instance), so `bench_stats.py` can pair them.
"""

import argparse
import json
import time

import numpy as np
import torch

import bench
import core
import evaluate2 as E


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--rooms", nargs="+", type=int, default=[8, 16, 32, 64])
    p.add_argument("--fills", nargs="+", type=float, default=[0.40, 0.60, 0.75])
    p.add_argument("--policy", default="runs/D1_constr_random.pt")
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--samples", type=int, default=16)
    p.add_argument("--label", default="constructive policy")
    p.add_argument("--bench", default="bench_test.json")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(a.bench) as f:
        d = json.load(f)
    records = [r for r in d["records"] if not r["method"].startswith(a.label)]

    for n in a.rooms:
        for fl in a.fills:
            cell = bench.suite_cell(a.split, n, fl, a.instances)
            inst = cell["inst"]
            obj = core.BatchObjective(inst)
            model, nrm, env, _ = E.load_construct(a.policy, inst, dev,
                                                  canvas=a.canvas)

            t = time.time()
            r1 = E.run_construct(model, nrm, env, dev, deterministic=True)
            t1 = time.time() - t

            t = time.time()
            runs = [E.run_construct(model, nrm, env, dev) for _ in range(a.samples)]
            bk = E.best_of_k(runs)
            tk = time.time() - t

            for label, res, calls, wall in (
                    (a.label, r1, n, t1),
                    (f"{a.label} x{a.samples}", bk, n * a.samples, tk)):
                tot, c = obj.evaluate(res["x"], res["y"], components=True)
                feas = c["overlap_area"] <= 1e-9
                for i in range(inst.B):
                    records.append({
                        "split": a.split, "n_rooms": n, "fill": fl, "instance": i,
                        "method": label, "J": float(tot[i]),
                        "feasible": bool(feas[i]), "obj_calls": float(calls),
                        "candidates": 0.0,
                        "ms_per_layout": float(wall / inst.B * 1000), "note": ""})
                print(f"n={n:3d} fill={fl:.2f}  {label:28s} J={tot.mean():9.2f} "
                      f"feas={feas.mean():4.2f}  {wall / inst.B * 1000:7.1f} ms",
                      flush=True)

    d["records"] = records
    with open(a.bench, "w") as f:
        json.dump(d, f)
    print(f"\nwrote {a.bench}  ({len(records)} records)")


if __name__ == "__main__":
    main()
