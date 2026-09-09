"""
Run every method once per benchmark instance and record per-instance results.

The unit of analysis is the *instance*, not the layout sample.  Everything is
written out per instance so that methods can be compared by paired differences
on identical problems, which is the only comparison that supports a significance
statement (see `bench_stats.py`).

Methods are grouped by what they need:
  classical search      lattice simulated annealing at matched budget,
                        shelf packing as a no-search floor;
  masked constructive   greedy (two orders), regret-k, beam search -- all
                        training-free, all inside the policy's action set;
  learned               the constructive policy, evaluated zero-shot.

Beam search is the expensive one: its cost grows with n^2 x W x H x width and it
runs one instance at a time, so its widths are configurable and it can be skipped
for the largest cell.  Whatever is skipped is recorded as skipped rather than
silently omitted.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

import bench
import baselines2 as B2
import core
from constructive_search import BeamConstructor, regret_construct
from greedy_construct import GreedyConstructor

CELLS_N = [8, 16, 32, 64]
CELLS_F = [0.40, 0.60, 0.75]


def per_instance(obj, x, y):
    tot, c = obj.evaluate(x, y, components=True)
    return tot, (c["overlap_area"] <= 1e-9)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--rooms", nargs="+", type=int, default=CELLS_N)
    p.add_argument("--fills", nargs="+", type=float, default=CELLS_F)
    p.add_argument("--beam-widths", dest="beam_widths", nargs="+", type=int,
                   default=[4, 16])
    p.add_argument("--beam-max-n", dest="beam_max_n", type=int, default=64)
    p.add_argument("--regret-k", dest="regret_k", nargs="+", type=int, default=[2, 4, 8])
    p.add_argument("--sa-steps", dest="sa_steps", nargs="+", type=int,
                   default=[8192, 45000])
    p.add_argument("--policy", default="runs/D1_constr_random.pt")
    p.add_argument("--out", default="bench_test.json")
    a = p.parse_args()

    records = []
    for n in a.rooms:
        for f in a.fills:
            cell = bench.suite_cell(a.split, n, f, a.instances)
            inst = cell["inst"]
            obj = core.BatchObjective(inst)
            wit = bench.verify_witness(cell)
            assert wit["max_overlap"] <= 1e-9 and wit["all_inside"], \
                f"infeasible witness in cell n={n} fill={f}"
            print(f"\n=== n={n} fill={f} ({a.split}, {a.instances} instances, "
                  f"plate {inst.gw[0]:.0f}x{inst.gh[0]:.0f}) ===", flush=True)

            def add(method, x, y, obj_calls, cands, wall, note=""):
                tot, feas = per_instance(obj, x, y)
                for i in range(inst.B):
                    records.append({
                        "split": a.split, "n_rooms": n, "fill": f, "instance": i,
                        "method": method, "J": float(tot[i]),
                        "feasible": bool(feas[i]),
                        "obj_calls": float(obj_calls), "candidates": float(cands),
                        "ms_per_layout": float(wall / inst.B * 1000), "note": note})
                print(f"  {method:28s} J={tot.mean():9.2f} feas={feas.mean():4.2f} "
                      f"{wall / inst.B * 1000:8.1f} ms/inst", flush=True)

            rng = np.random.default_rng(cell["seed"] + 1)

            xs, ys = B2.shelf_pack(obj, rng)
            add("shelf packing", xs, ys, 1, 0, 0.0)

            x0, y0 = core.random_layout(obj, np.random.default_rng(cell["seed"] + 2))
            for steps in a.sa_steps:
                t = time.time()
                r = B2.simulated_annealing(obj, x0, y0, steps,
                                           greedy_every=max(1, steps // 9),
                                           seed=cell["seed"] + 3, grid=True)
                add(f"SA {steps} lattice", r["x"], r["y"], r["evals"] / inst.B, 0,
                    time.time() - t)

            gc = GreedyConstructor(inst)
            for label, rule in (("greedy, area order", "area"),
                                ("greedy, room+position", "greedy")):
                t = time.time()
                r = gc.solve(rng=np.random.default_rng(cell["seed"] + 4),
                             room_rule=rule)
                add(label, r["x"], r["y"], 1, r["evals"] / inst.B, time.time() - t)

            for k in a.regret_k:
                r = regret_construct(inst, k=k)
                add(f"regret-{k} insertion", r["x"], r["y"], 1,
                    r["candidates"] / inst.B, r["wall_time"])

            for Wd in a.beam_widths:
                if n > a.beam_max_n:
                    print(f"  beam width {Wd}: SKIPPED at n={n}", flush=True)
                    continue
                t = time.time()
                X = np.zeros((inst.B, n))
                Y = np.zeros((inst.B, n))
                cands = 0
                for i in range(inst.B):
                    one = inst.take(np.array([i]))
                    r = BeamConstructor(one, width=Wd).solve()
                    X[i], Y[i] = r["x"][0], r["y"][0]
                    cands += r["candidates"]
                add(f"beam width {Wd}", X, Y, Wd, cands / inst.B, time.time() - t)

            with open(a.out, "w") as fh:      # checkpoint after every cell
                json.dump({"split": a.split, "instances": a.instances,
                           "records": records}, fh)

    print(f"\nwrote {a.out}  ({len(records)} records)")


if __name__ == "__main__":
    main()
