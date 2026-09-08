"""
Run the training-free methods on the same instances the CP-SAT model solves.

`export_exact.py` freezes small-plate n=8 cells (tag "exact") as JSON specs and
`exact_cpsat.py` proves optima for them in the ortools environment.  This runner
evaluates the heuristic suite on those identical instances in the main
environment, so that `exact_report.py` can print per-instance optimality gaps
--- the number the benchmark tables cannot provide on their own.

The methods and settings are the ones already validated for the main suite;
nothing is tuned on these cells.
"""

import argparse
import json
import time

import numpy as np

import bench
import baselines2 as B2
import core
from constructive_search import BeamConstructor, regret_construct
from greedy_construct import GreedyConstructor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--rooms", type=int, default=8)
    p.add_argument("--fills", nargs="+", type=float, default=[0.40, 0.60, 0.75])
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--tag", default="exact")
    p.add_argument("--avg-area", dest="avg_area", nargs=2, type=float,
                   default=[6.0, 10.0])
    p.add_argument("--beam-width", dest="beam_width", type=int, default=32)
    p.add_argument("--regret-k", dest="regret_k", type=int, default=4)
    p.add_argument("--sa-steps", dest="sa_steps", type=int, default=45000)
    p.add_argument("--tabu-tenure", dest="tenure", type=int, default=16)
    p.add_argument("--out", default="exact_cells_heur.json")
    a = p.parse_args()

    records = []
    for f in a.fills:
        cell = bench.suite_cell(a.split, a.rooms, f, a.instances,
                                tag=a.tag, avg_area=tuple(a.avg_area))
        inst = cell["inst"]
        obj = core.BatchObjective(inst)
        n = a.rooms
        print(f"\n=== n={n} fill={f} tag={a.tag} ({a.instances} inst) ===",
              flush=True)

        def add(method, x, y, wall):
            tot, c = obj.evaluate(x, y, components=True)
            feas = c["overlap_area"] <= 1e-9
            for i in range(inst.B):
                records.append({"split": a.split, "n_rooms": n, "fill": f,
                                "instance": i, "method": method,
                                "J": float(tot[i]), "feasible": bool(feas[i]),
                                "ms_per_layout": float(wall / inst.B * 1000)})
            print(f"  {method:26s} J={tot.mean():8.2f} feas={feas.mean():4.2f}",
                  flush=True)

        rng = np.random.default_rng(cell["seed"] + 1)

        gc = GreedyConstructor(inst)
        t = time.time()
        r = gc.solve(rng=np.random.default_rng(cell["seed"] + 4),
                     room_rule="area")
        add("largest-first greedy", r["x"], r["y"], time.time() - t)
        gx, gy = r["x"], r["y"]

        t = time.time()
        r = regret_construct(inst, k=a.regret_k)
        add(f"regret-{a.regret_k} insertion", r["x"], r["y"], time.time() - t)

        t = time.time()
        X = np.zeros((inst.B, n))
        Y = np.zeros((inst.B, n))
        for i in range(inst.B):
            one = inst.take(np.array([i]))
            rr = BeamConstructor(one, width=a.beam_width).solve()
            X[i], Y[i] = rr["x"][0], rr["y"][0]
        add(f"beam width {a.beam_width}", X, Y, time.time() - t)

        x0, y0 = core.random_layout(obj, np.random.default_rng(cell["seed"] + 2))
        t = time.time()
        r = B2.simulated_annealing(obj, x0, y0, a.sa_steps,
                                   greedy_every=max(1, a.sa_steps // 9),
                                   seed=cell["seed"] + 3, grid=True)
        add(f"SA {a.sa_steps} lattice", r["x"], r["y"], time.time() - t)

        t = time.time()
        r = B2.tabu(obj, gx, gy, budget=a.sa_steps * inst.B, tenure=a.tenure)
        add(f"tabu {a.sa_steps} greedy-start", r["x"], r["y"], time.time() - t)

        with open(a.out, "w") as fh:
            json.dump({"records": records}, fh)

    print(f"\nwrote {a.out} ({len(records)} records)")


if __name__ == "__main__":
    main()
