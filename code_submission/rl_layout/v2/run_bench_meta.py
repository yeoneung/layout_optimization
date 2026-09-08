"""
Tabu search and a genetic algorithm on the generated benchmark.

Both metaheuristics already exist in `baselines2.py` and were reported on the two
legacy hand-authored scenarios; this runner puts them on the generated suite under the same
protocol as every other method: hyperparameters are selected on the validation
split, the test split is evaluated once with those settings frozen, and the
budget is stated in objective evaluations (45,000 per instance, matching SA).

Fairness notes, in the direction that hurts our own method:

  * Tabu is offered both a random lattice start and a feasible constructive
    start (largest-first greedy), because steepest-descent tabu cannot teleport
    and a random start at high fill would handicap it for a reason that is an
    artifact of initialization, not of tabu search.  The start is a validated
    hyperparameter like any other.
  * The GA runs on the lattice (`grid=True`) with optional memetic polishing of
    its elites, which is the strong FLP configuration.
  * Both report the evaluations they actually consumed, including polish steps,
    so a small overshoot of the nominal budget is visible rather than hidden.

Record schema matches `run_bench.py`, so `bench_report.py` reads the output
unchanged.
"""

import argparse
import json
import time

import numpy as np

import bench
import baselines2 as B2
import core
from greedy_construct import GreedyConstructor


def per_instance(obj, x, y):
    tot, c = obj.evaluate(x, y, components=True)
    return tot, (c["overlap_area"] <= 1e-9)


def tabu_once(obj, inst, cell, tenure, start, budget):
    B = inst.B
    if start == "greedy":
        gc = GreedyConstructor(inst)
        r0 = gc.solve(rng=np.random.default_rng(cell["seed"] + 4),
                      room_rule="area")
        x0, y0 = r0["x"], r0["y"]
    else:
        x0, y0 = core.random_layout(obj, np.random.default_rng(cell["seed"] + 2))
        x0, y0 = B2.snap_to_grid(obj, x0, y0)
    t = time.time()
    r = B2.tabu(obj, x0, y0, budget=budget * B, tenure=tenure)
    r["wall"] = time.time() - t
    return r


def ga_once(obj, inst, cell, pop, ls_every, budget):
    x0, y0 = core.random_layout(obj, np.random.default_rng(cell["seed"] + 2))
    t = time.time()
    r = B2.genetic(obj, inst, x0, y0, pop=pop, generations=max(1, budget // pop),
                   seed=cell["seed"] + 5, grid=True,
                   local_search_every=ls_every)
    r["wall"] = time.time() - t
    return r


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--rooms", nargs="+", type=int, default=[32])
    p.add_argument("--fills", nargs="+", type=float,
                   default=[0.40, 0.60, 0.75])
    p.add_argument("--budget", type=int, default=45000)
    p.add_argument("--tabu-tenures", dest="tenures", nargs="+", type=int,
                   default=[16])
    p.add_argument("--tabu-starts", dest="starts", nargs="+",
                   default=["greedy"])
    p.add_argument("--ga-pops", dest="pops", nargs="+", type=int, default=[32])
    p.add_argument("--ga-ls", dest="ls", nargs="+", type=int, default=[5])
    p.add_argument("--skip-ga", dest="skip_ga", action="store_true")
    p.add_argument("--out", default="bench_meta.json")
    a = p.parse_args()

    records = []
    for n in a.rooms:
        for f in a.fills:
            cell = bench.suite_cell(a.split, n, f, a.instances)
            inst = cell["inst"]
            obj = core.BatchObjective(inst)
            wit = bench.verify_witness(cell)
            assert wit["max_overlap"] <= 1e-9 and wit["all_inside"]
            print(f"\n=== n={n} fill={f} ({a.split}, {a.instances} inst) ===",
                  flush=True)

            def add(method, r, note=""):
                tot, feas = per_instance(obj, r["x"], r["y"])
                for i in range(inst.B):
                    records.append({
                        "split": a.split, "n_rooms": n, "fill": f,
                        "instance": i, "method": method, "J": float(tot[i]),
                        "feasible": bool(feas[i]),
                        "obj_calls": float(r["evals"] / inst.B),
                        "candidates": 0.0,
                        "ms_per_layout": float(r["wall"] / inst.B * 1000),
                        "note": note})
                print(f"  {method:34s} J={tot.mean():9.2f} "
                      f"feas={feas.mean():4.2f} "
                      f"evals/inst={r['evals'] / inst.B:8.0f} "
                      f"{r['wall'] / inst.B * 1000:8.0f} ms/inst", flush=True)

            for tenure in a.tenures:
                for start in a.starts:
                    r = tabu_once(obj, inst, cell, tenure, start, a.budget)
                    add(f"tabu {a.budget} t{tenure} {start}-start", r)

            if not a.skip_ga:
                for pop in a.pops:
                    for ls in a.ls:
                        r = ga_once(obj, inst, cell, pop, ls, a.budget)
                        add(f"GA {a.budget} pop{pop} ls{ls}", r)

            with open(a.out, "w") as fh:
                json.dump({"split": a.split, "instances": a.instances,
                           "records": records}, fh)

    print(f"\nwrote {a.out}  ({len(records)} records)")


if __name__ == "__main__":
    main()
