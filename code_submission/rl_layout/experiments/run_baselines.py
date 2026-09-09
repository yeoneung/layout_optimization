"""
Classical side of the comparison, on the same held-out instances legacy used.

First it reproduces the legacy SA number as a regression check, then it runs the
added baselines (lattice SA, tabu, GA, multi-start SA) so we know what RL
actually has to beat.  Everything is reported per objective evaluation, because
that is the unit of work every method shares.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                       # noqa: E402

import baselines2 as B2                                 # noqa: E402
import core                                             # noqa: E402


def heldout(spec, n_inst, seed):
    """Exactly the draw legacy's evaluate.py uses, so numbers stay comparable."""
    rng = np.random.default_rng(seed)
    n = len(spec.room_order)
    xmin, xmax = spec.w / 2.0, spec.grid_w - spec.w / 2.0
    ymin, ymax = spec.h / 2.0, spec.grid_h - spec.h / 2.0
    X = rng.uniform(xmin, xmax, size=(n_inst, n))
    Y = rng.uniform(ymin, ymax, size=(n_inst, n))
    return X, Y


def report(name, obj, x, y, evals, wall, rows, n_inst):
    tot, c = obj.evaluate(x, y, components=True)
    row = {"method": name,
           "mean": float(tot.mean()), "std": float(tot.std()),
           "median": float(np.median(tot)), "max": float(tot.max()),
           "feasible_rate": float(np.mean(c["overlap_area"] <= 1e-9)),
           "adj": float(c["adj"].mean()), "edge": float(c["edge"].mean()),
           "evals_per_inst": float(evals / n_inst), "wall_s": float(wall)}
    rows.append(row)
    print(f"{name:34s} J={row['mean']:8.1f}+-{row['std']:5.1f} med={row['median']:8.1f} "
          f"feas={row['feasible_rate']:4.2f} adj={row['adj']:7.1f} edge={row['edge']:5.1f} "
          f"ev/inst={row['evals_per_inst']:9.0f} {row['wall_s']:6.1f}s")
    return row


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="comb_high")
    p.add_argument("--instances", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--flip-adj", dest="flip_adj", action="store_true", default=True)
    p.add_argument("--no-flip-adj", dest="flip_adj", action="store_false")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    spec = build_spec(a.scenario, flip_adj=a.flip_adj)
    inst = core.from_spec(spec, a.instances)
    obj = core.BatchObjective(inst)
    X0, Y0 = heldout(spec, a.instances, a.seed)
    N = a.instances
    rng = np.random.default_rng(a.seed + 1)

    print(f"=== {a.scenario} (flip_adj={a.flip_adj}), {N} held-out random inits ===")
    print(f"ceilings: adj {obj.adj_ceiling[0]:.0f} + edge {obj.edge_ceiling[0]:.0f} "
          f"+ bonus {spec.no_overlap_bonus:.0f}")
    print(f"initial objective: {obj.evaluate(X0, Y0).mean():.1f}\n")
    rows = []

    report("random init", obj, X0, Y0, N, 0.0, rows, N)

    xs, ys = B2.shelf_pack(obj, rng, order_by="random")
    report("shelf packing (no search)", obj, xs, ys, N, 0.0, rows, N)
    xs2, ys2 = B2.shelf_pack(obj, rng, order_by="area")
    report("shelf packing (area order)", obj, xs2, ys2, N, 0.0, rows, N)

    gx, gy, gv, ev = B2.greedy_polish(obj, X0.copy(), Y0.copy(), 200, spec.step_size)
    report("greedy steepest ascent", obj, gx, gy, ev, 0.0, rows, N)

    hx, hy, hv, ev2 = B2.greedy_polish(obj, xs.copy(), ys.copy(), 200, spec.step_size)
    report("shelf + greedy", obj, hx, hy, ev2, 0.0, rows, N)

    # --- regression check against legacy's headline SA number -------------------
    V1_REPORTED = {"comb_high": (3270.0, 1.00), "hospital": (3565.0, 0.20)}
    r = B2.simulated_annealing(obj, X0, Y0, 45000, greedy_every=5000, seed=a.seed)
    row = report("SA 45k + polish (legacy setup)", obj, r["x"], r["y"], r["evals"],
                 r["wall_time"], rows, N)
    if a.scenario in V1_REPORTED and a.flip_adj:
        j, f = V1_REPORTED[a.scenario]
        print(f"    [regression] legacy reported J={j:.1f} at {f*100:.0f}% feasible for "
              f"this configuration; batched gives J={row['mean']:.1f} at "
              f"{row['feasible_rate']*100:.0f}%")

    # --- the added, stronger baselines --------------------------------------
    for steps in (4096, 15000, 45000):
        r = B2.simulated_annealing(obj, X0, Y0, steps, greedy_every=max(1, steps // 9),
                                   seed=a.seed, grid=True)
        report(f"SA {steps} lattice + polish", obj, r["x"], r["y"], r["evals"],
               r["wall_time"], rows, N)

    r = B2.sa_restarts(obj, inst, X0, Y0, n_restarts=8, n_steps=5625,
                       greedy_every=1875, seed=a.seed, grid=True)
    report("SA 8x5625 restarts (lattice)", obj, r["x"], r["y"], r["evals"],
           r["wall_time"], rows, N)

    r = B2.tabu(obj, X0, Y0, budget=45000 * N, step=spec.step_size)
    report("tabu search", obj, r["x"], r["y"], r["evals"], r["wall_time"], rows, N)

    r = B2.genetic(obj, inst, X0, Y0, pop=64, generations=700, seed=a.seed,
                   grid=True, local_search_every=0)
    report("GA (pop 64 x 700 gen)", obj, r["x"], r["y"], r["evals"], r["wall_time"], rows, N)

    r = B2.genetic(obj, inst, X0, Y0, pop=48, generations=120, seed=a.seed,
                   grid=True, local_search_every=4)
    report("memetic GA (GA + local search)", obj, r["x"], r["y"], r["evals"],
           r["wall_time"], rows, N)

    best = max(rows[1:], key=lambda r: r["mean"])
    print(f"\nstrongest classical baseline: {best['method']} at J={best['mean']:.1f} "
          f"({best['evals_per_inst']:.0f} evals/instance)")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"scenario": a.scenario, "flip_adj": a.flip_adj,
                       "instances": N, "rows": rows}, f, indent=1)


if __name__ == "__main__":
    main()
