"""
Head-to-head on a fixed reference scenario, as an anytime comparison.

Every method produces `--instances` layouts for the same brief and is charged for
the objective evaluations and the seconds it used.  Learned methods additionally
report their one-off training cost, which is amortized over the number of
layouts requested -- the crossover point is the honest summary of when learning
pays for itself and is reported explicitly rather than argued for.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                              # noqa: E402

import baselines2 as B2                                        # noqa: E402
import core                                                    # noqa: E402
import evaluate2 as E                                          # noqa: E402


def heldout(spec, n_inst, seed):
    rng = np.random.default_rng(seed)
    n = len(spec.room_order)
    X = rng.uniform(spec.w / 2.0, spec.grid_w - spec.w / 2.0, size=(n_inst, n))
    Y = rng.uniform(spec.h / 2.0, spec.grid_h - spec.h / 2.0, size=(n_inst, n))
    return X, Y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="comb_high")
    p.add_argument("--instances", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--flip-adj", dest="flip_adj", action="store_true", default=True)
    p.add_argument("--no-flip-adj", dest="flip_adj", action="store_false")
    p.add_argument("--construct", nargs="*", default=[])
    p.add_argument("--improve", nargs="*", default=[])
    p.add_argument("--skip-classical", action="store_true")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = build_spec(a.scenario, flip_adj=a.flip_adj)
    N = a.instances
    inst = core.from_spec(spec, N)
    obj = core.BatchObjective(inst)
    X0, Y0 = heldout(spec, N, a.seed)
    rng = np.random.default_rng(a.seed + 5)
    rows = []

    print(f"=== {a.scenario} (flip_adj={a.flip_adj}), {N} layouts requested ===")
    print(f"ceiling adj {obj.adj_ceiling[0]:.0f} + edge {obj.edge_ceiling[0]:.0f} "
          f"+ bonus {spec.no_overlap_bonus:.0f} = {obj.adj_ceiling[0]+obj.edge_ceiling[0]+spec.no_overlap_bonus:.0f}\n")

    def add(r):
        rows.append(r)
        E.print_row(r)

    # ---------------- classical -------------------------------------------
    if not a.skip_classical:
        add(E.summarize("random init", obj, X0, Y0, N, 0.0, N))
        xs, ys = B2.shelf_pack(obj, rng)
        add(E.summarize("shelf packing (no search)", obj, xs, ys, N, 0.0, N))
        for steps in (512, 2048, 8192, 45000):
            r = B2.simulated_annealing(obj, X0, Y0, steps,
                                       greedy_every=max(1, steps // 9), seed=a.seed)
            add(E.summarize(f"SA {steps} continuous", obj, r["x"], r["y"],
                            r["evals"], r["wall_time"], N))
        for steps in (512, 2048, 8192, 45000):
            r = B2.simulated_annealing(obj, X0, Y0, steps,
                                       greedy_every=max(1, steps // 9), seed=a.seed, grid=True)
            add(E.summarize(f"SA {steps} lattice", obj, r["x"], r["y"],
                            r["evals"], r["wall_time"], N))

    # ---------------- constructive RL --------------------------------------
    for path in a.construct:
        tag = os.path.splitext(os.path.basename(path))[0]
        model, nrm, env, cargs = E.load_construct(path, inst, dev)
        r = E.run_construct(model, nrm, env, dev, deterministic=True)
        add(E.summarize(f"RL-construct {tag} greedy", obj, r["x"], r["y"],
                        r["evals"], r["wall_time"], N))
        runs = []
        for k in range(16):
            runs.append(E.run_construct(model, nrm, env, dev, deterministic=False))
            if k + 1 in (1, 4, 16):
                bk = E.best_of_k(runs)
                add(E.summarize(f"RL-construct {tag} sample x{k+1}", obj, bk["x"], bk["y"],
                                bk["evals"], bk["wall_time"], N))
        # offer the learned solution the same greedy polish SA gets
        bk = E.best_of_k(runs)
        px, py, _, ev = B2.greedy_polish(obj, bk["x"].copy(), bk["y"].copy(), 40,
                                         spec.step_size)
        add(E.summarize(f"RL-construct {tag} x16 + polish", obj, px, py,
                        bk["evals"] + ev, bk["wall_time"], N))

        # RL as an initializer for the annealer.  The honest control is the same
        # annealer started from the best classical constructive heuristic, so the
        # comparison is "which warm start is better", not "warm start vs cold".
        if not a.skip_classical:
            xs2, ys2 = B2.shelf_pack(obj, rng, order_by="area")
            for steps in (512, 4096):
                rs = B2.simulated_annealing(obj, xs2, ys2, steps,
                                            greedy_every=max(1, steps // 9),
                                            seed=a.seed, grid=True)
                add(E.summarize(f"SA {steps} lattice from shelf", obj, rs["x"], rs["y"],
                                rs["evals"] + N, rs["wall_time"], N))
                rr = B2.simulated_annealing(obj, bk["x"].copy(), bk["y"].copy(), steps,
                                            greedy_every=max(1, steps // 9),
                                            seed=a.seed, grid=True)
                add(E.summarize(f"SA {steps} lattice from RL-construct", obj,
                                rr["x"], rr["y"], rr["evals"] + bk["evals"],
                                rr["wall_time"] + bk["wall_time"], N))

    # ---------------- improvement RL ---------------------------------------
    for path in a.improve:
        tag = os.path.splitext(os.path.basename(path))[0]
        model, nrm, env, iargs = E.load_improve(path, inst, dev)
        for H in (512, 2048, 4096):
            r = E.run_improve(model, nrm, env, dev, H, X0, Y0, deterministic=False)
            add(E.summarize(f"RL-improve {tag} H{H} stoch", obj, r["x"], r["y"],
                            r["evals"], r["wall_time"], N))
        r = E.run_improve(model, nrm, env, dev, 4096, X0, Y0, deterministic=True)
        add(E.summarize(f"RL-improve {tag} H4096 det", obj, r["x"], r["y"],
                        r["evals"], r["wall_time"], N))

    rows.sort(key=lambda r: -r["mean"])
    print("\n--- ranked by objective ---")
    for r in rows[:14]:
        print(f"  {r['mean']:8.1f}  feas={r['feasible_rate']:4.2f}  "
              f"{r['evals_per_inst']:9.0f} ev/layout  {r['method']}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"scenario": a.scenario, "flip_adj": a.flip_adj,
                       "instances": N, "rows": rows}, f, indent=1)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
