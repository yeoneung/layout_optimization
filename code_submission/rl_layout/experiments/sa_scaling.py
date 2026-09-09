"""
How much annealing would it take to match the learned policy?

Reporting "RL beats SA at budget X" invites the obvious reply: give SA more
budget.  This script does exactly that -- lattice SA at increasing budgets, plus
multi-start variants -- and reports either the budget at which it catches the
policy or the fact that it did not within the range tested.  Either answer is
publishable; only leaving the question open is not.
"""

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                              # noqa: E402

import baselines2 as B2                                        # noqa: E402
import core                                                    # noqa: E402
import evaluate2 as E                                          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="comb_high")
    p.add_argument("--instances", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--target", type=float, required=True,
                   help="objective the learned policy reached")
    p.add_argument("--budgets", nargs="+", type=int,
                   default=[45000, 120000, 300000, 700000])
    p.add_argument("--warm-start", dest="warm", default=None,
                   help="npz with x,y to start every chain from")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    spec = build_spec(a.scenario, flip_adj=True)
    N = a.instances
    inst = core.from_spec(spec, N)
    obj = core.BatchObjective(inst)
    rng = np.random.default_rng(a.seed)
    n = len(spec.room_order)
    X0 = rng.uniform(spec.w / 2, spec.grid_w - spec.w / 2, size=(N, n))
    Y0 = rng.uniform(spec.h / 2, spec.grid_h - spec.h / 2, size=(N, n))
    if a.warm:
        d = np.load(a.warm)
        X0, Y0 = d["x"], d["y"]
        print(f"warm start from {a.warm}: J={obj.evaluate(X0, Y0).mean():.1f}")

    rows = []
    print(f"=== annealing scaling, target J = {a.target:.1f} ===")
    for steps in a.budgets:
        r = B2.simulated_annealing(obj, X0, Y0, steps, greedy_every=max(1, steps // 9),
                                   seed=a.seed, grid=True)
        row = E.summarize(f"SA {steps} lattice", obj, r["x"], r["y"], r["evals"],
                          r["wall_time"], N)
        rows.append(row)
        E.print_row(row)
        if row["mean"] >= a.target:
            print(f"  -> annealing reaches the policy's {a.target:.1f} at "
                  f"{row['evals_per_inst']:.0f} evaluations and "
                  f"{row['ms_per_layout']/1000:.2f} s per layout")
            break
    else:
        best = max(rows, key=lambda r: r["mean"])
        print(f"  -> annealing did not reach {a.target:.1f} within the range tested; "
              f"best was {best['mean']:.1f} at {best['evals_per_inst']:.0f} "
              f"evaluations per layout")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"target": a.target, "rows": rows}, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
