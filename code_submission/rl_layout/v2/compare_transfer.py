"""
Track D: does the learned policy transfer to briefs it has never seen?

This is the only regime in which learning has a structural advantage over a
metaheuristic.  A solver starts from scratch on every new brief; a policy trained
over a distribution of briefs answers a new one in a single forward pass.  If
that advantage does not show up here it does not exist, so the test is run on
three progressively harder held-out sets:

  1. fresh instances drawn from the same generator (in-distribution),
  2. `comb_high` -- the legacy 32-room office scenario,
  3. `hospital`  -- the legacy 32-room clinic, which additionally has
     different objective constants and a much tighter 68% fill ratio.

2 and 3 were authored by hand, not sampled from the generator, and the policy has
never optimized either of them.  Against them the classical solvers are run at
full budget, from scratch, exactly as they would be in practice.
"""

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                              # noqa: E402

import baselines2 as B2                                        # noqa: E402
import core                                                    # noqa: E402
import evaluate2 as E                                          # noqa: E402


def test_sets(n_inst, n_rooms, canvas, seed, flip_adj=True):
    """(name, InstanceBatch, x0, y0) for each held-out set."""
    out = []
    rng = np.random.default_rng(seed)
    gen = core.generate_instances(n_inst, n_rooms, rng, max_grid=canvas,
                                  jitter_weights=False)
    out.append(("generated (in-distribution)", gen))
    for s in ("comb_high", "hospital"):
        spec = build_spec(s, flip_adj=flip_adj)
        out.append((f"{s} (legacy, unseen)", core.from_spec(spec, n_inst)))
    res = []
    for name, inst in out:
        obj = core.BatchObjective(inst)
        r2 = np.random.default_rng(seed + 3)
        x0, y0 = core.random_layout(obj, r2)
        res.append((name, inst, obj, x0, y0))
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--construct", default="runs/D1_constr_random.pt")
    p.add_argument("--instances", type=int, default=64)
    p.add_argument("--n-rooms", dest="n_rooms", type=int, default=32)
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--seed", type=int, default=7777)
    p.add_argument("--sa-budgets", dest="sa_budgets", nargs="+", type=int,
                   default=[512, 2048, 8192, 45000])
    p.add_argument("--out", default="transfer.json")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N = a.instances
    all_rows = []

    for name, inst, obj, x0, y0 in test_sets(N, a.n_rooms, a.canvas, a.seed):
        ceil = obj.adj_ceiling + obj.edge_ceiling + inst.weights.no_overlap_bonus
        print(f"\n=== {name} ===  {N} instances, fill "
              f"{inst.fill_ratio().min():.2f}-{inst.fill_ratio().max():.2f}")
        rows = []

        def add(r, extra=None):
            r["test_set"] = name
            rows.append(r)
            all_rows.append(r)
            E.print_row(r)

        rng = np.random.default_rng(a.seed + 1)
        xs, ys = B2.shelf_pack(obj, rng)
        add(E.summarize("shelf packing", obj, xs, ys, N, 0.0, N))

        for steps in a.sa_budgets:
            r = B2.simulated_annealing(obj, x0, y0, steps, greedy_every=max(1, steps // 9),
                                       seed=a.seed, grid=True)
            add(E.summarize(f"SA {steps} lattice (from scratch)", obj, r["x"], r["y"],
                            r["evals"], r["wall_time"], N))

        if os.path.exists(a.construct):
            model, nrm, env, cargs = E.load_construct(a.construct, inst, dev,
                                                      canvas=a.canvas)
            r = E.run_construct(model, nrm, env, dev, deterministic=True)
            add(E.summarize("RL-construct zero-shot greedy", obj, r["x"], r["y"],
                            r["evals"], r["wall_time"], N))
            runs = [E.run_construct(model, nrm, env, dev) for _ in range(16)]
            for k in (1, 4, 16):
                bk = E.best_of_k(runs[:k])
                add(E.summarize(f"RL-construct zero-shot x{k}", obj, bk["x"], bk["y"],
                                bk["evals"], bk["wall_time"], N))
            bk = E.best_of_k(runs)
            px, py, _, ev = B2.greedy_polish(obj, bk["x"].copy(), bk["y"].copy(), 40, 1.0)
            add(E.summarize("RL-construct zero-shot x16 + polish", obj, px, py,
                            bk["evals"] + ev, bk["wall_time"], N))

        # how much annealing would be needed to match the policy?
        rl = [r for r in rows if r["method"].startswith("RL-construct zero-shot x16")]
        sa = [r for r in rows if r["method"].startswith("SA ")]
        if rl and sa:
            target = max(r["mean"] for r in rl)
            beat = [r for r in sa if r["mean"] >= target]
            if beat:
                cheapest = min(beat, key=lambda r: r["evals_per_inst"])
                print(f"  -> annealing needs >= {cheapest['evals_per_inst']:.0f} evals/layout "
                      f"({cheapest['method']}) to match RL's {target:.1f}")
            else:
                print(f"  -> no annealing budget tested reaches RL's {target:.1f} "
                      f"(best SA {max(r['mean'] for r in sa):.1f} at "
                      f"{max(r['evals_per_inst'] for r in sa):.0f} evals/layout)")

    with open(a.out, "w") as f:
        json.dump({"rows": all_rows}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
