"""
The constructive baselines, separated out because they turned out to matter more
than anything else in the comparison.

Ordered from weakest to strongest:
  shelf packing              -- no objective awareness at all
  greedy position, area order-- objective-aware placement, fixed commit order
  greedy room + position     -- one-step-optimal in both decisions
  best-of-K with jitter      -- the fair counterpart of sampling a policy K times

Whatever the learned policy is worth, it is worth the margin over the strongest
row here, not the margin over annealing.
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
from greedy_construct import GreedyConstructor                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="comb_high")
    p.add_argument("--instances", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--k", type=int, default=16)
    p.add_argument("--jitter", type=float, default=1.5)
    p.add_argument("--out", default=None)
    a = p.parse_args()

    spec = build_spec(a.scenario, flip_adj=True)
    N = a.instances
    inst = core.from_spec(spec, N)
    obj = core.BatchObjective(inst)
    rng = np.random.default_rng(a.seed)
    rows = []
    print(f"=== constructive baselines, {a.scenario}, {N} layouts ===")

    def add(r):
        rows.append(r)
        E.print_row(r)

    for k, tag in ((1, ""), (a.k, f" (best of {a.k})")):
        xs, ys = B2.shelf_pack(obj, rng, order_by="random", n_tries=k)
        add(E.summarize(f"shelf packing{tag}", obj, xs, ys, N * k, 0.0, N))

    G = GreedyConstructor(inst)
    r = G.solve(rng=rng, room_rule="area")
    add(E.summarize("greedy position, area order", obj, r["x"], r["y"],
                    r["evals"], r["wall_time"], N))

    r = G.solve(rng=rng, room_rule="greedy")
    add(E.summarize("greedy room + position", obj, r["x"], r["y"],
                    r["evals"], r["wall_time"], N))
    det = r

    runs = []
    for k in range(a.k):
        runs.append(G.solve(rng=np.random.default_rng(a.seed + 100 + k),
                            jitter=a.jitter, room_rule="greedy"))
        if k + 1 in (4, a.k):
            bk = E.best_of_k(runs)
            add(E.summarize(f"greedy + jitter, best of {k+1}", obj, bk["x"], bk["y"],
                            bk["evals"], bk["wall_time"], N))

    bk = E.best_of_k(runs + [det])
    px, py, _, ev = B2.greedy_polish(obj, bk["x"].copy(), bk["y"].copy(), 40,
                                     spec.step_size)
    add(E.summarize(f"greedy constructive best-of-{a.k+1} + polish", obj, px, py,
                    bk["evals"] + ev, bk["wall_time"], N))

    rs = B2.simulated_annealing(obj, bk["x"].copy(), bk["y"].copy(), 4096,
                                greedy_every=455, seed=a.seed, grid=True)
    add(E.summarize("SA 4096 lattice from greedy constructive", obj, rs["x"], rs["y"],
                    rs["evals"] + bk["evals"], rs["wall_time"], N))

    best = max(rows, key=lambda r: r["mean"])
    print(f"\nstrongest constructive baseline: {best['method']}  J={best['mean']:.1f}")

    if a.out:
        with open(a.out, "w") as f:
            json.dump({"scenario": a.scenario, "instances": N, "rows": rows}, f, indent=1)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
