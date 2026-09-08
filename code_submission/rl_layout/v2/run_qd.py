"""
Track E: the diversity claim, measured instead of asserted.

v1 found the usual "RL gives you a distribution over layouts" claim inverted --
its stochastic PPO reached diversity 0.26-0.35 while SA re-run from the same
start with a new seed reached 0.64.  That comparison was between a mode-seeking
policy at a single temperature and an annealer swept over budgets, i.e. one
point against a curve.

Here both sides are swept.  The policy's sampling temperature and the annealer's
budget each trace a quality/diversity front, and the fronts are compared.  A
method wins only if its front dominates -- higher quality at equal variety, or
more variety at equal quality.
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="comb_high")
    p.add_argument("--flip-adj", dest="flip_adj", action="store_true", default=True)
    p.add_argument("--construct", default="runs/B1_constr_fixed.pt")
    p.add_argument("--improve", default=None)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="qd_comb_high.json")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = build_spec(a.scenario, flip_adj=a.flip_adj)
    K = a.samples
    inst = core.from_spec(spec, K)
    obj = core.BatchObjective(inst)
    rng = np.random.default_rng(a.seed)
    n = len(spec.room_order)

    # every SA chain starts from the SAME layout, so its variety comes from the
    # chain itself and not from the initialization -- v1's protocol, kept
    x1 = rng.uniform(spec.w / 2, spec.grid_w - spec.w / 2, size=(1, n))
    y1 = rng.uniform(spec.h / 2, spec.grid_h - spec.h / 2, size=(1, n))
    X0 = np.repeat(x1, K, axis=0)
    Y0 = np.repeat(y1, K, axis=0)
    print(f"sanity: diversity of {K} identical starts = "
          f"{E.diversity(inst, X0, Y0):.3f}  (must be 0)\n")

    rows = []

    def add(name, x, y, wall, evals, note=""):
        tot, c = obj.evaluate(x, y, components=True)
        d = E.diversity(inst, x, y, rng=rng)
        r = {"method": name, "quality_mean": float(tot.mean()),
             "quality_std": float(tot.std()), "quality_max": float(tot.max()),
             "diversity": d, "feasible_rate": float(np.mean(c["overlap_area"] <= 1e-9)),
             "ms_per_sample": wall / K * 1000, "evals_per_sample": evals / K,
             "note": note}
        rows.append(r)
        print(f"{name:36s} J={r['quality_mean']:7.1f}+-{r['quality_std']:5.1f} "
              f"div={d:.3f} feas={r['feasible_rate']:4.2f} "
              f"{r['ms_per_sample']:7.1f} ms/sample  {r['evals_per_sample']:8.0f} ev")
        return r

    # ---- learned constructive policy, swept over sampling temperature ------
    if a.construct and os.path.exists(a.construct):
        model, nrm, env, _ = E.load_construct(a.construct, inst, dev)
        r = E.run_construct(model, nrm, env, dev, deterministic=True)
        add("RL-construct greedy (det.)", r["x"], r["y"], r["wall_time"], r["evals"],
            "one layout per brief: a function, not a distribution")
        for T in (0.4, 0.7, 1.0, 1.5, 2.5, 4.0):
            r = E.run_construct(model, nrm, env, dev, temperature=T)
            add(f"RL-construct T={T}", r["x"], r["y"], r["wall_time"], r["evals"])

    # ---- improvement policy, if one was trained ---------------------------
    if a.improve and os.path.exists(a.improve):
        model, nrm, env, _ = E.load_improve(a.improve, inst, dev)
        for H in (512, 4096):
            r = E.run_improve(model, nrm, env, dev, H, X0, Y0, deterministic=False)
            add(f"RL-improve H{H} stoch", r["x"], r["y"], r["wall_time"], r["evals"])

    # ---- annealing, swept over budget ------------------------------------
    for steps in (512, 2048, 8192, 45000):
        for grid in (False, True):
            t = time.time()
            res = B2.simulated_annealing(obj, X0, Y0, steps, greedy_every=max(1, steps // 9),
                                         seed=a.seed + 7, grid=grid)
            add(f"SA {steps} {'lattice' if grid else 'cont'} (same start)",
                res["x"], res["y"], time.time() - t, res["evals"])

    rx = rng.uniform(spec.w / 2, spec.grid_w - spec.w / 2, size=(K, n))
    ry = rng.uniform(spec.h / 2, spec.grid_h - spec.h / 2, size=(K, n))
    t = time.time()
    res = B2.simulated_annealing(obj, rx, ry, 45000, greedy_every=5000, seed=a.seed + 11)
    add("SA 45000 (random restarts)", res["x"], res["y"], time.time() - t, res["evals"],
        "different starts, upper bound on SA variety")

    xs, ys = B2.shelf_pack(obj, rng)
    add("shelf packing (random order)", xs, ys, 0.0, K)

    print("\n--- quality/diversity front ---")
    for r in sorted(rows, key=lambda r: -r["diversity"]):
        print(f"  div={r['diversity']:.3f}  J={r['quality_mean']:7.1f}  "
              f"{r['ms_per_sample']:7.1f} ms  {r['method']}")

    with open(a.out, "w") as f:
        json.dump({"scenario": a.scenario, "samples": K, "rows": rows}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
