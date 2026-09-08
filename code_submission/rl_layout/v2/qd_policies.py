"""
Quality and variety for several trained policies, without re-running the
annealing sweep (that lives in `run_qd.py` and its numbers are reused).

The three policies differ only in how they are allowed to be a distribution:
  B1  the policy chooses the commit order; entropy annealed to ~0
  B3  the environment draws the commit order; entropy annealed to ~0
  B4  the policy chooses the order; entropy held up instead of annealed
"""

import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                              # noqa: E402

import core                                                    # noqa: E402
import evaluate2 as E                                          # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="comb_high")
    p.add_argument("--ckpt", nargs="+", required=True)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--temps", nargs="+", type=float, default=[1.0, 1.5, 2.5, 4.0])
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--out", default="qd_policies.json")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = build_spec(a.scenario, flip_adj=True)
    K = a.samples
    inst = core.from_spec(spec, K)
    obj = core.BatchObjective(inst)
    rng = np.random.default_rng(a.seed)
    rows = []

    for path in a.ckpt:
        if not os.path.exists(path):
            print(f"(missing {path})")
            continue
        tag = os.path.splitext(os.path.basename(path))[0]
        model, nrm, env, cargs = E.load_construct(path, inst, dev, seed=a.seed)
        order = cargs.get("order", "policy")
        for T in a.temps:
            r = E.run_construct(model, nrm, env, dev, temperature=T)
            tot, c = obj.evaluate(r["x"], r["y"], components=True)
            d = E.diversity(inst, r["x"], r["y"], rng=rng)
            row = {"method": f"{tag} T={T}", "policy": tag, "order": order,
                   "temperature": T,
                   "quality_mean": float(tot.mean()), "quality_std": float(tot.std()),
                   "quality_max": float(tot.max()), "diversity": d,
                   "feasible_rate": float(np.mean(c["overlap_area"] <= 1e-9)),
                   "evals_per_sample": r["evals"] / K,
                   "ms_per_sample": r["wall_time"] / K * 1000}
            rows.append(row)
            print(f"{row['method']:34s} order={order:6s} J={row['quality_mean']:7.1f}"
                  f"+-{row['quality_std']:5.1f} max={row['quality_max']:7.1f} "
                  f"div={d:.3f} feas={row['feasible_rate']:4.2f} "
                  f"{row['ms_per_sample']:6.1f} ms")

    with open(a.out, "w") as f:
        json.dump({"scenario": a.scenario, "samples": K, "rows": rows}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
