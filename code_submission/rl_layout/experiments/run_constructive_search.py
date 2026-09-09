"""
The non-myopic training-free control (ablation row B4), swept.

The question this answers is the one an operations-research reader asks as soon
as the masked action set is introduced: if the mask is what matters, does a
standard search inside the mask -- no learning -- already reach what the policy
reaches?  Greedy answers it for one-step optimality only.  Here the same state
space is searched with beam search over insertion decisions and with regret-k
insertion ordering.

Two deliberate choices make this control generous to the baseline rather than to
the policy:

  * every width and every k is reported, and the paper quotes the best one *per
    scenario* -- that is selection on the test instance itself, which no learned
    method is allowed here and which can only flatter the baseline;
  * cost is reported in candidate scores as well as full objective calls, so the
    baseline is not charged the policy's accounting.
"""

import argparse
import json
import sys
import time

import numpy as np

import core
import evaluate2 as E
from constructive_search import BeamConstructor, regret_construct
from greedy_construct import GreedyConstructor

sys.path.insert(0, "..")
from layout_env import build_spec                                 # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenarios", nargs="+", default=["comb_high", "hospital"])
    p.add_argument("--widths", nargs="+", type=int,
                   default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--regret-k", dest="regret_k", nargs="+", type=int,
                   default=[2, 3, 4, 8])
    p.add_argument("--out", default="constr_search.json")
    a = p.parse_args()

    rows = []
    for scen in a.scenarios:
        spec = build_spec(scen, flip_adj=True)
        inst = core.from_spec(spec, 1)
        obj = core.BatchObjective(inst)
        print(f"\n=== {scen} ===", flush=True)

        def add(name, x, y, obj_calls, cands, wall):
            tot, c = obj.evaluate(x, y, components=True)
            r = {"scenario": scen, "method": name,
                 "mean": float(tot.mean()),
                 "feasible_rate": float(np.mean(c["overlap_area"] <= 1e-9)),
                 "adj": float(c["adj"].mean()), "edge": float(c["edge"].mean()),
                 "obj_calls": float(obj_calls), "candidates": float(cands),
                 "wall_s": float(wall), "ms_per_layout": float(wall * 1000)}
            rows.append(r)
            print(f"  {name:34s} J={r['mean']:9.2f} feas={r['feasible_rate']:4.2f} "
                  f"obj_calls={obj_calls:7.0f} cand={cands:11,.0f} "
                  f"{r['ms_per_layout']:9.1f} ms", flush=True)
            return r

        gc = GreedyConstructor(inst)
        for label, rule in (("greedy, area order", "area"),
                            ("greedy, room+position", "greedy")):
            t = time.time()
            r = gc.solve(rng=np.random.default_rng(0), room_rule=rule)
            add(label, r["x"], r["y"], 1, r["evals"], time.time() - t)

        for k in a.regret_k:
            r = regret_construct(inst, k=k)
            add(f"regret-{k} insertion", r["x"], r["y"], r["obj_calls"],
                r["candidates"], r["wall_time"])

        for W in a.widths:
            r = BeamConstructor(inst, width=W).solve()
            add(f"beam search, width {W}", r["x"], r["y"], r["obj_calls"],
                r["candidates"], r["wall_time"])

        best = max((r for r in rows if r["scenario"] == scen),
                   key=lambda r: r["mean"])
        print(f"  -> best training-free constructive: {best['method']} "
              f"at {best['mean']:.2f}", flush=True)

    with open(a.out, "w") as f:
        json.dump({"rows": rows}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
