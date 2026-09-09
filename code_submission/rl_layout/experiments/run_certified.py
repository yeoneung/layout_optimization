"""
Evaluate certificate-guided and witness-certified policies across density.

Three things are recorded per instance so the paper's central claim can be
checked rather than asserted:

  * the guided heuristic's empirical feasibility and unsafe fallback count;
  * the certified decoder's selective coverage and verified feasibility;
  * objective and cost, kept separate for the two algorithms.

Results are merged into the density benchmark file so that every comparison is
paired on identical instances.
"""

import argparse
import json
import time

import numpy as np
import torch

import bench
import core
import evaluate2 as E
from certified_decode import run_certified_decode
from certified_policy import run_guided_heuristic
from viability import ViabilityConstructor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--rooms", type=int, default=32)
    p.add_argument("--fills", nargs="+", type=float,
                   default=[0.75, 0.80, 0.85, 0.90, 0.95])
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--policies", nargs="+",
                   default=["runs/S_dord_seed1.pt", "runs/S_dord_seed2.pt"])
    p.add_argument("--samples", type=int, default=8)
    p.add_argument("--n-cand", dest="n_cand", type=int, default=32)
    p.add_argument("--init-tries", dest="init_tries", type=int, default=1)
    p.add_argument("--step-tries", dest="step_tries", type=int, default=1)
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--also-guided-greedy", dest="also_greedy",
                   action="store_true", default=True,
                   help="also run the training-free certificate-guided control")
    p.add_argument("--no-guided-greedy", dest="also_greedy",
                   action="store_false")
    p.add_argument("--bench", default="bench_density.json")
    p.add_argument("--out", default="bench_certified.json",
                   help="output JSON (kept separate from the source benchmark)")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(a.bench) as f:
        d = json.load(f)
    drop = ("certified policy", "certificate-guided", "witness-certified",
            "greedy+cert")
    records = [r for r in d["records"] if not r["method"].startswith(drop)]

    for fl in a.fills:
        cell = bench.suite_cell(a.split, a.rooms, fl, a.instances)
        inst = cell["inst"]
        obj = core.BatchObjective(inst)

        def add(method, x, y, calls, wall, extra=""):
            tot, c = obj.evaluate(x, y, components=True)
            feas = c["overlap_area"] <= 1e-9
            for i in range(inst.B):
                records.append({
                    "split": a.split, "n_rooms": a.rooms, "fill": fl,
                    "instance": i, "method": method, "J": float(tot[i]),
                    "feasible": bool(feas[i]), "obj_calls": float(calls),
                    "candidates": 0.0,
                    "ms_per_layout": float(wall / inst.B * 1000), "note": extra})
            print(f"fill={fl:.2f}  {method:34s} J={tot.mean():9.2f} "
                  f"feas={feas.mean():4.2f} {wall / inst.B * 1000:8.1f} ms {extra}",
                  flush=True)

        def add_selective(method, result, calls, wall):
            """Record only returned certified solutions as solutions."""
            tot, c = obj.evaluate(result["x"], result["y"], components=True)
            geometric = c["overlap_area"] <= 1e-9
            returned = np.asarray(result["certified"], dtype=bool)
            if np.any(returned & ~geometric):
                raise RuntimeError("certified decoder returned an infeasible layout")
            for i in range(inst.B):
                records.append({
                    "split": a.split, "n_rooms": a.rooms, "fill": fl,
                    "instance": i, "method": method,
                    "J": float(tot[i]) if returned[i] else None,
                    "feasible": bool(returned[i] and geometric[i]),
                    "returned": bool(returned[i]),
                    "obj_calls": float(calls), "candidates": 0.0,
                    "ms_per_layout": float(wall / inst.B * 1000),
                    "note": (f"coverage={returned.mean():.3f} "
                             f"witness_used={result['witness_used']} "
                             f"rejects={result['rejects']}")})
            vals = tot[returned]
            mean_j = float(vals.mean()) if len(vals) else float("nan")
            print(f"fill={fl:.2f}  {method:34s} J|returned={mean_j:9.2f} "
                  f"coverage={returned.mean():4.2f} "
                  f"{wall / inst.B * 1000:8.1f} ms", flush=True)

        if a.also_greedy:
            t = time.time()
            X = np.zeros((inst.B, a.rooms))
            Y = np.zeros((inst.B, a.rooms))
            for b in range(inst.B):
                r = ViabilityConstructor(inst.take(np.array([b])),
                                         canvas=a.canvas, filter_mode="pack",
                                         n_cand=a.n_cand,
                                         room_rule="greedy").solve()
                X[b], Y[b] = r["x"][0], r["y"][0]
            add("certificate-guided greedy (no learning)",
                X, Y, 1, time.time() - t)

        for path in a.policies:
            tag = path.split("/")[-1].replace(".pt", "").replace("S_dord_", "")
            model, nrm, env, _ = E.load_construct(path, inst, dev,
                                                  canvas=a.canvas)
            # The policy is certified with first-fit, not with the rollout of the
            # base heuristic, even though the rollout certificate dominates for
            # the largest-first constructor.  The reason is the same one that
            # makes the rollout certificate good there: a certificate helps in
            # proportion to how well its witness matches the constructor's own
            # continuation, and the base heuristic's rollout is not the policy's
            # continuation.  Measured at fill 0.97 the rollout certificate gives
            # the policy 0.70 feasibility against first-fit's 0.80, at 17x the
            # cost.
            t = time.time()
            r1 = run_guided_heuristic(
                model, nrm, env, dev, inst, n_cand=a.n_cand,
                certificate="pack")
            w1 = time.time() - t
            add(f"certificate-guided heuristic {tag}",
                r1["x"], r1["y"], a.rooms, w1,
                f"rej={r1['rejects']} fb={r1['fallbacks']}")

            t = time.time()
            runs = [run_guided_heuristic(
                        model, nrm, env, dev, inst, n_cand=a.n_cand,
                        seed=s, certificate="pack")
                    for s in range(a.samples)]
            bk = E.best_of_k(runs)
            add(f"certificate-guided heuristic {tag} x{a.samples}",
                bk["x"], bk["y"],
                a.rooms * a.samples, time.time() - t)

            t = time.time()
            rc = run_certified_decode(
                model, nrm, env, dev, inst, n_cand=a.n_cand,
                deterministic=True, seed=0, init_tries=a.init_tries,
                step_tries=a.step_tries)
            add_selective(
                f"witness-certified policy {tag}", rc, a.rooms,
                time.time() - t)

    d["records"] = records
    with open(a.out, "w") as f:
        json.dump(d, f)
    print(f"\nwrote {a.out}  ({len(records)} records)")


if __name__ == "__main__":
    main()
