"""Confirmatory evaluation of guided and witness-certified decoding.

This script supplies the two experiments added for the v3 submission:

1. a complete density sweep for the actual witness-preserving decoder, on the
   same held-out cells used by the paper; and
2. a disjoint 100-instance confirmatory family for the two dense cells where the
   reliability claims matter most.

The output is intentionally separate from ``bench_density*.json``.  Every
method-level record contains per-instance objective, feasibility and return
flags, so coverage, conditional quality and paired comparisons can be recomputed
without parsing console output.  Abstentions have ``J=null`` and are never
silently treated as layouts.
"""

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from scipy.stats import beta

import bench
import core
import evaluate2 as E
from certified_decode import run_certified_decode
from certified_policy import run_guided_heuristic
from viability import ViabilityConstructor


POLICIES = {
    32: ["runs/S_dord_seed1.pt", "runs/S_dord_seed2.pt"],
    64: ["runs/S64_seed1.pt", "runs/S64_seed2.pt"],
}


def exact_ci(successes, trials, alpha=0.05):
    """Two-sided Clopper--Pearson interval."""
    if trials == 0:
        return [None, None]
    lo = 0.0 if successes == 0 else float(
        beta.ppf(alpha / 2.0, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(
        beta.ppf(1.0 - alpha / 2.0, successes + 1, trials - successes))
    return [lo, hi]


def policy_id(path, args):
    seed = args.get("seed")
    return f"seed{seed}" if seed is not None else Path(path).stem


def record_layout(suite, cell, method, policy, obj, x, y, wall,
                  returned=None, guaranteed=False, effort=None, extra=None):
    """Build a self-contained raw and summary record for one method/cell."""
    total, comp = obj.evaluate(x, y, components=True)
    geometric = np.asarray(comp["overlap_area"] <= 1e-9, dtype=bool)
    if returned is None:
        returned = np.ones(cell["inst"].B, dtype=bool)
    returned = np.asarray(returned, dtype=bool)
    if guaranteed and np.any(returned & ~geometric):
        bad = np.flatnonzero(returned & ~geometric).tolist()
        raise RuntimeError(f"certified return is infeasible at indices {bad}")

    valid = returned & geometric
    n = len(returned)
    n_ret = int(returned.sum())
    n_geom = int(geometric.sum())
    n_valid = int(valid.sum())
    values = [float(v) if returned[i] else None for i, v in enumerate(total)]
    rec = {
        "suite": suite,
        "split": cell["split"],
        "suite_tag": cell.get("suite_tag", ""),
        "bench_seed": int(cell["seed"]),
        "n_rooms": int(cell["n_rooms"]),
        "fill": float(cell["fill"]),
        "instances": n,
        "method": method,
        "policy": policy,
        "witness_effort": effort,
        "guaranteed": bool(guaranteed),
        "instance": list(range(n)),
        "J": values,
        "returned": returned.tolist(),
        "geometrically_feasible": geometric.tolist(),
        "summary": {
            "returned": n_ret,
            "coverage": float(n_ret / n),
            "coverage_ci95": exact_ci(n_ret, n),
            # For selective methods, padded abstention slots are not outputs;
            # their geometry is retained only as an implementation diagnostic.
            "geometrically_feasible_emissions": n_geom,
            "emitted_geometric_rate": float(n_geom / n),
            "valid_returns": n_valid,
            "valid_return_rate": float(n_valid / n),
            "return_feasibility": (float(n_valid / n_ret)
                                    if n_ret else None),
            "return_feasibility_ci95": exact_ci(n_valid, n_ret),
            "mean_J_emitted": (float(np.mean(total))
                               if returned.all() else None),
            "mean_J_valid": (float(np.mean(total[valid]))
                             if valid.any() else None),
            "ms_per_instance": float(wall / n * 1000.0),
        },
    }
    if extra:
        rec["summary"].update(extra)
    return rec


def record_key(rec):
    return (rec["suite"], rec["n_rooms"], rec["fill"], rec["method"],
            rec.get("policy"), rec.get("witness_effort"))


def save_atomic(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1, allow_nan=False)
    os.replace(tmp, path)


def print_record(rec):
    s = rec["summary"]
    who = f" {rec['policy']}" if rec.get("policy") else ""
    effort = (f" effort={rec['witness_effort']}"
              if rec.get("witness_effort") is not None else "")
    j = s["mean_J_valid"]
    jtxt = "nan" if j is None else f"{j:.1f}"
    reliability = (s["return_feasibility"] if rec["guaranteed"]
                   else s["emitted_geometric_rate"])
    rlabel = "return-feas" if rec["guaranteed"] else "feas"
    rtxt = "n/a" if reliability is None else f"{reliability:.3f}"
    print(f"{rec['suite']:8s} n={rec['n_rooms']:2d} f={rec['fill']:.2f} "
          f"{rec['method']}{who}{effort}: "
          f"coverage={s['coverage']:.3f} {rlabel}={rtxt} "
          f"J|valid={jtxt} {s['ms_per_instance']:.1f} ms", flush=True)


def viability_layout(inst, mode, n_cand, canvas):
    x = np.zeros((inst.B, inst.n))
    y = np.zeros((inst.B, inst.n))
    rejects = 0
    fallbacks = 0
    t0 = time.time()
    for b in range(inst.B):
        r = ViabilityConstructor(
            inst.take(np.array([b])), canvas=canvas, filter_mode=mode,
            n_cand=n_cand, room_rule="area", seed=0).solve()
        x[b], y[b] = r["x"][0], r["y"][0]
        rejects += int(r["rejects"])
        # ViabilityConstructor exposes stranding through overlap rather than an
        # explicit fallback counter.  Keep rejects, which is the auditable cost.
        fallbacks += 0
    return x, y, time.time() - t0, rejects, fallbacks


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--sizes", nargs="+", type=int, default=[32, 64])
    p.add_argument("--full-fills", nargs="+", type=float,
                   default=[0.75, 0.80, 0.85, 0.90, 0.95])
    p.add_argument("--dense-fills", nargs="+", type=float,
                   default=[0.90, 0.95])
    p.add_argument("--full-instances", type=int, default=20)
    p.add_argument("--dense-instances", type=int, default=100)
    p.add_argument("--dense-tag", default="dense100_v3")
    p.add_argument("--efforts", nargs="+", type=int, default=[1, 128])
    p.add_argument("--n-cand", type=int, default=32)
    p.add_argument("--step-tries", type=int, default=1)
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--out", default="certified_study_v3.json")
    p.add_argument("--skip-full", action="store_true")
    p.add_argument("--skip-dense", action="store_true")
    p.add_argument("--no-resume", action="store_true")
    a = p.parse_args()

    out = Path(a.out)
    if out.exists() and not a.no_resume:
        with open(out, encoding="utf-8") as f:
            data = json.load(f)
        data["metadata"]["efforts"] = sorted(set(
            data["metadata"].get("efforts", []) + list(a.efforts)))
    else:
        data = {
            "metadata": {
                "purpose": "v3 witness-certified full sweep and dense confirmation",
                "split": a.split,
                "full_instances": a.full_instances,
                "dense_instances": a.dense_instances,
                "dense_tag": a.dense_tag,
                "n_cand": a.n_cand,
                "step_tries": a.step_tries,
                "efforts": a.efforts,
                "torch": torch.__version__,
                "numpy": np.__version__,
                "device": (torch.cuda.get_device_name(0)
                           if torch.cuda.is_available() else "CPU"),
                "deterministic_policy_decode": True,
            },
            "witness_checks": [],
            "records": [],
        }

    done = {record_key(r) for r in data["records"]}
    checked = {(r["suite"], r["n_rooms"], r["fill"])
               for r in data["witness_checks"]}
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    np.random.seed(0)

    suites = []
    if not a.skip_full:
        suites.append(("full", a.full_fills, a.full_instances, ""))
    if not a.skip_dense:
        suites.append(("dense", a.dense_fills, a.dense_instances,
                       a.dense_tag))

    for suite, fills, n_inst, suite_tag in suites:
        for n_rooms in a.sizes:
            if n_rooms not in POLICIES:
                raise ValueError(f"no policy mapping for n={n_rooms}")
            for fill in fills:
                cell = bench.suite_cell(
                    a.split, n_rooms, fill, n_inst, tag=suite_tag)
                cell["suite_tag"] = suite_tag
                inst = cell["inst"]
                obj = core.BatchObjective(inst)

                ck = (suite, n_rooms, float(fill))
                if ck not in checked:
                    wcheck = bench.verify_witness(cell)
                    if wcheck["max_overlap"] > 1e-9 or not wcheck["all_inside"]:
                        raise RuntimeError(f"invalid planted witness for {ck}: {wcheck}")
                    data["witness_checks"].append({
                        "suite": suite, "n_rooms": n_rooms,
                        "fill": float(fill), **wcheck})
                    checked.add(ck)
                    save_atomic(out, data)

                # The confirmatory dense suite includes the strongest simple
                # training-free ordering with and without certificate guidance.
                if suite == "dense":
                    for mode, label in (("none", "largest_first_unfiltered"),
                                        ("pack", "largest_first_guided")):
                        key = (suite, n_rooms, float(fill), label, None, None)
                        if key in done:
                            continue
                        x, y, wall, rejects, fallbacks = viability_layout(
                            inst, mode, a.n_cand, a.canvas)
                        rec = record_layout(
                            suite, cell, label, None, obj, x, y, wall,
                            extra={"rejects_total": rejects,
                                   "fallbacks_total": fallbacks})
                        data["records"].append(rec)
                        done.add(key)
                        print_record(rec)
                        save_atomic(out, data)

                for path in POLICIES[n_rooms]:
                    model, nrm, env, pargs = E.load_construct(
                        path, inst, dev, canvas=a.canvas)
                    pid = policy_id(path, pargs)

                    key = (suite, n_rooms, float(fill),
                           "policy_unfiltered", pid, None)
                    if key not in done:
                        t0 = time.time()
                        result = E.run_construct(
                            model, nrm, env, dev, deterministic=True)
                        rec = record_layout(
                            suite, cell, "policy_unfiltered", pid, obj,
                            result["x"], result["y"], time.time() - t0)
                        data["records"].append(rec)
                        done.add(key)
                        print_record(rec)
                        save_atomic(out, data)

                    key = (suite, n_rooms, float(fill),
                           "policy_guided", pid, None)
                    if key not in done:
                        t0 = time.time()
                        result = run_guided_heuristic(
                            model, nrm, env, dev, inst, n_cand=a.n_cand,
                            deterministic=True, seed=0, certificate="pack")
                        rec = record_layout(
                            suite, cell, "policy_guided", pid, obj,
                            result["x"], result["y"], time.time() - t0,
                            extra={"rejects_total": int(result["rejects"]),
                                   "fallbacks_total": int(result["fallbacks"])})
                        data["records"].append(rec)
                        done.add(key)
                        print_record(rec)
                        save_atomic(out, data)

                    for effort in a.efforts:
                        key = (suite, n_rooms, float(fill),
                               "policy_witness_certified", pid, effort)
                        if key in done:
                            continue
                        t0 = time.time()
                        result = run_certified_decode(
                            model, nrm, env, dev, inst, n_cand=a.n_cand,
                            deterministic=True, seed=0,
                            init_tries=effort, step_tries=a.step_tries)
                        rec = record_layout(
                            suite, cell, "policy_witness_certified", pid,
                            obj, result["x"], result["y"],
                            time.time() - t0,
                            returned=result["certified"], guaranteed=True,
                            effort=effort,
                            extra={
                                "witness_used_total": int(result["witness_used"]),
                                "rejects_total": int(result["rejects"]),
                            })
                        data["records"].append(rec)
                        done.add(key)
                        print_record(rec)
                        save_atomic(out, data)

    print(f"\nwrote {out} ({len(data['records'])} method-cell records)")


if __name__ == "__main__":
    main()
