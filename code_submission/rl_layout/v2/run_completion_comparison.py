"""Matched-time comparison from a shared verified initial layout.

The default configuration is the confirmatory protocol.  For development, use
``--instances`` and shorter ``--budgets`` explicitly and keep a different tag.
Each stochastic method is run once to the largest budget; shorter budgets are
read from the same anytime trace.  Initial witness generation and verification
are measured once and charged to every method.
"""

import argparse
import json
import platform
import sys
import time

import numpy as np

import bench
from completion_search import LayoutProblem, METHODS, run_anytime


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test")
    parser.add_argument("--tag", default="shared_incumbent_confirm_v5")
    parser.add_argument("--geometry", nargs="+",
                        choices=("guillotine", "nonslicing"),
                        default=["guillotine", "nonslicing"])
    parser.add_argument("--rooms", nargs="+", type=int, default=[32, 64])
    parser.add_argument("--fills", nargs="+", type=float, default=[0.90, 0.95])
    parser.add_argument("--instances", type=int, default=100)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--budgets", nargs="+", type=float,
                        default=[0.1, 0.3, 1.0, 3.0, 10.0])
    parser.add_argument("--methods", nargs="+", choices=METHODS,
                        default=list(METHODS))
    parser.add_argument("--kappa", type=int, default=16)
    parser.add_argument("--repair-cap", type=int, default=8)
    parser.add_argument("--lns-cap", type=int, default=16)
    parser.add_argument("--completion-tries", type=int, default=1)
    parser.add_argument("--out", default="completion_comparison_v5.json")
    args = parser.parse_args()

    payload = {
        "metadata": {
            "created_unix": time.time(),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "split": args.split,
            "tag": args.tag,
            "geometry": args.geometry,
            "rooms": args.rooms,
            "fills": args.fills,
            "instances_per_cell": args.instances,
            "evaluation_seeds": args.seeds,
            "post_initialization_budgets_s": sorted(args.budgets),
            "methods": args.methods,
            "initializer": "deterministic best-contact, decreasing area order",
            "initialization_accounting": "measured once and charged to every method",
            "event_time_rule": "candidate admitted only after verification and scoring finish by budget",
            "kappa": args.kappa,
            "repair_cap": args.repair_cap,
            "lns_cap": args.lns_cap,
            "lns_small_cap": 4,
            "completion_tries": args.completion_tries,
            "objective_direction": "maximize",
            "objective_convention": "feasible J_beta includes no-overlap bonus beta_max",
            "practical_effect_threshold": "1 percent of shared initial objective",
        },
        "initializations": [],
        "runs": [],
    }

    total_cells = len(args.geometry) * len(args.rooms) * len(args.fills)
    cell_count = 0
    for geometry in args.geometry:
        for n in args.rooms:
            for fill in args.fills:
                cell_count += 1
                family_tag = args.tag + "_" + geometry
                cell = bench.suite_cell(args.split, n, fill, args.instances,
                                        tag=family_tag, geometry=geometry)
                planted = bench.verify_witness(cell)
                if not planted["all_inside"] or planted["max_overlap"] > 1e-9:
                    raise RuntimeError("invalid source packing")
                initialized = 0
                print("cell {}/{}: {} n={} fill={:.2f}".format(
                    cell_count, total_cells, geometry, n, fill), flush=True)
                for instance_index in range(args.instances):
                    problem = LayoutProblem(cell["inst"].take([instance_index]))
                    ok, witness, init_seconds = problem.initial_best_contact()
                    init_row = {
                        "geometry": geometry, "n": n, "fill": fill,
                        "instance": instance_index, "initialized": bool(ok),
                        "initialization_s": init_seconds,
                    }
                    initial_value = None
                    if ok:
                        initialized += 1
                        score_started = time.perf_counter()
                        initial_value = problem.score(witness)
                        init_seconds += time.perf_counter() - score_started
                        init_row["initialization_s"] = init_seconds
                        init_row["initial_J"] = initial_value
                    payload["initializations"].append(init_row)
                    if not ok:
                        continue
                    seed_offsets = {
                        "B0": 0, "B1": 1, "B2": 2, "B3": 3,
                        "M0": 4, "M1": 5, "R0": 6, "B2S": 7,
                    }
                    for method in args.methods:
                        seed_count = 1 if method == "B0" else args.seeds
                        for evaluation_seed in range(seed_count):
                            seed = (cell["seed"] + instance_index * 1009
                                    + evaluation_seed * 9176
                                    + seed_offsets[method] * 1000003)
                            result = run_anytime(
                                problem, witness, method, args.budgets, seed=seed,
                                kappa=args.kappa, repair_cap=args.repair_cap,
                                lns_cap=args.lns_cap,
                                completion_tries=args.completion_tries,
                                initial_value=initial_value)
                            payload["runs"].append({
                                "geometry": geometry, "n": n, "fill": fill,
                                "instance": instance_index,
                                "evaluation_seed": evaluation_seed,
                                "search_seed": seed,
                                "initialization_s": init_seconds,
                                **result,
                            })
                    print("  {:3d}/{:3d}, initialized {:3d}".format(
                        instance_index + 1, args.instances, initialized), flush=True)
                print("  coverage {}/{} = {:.3f}".format(
                    initialized, args.instances, initialized / args.instances),
                    flush=True)
                with open(args.out, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle, indent=1)
    print("wrote {} with {} initialized runs".format(args.out,
                                                      len(payload["runs"])))


if __name__ == "__main__":
    main()
