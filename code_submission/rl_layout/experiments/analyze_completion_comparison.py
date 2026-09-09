"""Instance-level analysis of the matched-time completion comparison."""

import argparse
import json
from collections import defaultdict

import numpy as np


def bootstrap_mean_ci(values, rng, draws=5000):
    values = np.asarray(values, dtype=float)
    if not len(values):
        return [None, None]
    samples = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(v) for v in np.quantile(samples, [0.025, 0.975])]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="completion_comparison.json")
    parser.add_argument("--out", default="completion_comparison_summary.json")
    parser.add_argument("--controls", nargs="+",
                        default=["B1", "B2", "B2S", "B3", "M0", "R0"])
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)
    rng = np.random.default_rng(20260907)

    init_groups = defaultdict(list)
    for row in data["initializations"]:
        key = (row["geometry"], row["n"], row["fill"])
        init_groups[key].append(row)

    # Average stochastic seeds within an instance before paired inference.
    seed_values = defaultdict(list)
    init_value = {}
    stats_by_method = defaultdict(list)
    for run in data["runs"]:
        cell = (run["geometry"], run["n"], run["fill"])
        init_value[cell + (run["instance"],)] = run["initial_J"]
        stats_by_method[cell + (run["method"],)].append(run["stats"])
        for point in run["rows"]:
            key = cell + (run["instance"], run["method"], point["budget_s"])
            seed_values[key].append(point["J"])
    instance_value = {key: float(np.mean(values))
                      for key, values in seed_values.items()}

    cells = []
    for cell in sorted(init_groups):
        inits = init_groups[cell]
        initialized = [r for r in inits if r["initialized"]]
        cell_row = {
            "geometry": cell[0], "n": cell[1], "fill": cell[2],
            "attempted_instances": len(inits),
            "initialized_instances": len(initialized),
            "coverage": len(initialized) / len(inits),
            "mean_initialization_s": float(np.mean(
                [r["initialization_s"] for r in inits])),
            "budgets": [],
        }
        methods = sorted({key[4] for key in instance_value
                          if key[:3] == cell})
        budgets = sorted({key[5] for key in instance_value
                          if key[:3] == cell})
        for budget in budgets:
            entry = {"budget_s": budget, "methods": {}}
            for method in methods:
                values = {key[3]: value for key, value in instance_value.items()
                          if key[:3] == cell and key[4] == method
                          and key[5] == budget}
                ids = sorted(values)
                improvements = [values[i] - init_value[cell + (i,)] for i in ids]
                normalized = [100.0 * d / init_value[cell + (i,)]
                              for i, d in zip(ids, improvements)]
                row = {
                    "instances": len(ids),
                    "mean_J": float(np.mean([values[i] for i in ids])),
                    "mean_improvement": float(np.mean(improvements)),
                    "mean_improvement_pct": float(np.mean(normalized)),
                    "improvement_pct_ci95": bootstrap_mean_ci(normalized, rng),
                    "improved_instances": int(np.sum(np.asarray(improvements) > 1e-9)),
                }
                for control_name in args.controls:
                    control = {key[3]: value for key, value in instance_value.items()
                               if key[:3] == cell and key[4] == control_name
                               and key[5] == budget}
                    paired_ids = sorted(set(ids) & set(control))
                    if not paired_ids:
                        continue
                    delta = np.array([values[i] - control[i] for i in paired_ids])
                    practical = np.array([0.01 * init_value[cell + (i,)]
                                          for i in paired_ids])
                    delta_pct = np.array([
                        100.0 * (values[i] - control[i]) / init_value[cell + (i,)]
                        for i in paired_ids])
                    row["vs_{}_mean".format(control_name)] = float(delta.mean())
                    row["vs_{}_ci95".format(control_name)] = \
                        bootstrap_mean_ci(delta, rng)
                    row["vs_{}_mean_pct_points".format(control_name)] = \
                        float(delta_pct.mean())
                    row["vs_{}_pct_points_ci95".format(control_name)] = \
                        bootstrap_mean_ci(delta_pct, rng)
                    row["vs_{}_win_tie_loss".format(control_name)] = [
                        int(np.sum(delta > 1e-9)),
                        int(np.sum(np.abs(delta) <= 1e-9)),
                        int(np.sum(delta < -1e-9)),
                    ]
                    row["vs_{}_practical_wins_1pct".format(control_name)] = \
                        int(np.sum(delta >= practical))
                entry["methods"][method] = row
            cell_row["budgets"].append(entry)

        mechanism = {}
        for method in methods:
            stats = stats_by_method[cell + (method,)]
            if not stats:
                continue
            displaced = [value for s in stats
                         for value in s.get("displaced_sizes", [])]
            destroyed = [value for s in stats
                         for value in s.get("destroy_sizes", [])]
            branch_names = sorted({name for s in stats
                                   for name in s.get("verified_events_by_branch", {})})
            improvement_branches = sorted({name for s in stats
                                           for name in s.get(
                                               "incumbent_improvements_by_branch", {})})
            mechanism[method] = {
                "runs": len(stats),
                "mean_verified_completions": float(np.mean(
                    [s["verified_completions"] for s in stats])),
                "mean_full_completion_calls": float(np.mean(
                    [s["full_completion_calls"] for s in stats])),
                "mean_unchanged_reuses": float(np.mean(
                    [s["unchanged_reuses"] for s in stats])),
                "mean_repair_attempts": float(np.mean(
                    [s["repair_attempts"] for s in stats])),
                "repair_success_rate": (sum(s["repair_successes"] for s in stats)
                                        / max(1, sum(s["repair_attempts"] for s in stats))),
                "repair_attempts_total": int(sum(
                    s.get("repair_attempts", 0) for s in stats)),
                "repair_successes_total": int(sum(
                    s.get("repair_successes", 0) for s in stats)),
                "mean_displaced": (float(np.mean(displaced))
                                   if displaced else None),
                "displaced_observations": len(displaced),
                "proposals_D0_total": int(sum(
                    s.get("proposals_D0", 0) for s in stats)),
                "proposals_D1_to_cap_total": int(sum(
                    s.get("proposals_D1_to_cap", 0) for s in stats)),
                "proposals_D_above_cap_total": int(sum(
                    s.get("proposals_D_above_cap", 0) for s in stats)),
                "mean_fallback_reuses": float(np.mean(
                    [s.get("fallback_reuses", 0) for s in stats])),
                "destroy_attempts": len(destroyed),
                "mean_destroy_size": (float(np.mean(destroyed))
                                      if destroyed else None),
                "destroy_size_histogram": {
                    str(size): int(sum(value == size for value in destroyed))
                    for size in sorted(set(destroyed))},
                "mean_completion_generation_s": float(np.mean(
                    [s.get("completion_generation_seconds", 0.0) for s in stats])),
                "mean_repair_generation_s": float(np.mean(
                    [s.get("repair_generation_seconds", 0.0) for s in stats])),
                "mean_intermediate_verification_s": float(np.mean(
                    [s.get("intermediate_verification_seconds", 0.0)
                     for s in stats])),
                "mean_final_verification_s": float(np.mean(
                    [s.get("final_verification_seconds", 0.0) for s in stats])),
                "mean_objective_s": float(np.mean(
                    [s.get("objective_seconds", 0.0) for s in stats])),
                "mean_actual_search_elapsed_s": float(np.mean(
                    [s.get("actual_search_elapsed_s", 0.0) for s in stats])),
                "mean_overrun_s": float(np.mean(
                    [s.get("overrun_s", 0.0) for s in stats])),
                "max_overrun_s": float(np.max(
                    [s.get("overrun_s", 0.0) for s in stats])),
                "runs_with_overrun": int(sum(
                    s.get("overrun_s", 0.0) > 1e-12 for s in stats)),
                "late_completed_events": int(sum(
                    s.get("late_completed_events", 0) for s in stats)),
                "mean_verified_events_by_branch": {
                    name: float(np.mean([
                        s.get("verified_events_by_branch", {}).get(name, 0)
                        for s in stats])) for name in branch_names},
                "incumbent_improvements_by_branch": {
                    name: int(sum(s.get("incumbent_improvements_by_branch", {})
                                  .get(name, 0) for s in stats))
                    for name in improvement_branches},
                "incumbent_gain_by_branch": {
                    name: float(sum(s.get("incumbent_gain_by_branch", {})
                                    .get(name, 0.0) for s in stats))
                    for name in improvement_branches},
                "final_verification_failures": int(sum(
                    s["final_verification_failures"] for s in stats)),
            }
        cell_row["mechanism"] = mechanism
        cells.append(cell_row)

    # One pooled contrast can serve as the confirmatory endpoint without
    # promoting four cellwise intervals to four separate primary tests.  The
    # cell rows above remain useful for examining heterogeneity.
    overall = []
    fills = sorted({key[2] for key in instance_value})
    all_budgets = sorted({key[5] for key in instance_value})
    for fill in fills:
        for budget in all_budgets:
            identifiers = sorted({key[:4] for key in instance_value
                                  if key[2] == fill and key[4] == "M1"
                                  and key[5] == budget})
            if not identifiers:
                continue
            row = {"fill": fill, "budget_s": budget,
                   "instances": len(identifiers), "contrasts": {}}
            for control_name in args.controls:
                paired = [identifier for identifier in identifiers
                          if identifier + (control_name, budget)
                          in instance_value]
                if not paired:
                    continue
                delta = np.asarray([
                    instance_value[identifier + ("M1", budget)]
                    - instance_value[identifier + (control_name, budget)]
                    for identifier in paired])
                delta_pct = np.asarray([
                    100.0 * value / init_value[identifier]
                    for identifier, value in zip(paired, delta)])
                practical = np.asarray([0.01 * init_value[identifier]
                                        for identifier in paired])
                row["contrasts"][control_name] = {
                    "instances": len(paired),
                    "mean": float(delta.mean()),
                    "ci95": bootstrap_mean_ci(delta, rng),
                    "mean_pct_points": float(delta_pct.mean()),
                    "pct_points_ci95": bootstrap_mean_ci(delta_pct, rng),
                    "win_tie_loss": [
                        int(np.sum(delta > 1e-9)),
                        int(np.sum(np.abs(delta) <= 1e-9)),
                        int(np.sum(delta < -1e-9)),
                    ],
                    "practical_wins_1pct": int(np.sum(delta >= practical)),
                }
            overall.append(row)

    summary = {
        "metadata": {
            "source": args.input,
            "inference_unit": "instance after averaging evaluation seeds",
            "bootstrap_draws": 5000,
            "controls": args.controls,
            "practical_threshold": "one percent of shared initial J",
            "event_time_rule": data["metadata"].get("event_time_rule"),
            "objective_convention": data["metadata"].get("objective_convention"),
        },
        "cells": cells,
        "overall": overall,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=1)
    for cell in cells:
        print("{} n={} f={:.2f}: init {}/{}".format(
            cell["geometry"], cell["n"], cell["fill"],
            cell["initialized_instances"], cell["attempted_instances"]))
        if cell["budgets"]:
            last = cell["budgets"][-1]
            print("  budget {:.3g}s: ".format(last["budget_s"]) + ", ".join(
                "{} {:+.2f}%".format(m, r["mean_improvement_pct"])
                for m, r in last["methods"].items()))
    print("wrote " + args.out)


if __name__ == "__main__":
    main()
