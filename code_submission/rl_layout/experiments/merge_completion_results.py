"""Merge disjoint cells produced by ``run_completion_comparison.py``."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    payloads = []
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as handle:
            payloads.append(json.load(handle))

    first = payloads[0]
    metadata = dict(first["metadata"])
    metadata["geometry"] = sorted({
        geometry
        for payload in payloads
        for geometry in payload["metadata"]["geometry"]
    })
    metadata["rooms"] = sorted({
        rooms
        for payload in payloads
        for rooms in payload["metadata"]["rooms"]
    })
    metadata["fills"] = sorted({
        fill
        for payload in payloads
        for fill in payload["metadata"]["fills"]
    })
    metadata["merged_from"] = list(args.inputs)
    metadata["objective_convention"] = (
        "feasible J_beta includes no-overlap bonus beta_max")

    invariant_keys = (
        "split", "tag", "instances_per_cell", "evaluation_seeds",
        "post_initialization_budgets_s", "methods", "initializer",
        "initialization_accounting", "event_time_rule", "kappa",
        "repair_cap", "lns_cap", "lns_small_cap", "completion_tries",
        "objective_direction", "practical_effect_threshold",
    )
    for payload in payloads[1:]:
        for key in invariant_keys:
            if payload["metadata"].get(key) != metadata.get(key):
                raise ValueError("metadata mismatch for {}".format(key))

    merged = {
        "metadata": metadata,
        "initializations": [
            row for payload in payloads for row in payload["initializations"]
        ],
        "runs": [row for payload in payloads for row in payload["runs"]],
    }
    merged["initializations"].sort(
        key=lambda row: (row["geometry"], row["n"], row["fill"],
                         row["instance"]))
    merged["runs"].sort(
        key=lambda row: (row["geometry"], row["n"], row["fill"],
                         row["instance"], row["method"],
                         row["evaluation_seed"]))

    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(merged, handle, indent=1)
    print("wrote {} with {} initializations and {} runs".format(
        args.out, len(merged["initializations"]), len(merged["runs"])))


if __name__ == "__main__":
    main()
