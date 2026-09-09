"""Audit stored CP-SAT runs against the stated real-valued objective J_beta.

The script upgrades result files produced before the integer-to-real transfer
was recorded.  It does not rerun CP-SAT.  Each stored placement is independently
checked for geometry, scored with the real objective, and paired with a uniform
rounding allowance for the stored integer-model upper bound.
"""

import argparse
import json

from exact_cpsat import evaluate_real_objective, quantization_error_bound
from layout_verifier import verify_complete_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--specs", default="exact_specs_full.json")
    parser.add_argument("--results", default="exact_results_full.json")
    parser.add_argument("--out", default="exact_results_audited.json")
    args = parser.parse_args()

    with open(args.specs, encoding="utf-8") as handle:
        specs = json.load(handle)["specs"]
    with open(args.results, encoding="utf-8") as handle:
        results = json.load(handle)["results"]
    by_key = {(s["fill"], s["instance"]): s for s in specs}

    audited = []
    worst_score_error = 0.0
    worst_allowance = 0.0
    for source in results:
        row = dict(source)
        spec = by_key[(row["fill"], row["instance"])]
        delta, terms = quantization_error_bound(spec, row.get("scale", 10_000))
        row["rounded_terms"] = terms
        row["quantization_error_bound"] = delta
        worst_allowance = max(worst_allowance, delta)
        if "x" in row:
            placements = []
            for i, (x, y) in enumerate(zip(row["x"], row["y"])):
                placements.append((i, y - spec["h"][i] / 2.0,
                                   x - spec["w"][i] / 2.0))
            check = verify_complete_layout(spec["w"], spec["h"],
                                           spec["gw"], spec["gh"], placements)
            if not check.valid:
                raise RuntimeError("invalid CP-SAT layout {}: {}".format(
                    (row["fill"], row["instance"]), check.reason))
            old_objective = row["objective"]
            old_bound = row["bound"]
            row["quantized_objective"] = old_objective
            row["quantized_bound"] = old_bound
            row["objective"] = evaluate_real_objective(spec, row["x"], row["y"])
            row["bound"] = old_bound + delta
            error = abs(row["objective"] - row["quantized_objective"])
            if error > delta + 1e-10:
                raise RuntimeError("rounding bound violated: {} > {}".format(
                    error, delta))
            row["observed_quantization_error"] = error
            worst_score_error = max(worst_score_error, error)
        audited.append(row)

    metadata = {
        "objective": "real-valued J_beta in the main manuscript",
        "integer_scale": 10_000,
        "bound_transfer": "real upper bound = quantized CP-SAT bound + delta",
        "delta": "number of rounded active terms divided by twice the scale",
        "worst_delta": worst_allowance,
        "worst_observed_solution_error": worst_score_error,
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump({"metadata": metadata, "results": audited}, handle, indent=1)
    print("audited {} runs; worst |J-Jhat|={:.6g}, worst delta={:.6g}".format(
        len(audited), worst_score_error, worst_allowance))
    print("wrote {}".format(args.out))


if __name__ == "__main__":
    main()
