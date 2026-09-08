"""Independent audit of stored shared-incumbent result files."""

import argparse
import json

import bench
from completion_search import LayoutProblem
from layout_verifier import verify_complete_layout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="completion_comparison_v5.json")
    parser.add_argument("--out", default="completion_comparison_audit_v5.json")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)
    metadata = data["metadata"]

    problems = {}
    for geometry in metadata["geometry"]:
        for n in metadata["rooms"]:
            for fill in metadata["fills"]:
                cell = bench.suite_cell(
                    metadata["split"], n, fill, metadata["instances_per_cell"],
                    tag=metadata["tag"] + "_" + geometry, geometry=geometry)
                for instance in range(metadata["instances_per_cell"]):
                    problems[(geometry, n, fill, instance)] = LayoutProblem(
                        cell["inst"].take([instance]))

    layouts = 0
    traces = 0
    largest_score_error = 0.0
    largest_trace_error = 0.0
    largest_admitted_time = 0.0
    for run in data["runs"]:
        key = (run["geometry"], run["n"], run["fill"], run["instance"])
        problem = problems[key]
        trace = run["trace"]
        times = [event["elapsed_s"] for event in trace]
        if times != sorted(times):
            raise RuntimeError("nonmonotone trace: {}".format(key))
        max_budget = max(row["budget_s"] for row in run["rows"])
        if times and max(times) > max_budget + 1e-12:
            raise RuntimeError("late event admitted: {}".format(key))
        largest_admitted_time = max(largest_admitted_time, max(times))
        traces += len(trace)
        for row in run["rows"]:
            placements = [tuple(item) for item in row["layout"]]
            check = verify_complete_layout(problem.iw, problem.ih,
                                           problem.gw, problem.gh, placements)
            if not check.valid:
                raise RuntimeError("invalid stored layout {}: {}".format(
                    key, check.reason))
            score = problem.score(placements, already_verified=True)
            error = abs(score - row["J"])
            largest_score_error = max(largest_score_error, error)
            if error > 1e-8:
                raise RuntimeError("stored objective mismatch: {}".format(key))
            eligible = [event["J"] for event in trace
                        if event["elapsed_s"] <= row["budget_s"] + 1e-12]
            trace_error = abs(max(eligible) - row["J"])
            largest_trace_error = max(largest_trace_error, trace_error)
            if trace_error > 1e-8:
                raise RuntimeError("budgeted trace mismatch: {}".format(key))
            layouts += 1
        if run["stats"].get("max_admitted_event_s", 0.0) > max_budget + 1e-12:
            raise RuntimeError("invalid max event statistic: {}".format(key))
        if run["stats"].get("final_verification_failures", 0):
            raise RuntimeError("verification failure recorded: {}".format(key))

    report = {
        "source": args.input,
        "runs": len(data["runs"]),
        "stored_budget_layouts": layouts,
        "trace_events": traces,
        "largest_objective_recheck_error": largest_score_error,
        "largest_budget_trace_error": largest_trace_error,
        "largest_admitted_event_s": largest_admitted_time,
        "event_time_rule": metadata.get("event_time_rule"),
        "objective_convention": metadata.get("objective_convention"),
        "status": "passed",
    }
    with open(args.out, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=1)
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
