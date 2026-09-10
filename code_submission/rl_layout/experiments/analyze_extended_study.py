"""Independent saved-layout audit and instance-paired extension analysis.

Only complete phases produce tables. The bootstrap unit is an instance; seeds
are averaged before sampling, and geometry strata retain their observed sizes.
All time-to-target summaries retain non-hits as administratively censored runs.
"""

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from extended_protocol import (atomic_json, cell_seed, digest, generate_cell,
                               run_key, search_seed, source_hashes, task_key)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory):
    from completion_search import LayoutProblem
    from layout_verifier import verify_complete_layout
    import bench

    frozen = read(directory / "protocol.json")
    protocol = frozen["settings"]
    require((directory / "complete.json").exists(), "phase is incomplete: no final tables")
    require(frozen["protocol_sha256"] == digest(protocol), "protocol digest mismatch")
    require(frozen["source_sha256"] == source_hashes(), "source differs from the frozen study")
    initializations, runs, expected_initializations, expected_runs = [], [], set(), set()
    manifest, verified_layouts, max_score_error = {}, 0, 0.0
    for cell in protocol["cells"]:
        generated = generate_cell(protocol, cell)
        source_check = bench.verify_witness(generated)
        require(source_check["all_inside"] and source_check["max_overlap"] <= 1e-9,
                "generated packing is not feasible")
        for instance in range(cell["instances"]):
            identity = task_key(cell, instance)
            init_path = directory / "initializations" / (identity + ".json")
            expected_initializations.add(init_path.name)
            init = read(init_path)
            manifest[str(init_path.relative_to(directory))] = hashlib.sha256(init_path.read_bytes()).hexdigest()
            require(all(init[k] == cell[k] for k in ("geometry", "n", "fill"))
                    and init["instance"] == instance, "initialization identity mismatch")
            require(init["protocol_sha256"] == frozen["protocol_sha256"], "initializer protocol mismatch")
            require(init["cell_seed"] == cell_seed(protocol, cell) == generated["seed"], "cell seed mismatch")
            require(abs(init["achieved_fill"] - generated["achieved_fill"][instance]) < 1e-12,
                    "achieved fill mismatch")
            problem = LayoutProblem(generated["inst"].take([instance]))
            require((init["gw"], init["gh"]) == (problem.gw, problem.gh), "plate size mismatch")
            require(np.isfinite(init["initialization_s"]) and init["initialization_s"] >= 0,
                    "invalid initialization time")
            # Initialization is deterministic; coverage and coordinates can be
            # regenerated even for records claiming that initialization failed.
            ok, witness, _ = problem.initial_best_contact()
            require(bool(ok) == init["initialized"], "initialization coverage mismatch")
            initializations.append(init)
            if not ok:
                require(init["initial_layout"] is None and init["initial_J"] is None,
                        "failed initialization contains a layout")
                continue
            require(sorted(map(tuple, witness)) == sorted(map(tuple, init["initial_layout"])),
                    "initial layout differs from deterministic initializer")
            initial_value = problem.score(witness)
            require(abs(initial_value - init["initial_J"]) < 1e-8, "initial objective mismatch")
            verified_layouts += 1
            for method in protocol["methods"]:
                for seed in range(protocol["evaluation_seeds"]):
                    path = directory / "runs" / (run_key(cell, instance, method, seed) + ".json")
                    expected_runs.add(path.name)
                    run = read(path)
                    manifest[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
                    require(all(run[k] == cell[k] for k in ("geometry", "n", "fill"))
                            and run["instance"] == instance and run["method"] == method
                            and run["evaluation_seed"] == seed, "run identity mismatch")
                    require(run["search_seed"] == search_seed(protocol, cell, instance, method, seed),
                            "search seed mismatch")
                    require(run["protocol_sha256"] == frozen["protocol_sha256"], "run protocol mismatch")
                    require(run["cpu_id"] == init["cpu_id"], "instance used different CPUs")
                    require(abs(run["initial_J"] - initial_value) < 1e-8, "not a common initialization")
                    require(run["initialization_s"] == init["initialization_s"], "initialization time mismatch")
                    trace = run["trace"]
                    times = [event["elapsed_s"] for event in trace]
                    values = [event["J"] for event in trace]
                    require(bool(trace) and times == sorted(times) and times[0] == 0,
                            "invalid trace chronology")
                    require(np.isfinite(times).all() and np.isfinite(values).all(), "nonfinite trace")
                    require(abs(values[0] - initial_value) < 1e-8, "trace initial value mismatch")
                    require(times[-1] <= protocol["budgets_s"][-1], "late event admitted")
                    require(run["stats"]["actual_search_elapsed_s"] + 1e-6 >= protocol["budgets_s"][-1],
                            "search stopped before the common cutoff")
                    require(run["stats"]["final_verification_failures"] == 0, "recorded verifier failure")
                    require([r["budget_s"] for r in run["rows"]] == protocol["budgets_s"],
                            "budget row mismatch")
                    previous = initial_value
                    for row in run["rows"]:
                        check = verify_complete_layout(problem.iw, problem.ih, problem.gw,
                                                       problem.gh, row["layout"])
                        require(check.valid, "invalid stored layout: " + check.reason)
                        value = problem.score(row["layout"], already_verified=True)
                        error = abs(value - row["J"])
                        max_score_error = max(max_score_error, error)
                        require(error < 1e-8, "stored objective mismatch")
                        eligible = max(event["J"] for event in trace if event["elapsed_s"] <= row["budget_s"])
                        require(abs(value - eligible) < 1e-8, "budget row is not best eligible event")
                        require(value >= previous - 1e-8, "incumbent regression")
                        require(abs(row["improvement"] - (value - initial_value)) < 1e-8,
                                "raw improvement mismatch")
                        require(abs(row["improvement_pct"] - 100 * (value - initial_value) / initial_value) < 1e-8,
                                "percent improvement mismatch")
                        previous = value
                        verified_layouts += 1
                    runs.append(run)
    require({p.name for p in (directory / "runs").glob("*.json")} == expected_runs,
            "missing or unexpected run files")
    require({p.name for p in (directory / "initializations").glob("*.json")} == expected_initializations,
            "missing or unexpected initialization files")
    complete = read(directory / "complete.json")
    require(complete["runs"] == complete["expected_runs"] == len(runs), "completion count mismatch")
    require(complete["attempted"] == len(initializations)
            and complete["initialized"] == sum(i["initialized"] for i in initializations),
            "completion coverage mismatch")
    report = {"passed": True, "attempted_instances": len(initializations),
              "initialized_instances": sum(i["initialized"] for i in initializations),
              "runs": len(runs), "verified_saved_layouts": verified_layouts,
              "max_objective_error": max_score_error,
              "protocol_sha256": frozen["protocol_sha256"],
              "results_manifest_sha256": digest(manifest), "files_sha256": manifest}
    atomic_json(directory / "audit.json", report)
    return protocol, initializations, runs, report


def paired_interval(values, strata, seed, replicates=10000):
    values = np.asarray(values, dtype=float)
    require(len(values) > 0 and len(values) == len(strata), "invalid paired observations")
    rng = np.random.default_rng(seed)
    resampled_sum = np.zeros(replicates)
    means = []
    groups = sorted(set(strata))
    for stratum in groups:
        data = values[[i for i, label in enumerate(strata) if label == stratum]]
        indices = rng.integers(len(data), size=(replicates, len(data)))
        resampled_sum += data[indices].mean(axis=1)
        means.append(data.mean())
    low, high = np.percentile(resampled_sum / len(groups), [2.5, 97.5])
    return float(np.mean(means)), float(low), float(high)


def distribution(values):
    values = np.asarray(values, dtype=float)
    return {"mean": float(values.mean()), "min": float(values.min()),
            "median": float(np.median(values)), "p95": float(np.percentile(values, 95)),
            "max": float(values.max())} if len(values) else None


def summarize(protocol, inits, runs):
    quality, targets, contrasts, coverage = [], [], [], []
    grouped = defaultdict(list)
    for run in runs:
        grouped[(run["geometry"], run["n"], run["fill"])].append(run)
    # Always include failed-initialization cells in coverage, even without runs.
    for cell in protocol["cells"]:
        geometry, n, fill = cell["geometry"], cell["n"], cell["fill"]
        local_init = [i for i in inits if (i["geometry"], i["n"], i["fill"]) == (geometry, n, fill)]
        coverage.append({"geometry": geometry, "n": n, "fill": fill,
                         "attempted": len(local_init), "initialized": sum(i["initialized"] for i in local_init),
                         "achieved_fill": distribution([i["achieved_fill"] for i in local_init]),
                         "initialization_s": distribution([i["initialization_s"] for i in local_init]),
                         "mean_facility_area": distribution([i["mean_facility_area"] for i in local_init])})
        for method in protocol["methods"]:
            selected = [r for r in grouped[(geometry, n, fill)] if r["method"] == method]
            if not selected:
                continue
            identity = {"geometry": geometry, "n": n, "fill": fill, "method": method}
            for index, budget in enumerate(protocol["budgets_s"]):
                by_instance = defaultdict(list)
                for r in selected:
                    by_instance[r["instance"]].append(r["rows"][index])
                quality.append(dict(identity, budget_s=budget, instances=len(by_instance),
                                    mean_J=float(np.mean([np.mean([v["J"] for v in rows]) for rows in by_instance.values()])),
                                    mean_gain=float(np.mean([np.mean([v["improvement"] for v in rows]) for rows in by_instance.values()])),
                                    improvement_pct=float(np.mean([np.mean([v["improvement_pct"] for v in rows]) for rows in by_instance.values()])),
                                    total_time_mean_s=budget + float(np.mean([r["initialization_s"] for r in selected]))))
            cutoff = protocol["budgets_s"][-1]
            for target in protocol["target_improvements_pct"]:
                observations = []
                for run in selected:
                    hits = [event["elapsed_s"] for event in run["trace"]
                            if 100 * (event["J"] - run["initial_J"]) / run["initial_J"] >= target]
                    time = min(hits) if hits else cutoff
                    observations.append((bool(hits), time, time + run["initialization_s"]))
                targets.append(dict(identity, target_pct=target, cutoff_s=cutoff, runs=len(observations),
                                    hit_fraction=float(np.mean([v[0] for v in observations])),
                                    restricted_mean_search_s=float(np.mean([v[1] for v in observations])),
                                    restricted_mean_plus_initialization_s=float(np.mean([v[2] for v in observations]))))
    if "M1" in protocol["methods"]:
        for n in sorted({c["n"] for c in protocol["cells"]}):
            for fill in sorted({c["fill"] for c in protocol["cells"]}):
                for geometry in ["pooled", "guillotine", "nonslicing"]:
                    chosen = [r for r in runs if r["n"] == n and r["fill"] == fill
                              and (geometry == "pooled" or r["geometry"] == geometry)]
                    if geometry == "pooled" and {r["geometry"] for r in chosen} != {"guillotine", "nonslicing"}:
                        continue  # no two-family estimate if one has zero coverage
                    for control in [m for m in protocol["methods"] if m != "M1"]:
                        for index, budget in enumerate(protocol["budgets_s"]):
                            paired = defaultdict(lambda: defaultdict(list))
                            for run in chosen:
                                paired[(run["geometry"], run["instance"])][run["method"]].append(run["rows"][index])
                            keys = sorted(paired)
                            if not keys:
                                continue
                            for key in keys:
                                require(len(paired[key]["M1"]) == len(paired[key][control]) == protocol["evaluation_seeds"],
                                        "unpaired or incomplete seeds")
                            diff = [float(np.mean([x["improvement_pct"] for x in paired[k]["M1"]])
                                          - np.mean([x["improvement_pct"] for x in paired[k][control]])) for k in keys]
                            raw = [float(np.mean([x["improvement"] for x in paired[k]["M1"]])
                                         - np.mean([x["improvement"] for x in paired[k][control]])) for k in keys]
                            bootstrap_seed = int(digest([n, fill, geometry, control, budget])[:12], 16)
                            effect, low, high = paired_interval(diff, [k[0] for k in keys], bootstrap_seed)
                            contrasts.append({"geometry": geometry, "n": n, "fill": fill,
                                              "control": control, "budget_s": budget, "instances": len(keys),
                                              "M1_minus_control_pp": effect, "ci_low": low, "ci_high": high,
                                              "mean_raw_difference": float(np.mean([np.mean([raw[i] for i, k in enumerate(keys) if k[0] == g]) for g in sorted({k[0] for k in keys})])),
                                              "win_fraction": float(np.mean([np.mean([diff[i] > 1e-9 for i, k in enumerate(keys) if k[0] == g]) for g in sorted({k[0] for k in keys})]))})
    timing = {}
    for method in protocol["methods"]:
        selected = [r for r in runs if r["method"] == method]
        timing[method] = {"runs": len(selected),
                          "cpu_wall_ratio": distribution([r["cpu_wall_ratio"] for r in selected]),
                          "search_overrun_s": distribution([r["stats"]["overrun_s"] for r in selected]),
                          "verification_failures": sum(r["stats"]["final_verification_failures"] for r in selected),
                          "late_completed_events": sum(r["stats"]["late_completed_events"] for r in selected)}
    return {"complete": True, "phase": protocol["phase"], "coverage": coverage,
            "quality": quality, "paired_contrasts": contrasts, "time_to_target": targets,
            "timing": timing, "bootstrap_replicates": 10000,
            "interval_scope": "descriptive pointwise 95% intervals; no multiplicity adjustment"}


def write_report(directory, summary, audit_report):
    for key in ("quality", "paired_contrasts", "time_to_target"):
        rows = summary[key]
        if rows:
            with (directory / (key + ".csv")).open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
    lines = ["# Extended comparison: " + summary["phase"], "",
             "Complete phase; independent saved-layout audit passed.", "",
             "Attempted: {}; initialized: {}; runs: {}; verified saved layouts: {}.".format(
                 audit_report["attempted_instances"], audit_report["initialized_instances"],
                 audit_report["runs"], audit_report["verified_saved_layouts"]), "",
             "Seeds are averaged within instance. Intervals are descriptive, geometry-stratified",
             "paired bootstrap intervals, not multiplicity-adjusted significance tests.", "",
             "| Geometry | n | Nominal fill | Initialized / attempted | Achieved fill range |",
             "|---|---:|---:|---:|---:|"]
    for row in summary["coverage"]:
        lines.append("| {geometry} | {n} | {fill:.2f} | {initialized}/{attempted} | ".format(**row)
                     + "{min:.5f} to {max:.5f} |".format(**row["achieved_fill"]))
    primary = [r for r in summary["paired_contrasts"] if r["geometry"] == "pooled" and r["budget_s"] in (10, 60)]
    if primary:
        lines += ["", "## Primary contrasts", "",
                  "| n | Fill | Seconds | Control | Instances | M1 minus control (pp) | 95% interval |",
                  "|---:|---:|---:|---|---:|---:|---:|"]
        for row in primary:
            lines.append("| {n} | {fill:.2f} | {budget_s:g} | {control} | {instances} | {M1_minus_control_pp:+.2f} | [{ci_low:+.2f}, {ci_high:+.2f}] |".format(**row))
        tex = ["% Generated only after a complete phase and successful audit.",
               "% Descriptive paired bootstrap intervals, geometry-stratified.",
               r"\begin{tabular}{rrrlrr}", r"\toprule",
               r"$n$ & Fill & Seconds & Control & Difference (pp) & 95\% interval \\", r"\midrule"]
        for row in primary:
            label = "M0" if row["control"] == "M0" else "ALNS"
            tex.append("{} & {:.2f} & {:g} & {} & {:+.2f} & [{:+.2f}, {:+.2f}] \\\\".format(
                row["n"], row["fill"], row["budget_s"], label, row["M1_minus_control_pp"], row["ci_low"], row["ci_high"]))
        tex += [r"\bottomrule", r"\end{tabular}"]
        (directory / "primary_contrasts.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    lines += ["", "Full quality, target-time and paired-contrast records are in the adjacent CSV files.",
              "Non-hits contribute the cutoff to restricted mean search time, rather than being dropped.",
              "Initialization is included in the companion total-time column.", ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path, help="completed phase directory, not its parent")
    args = parser.parse_args()
    directory = args.directory.resolve()
    protocol, inits, runs, checked = audit(directory)
    summary = summarize(protocol, inits, runs)
    summary["protocol_sha256"] = checked["protocol_sha256"]
    summary["results_manifest_sha256"] = checked["results_manifest_sha256"]
    atomic_json(directory / "summary.json", summary)
    write_report(directory, summary, checked)
    print("AUDIT AND ANALYSIS PASSED: {} instances, {} runs, {} saved layouts".format(
        len(inits), len(runs), checked["verified_saved_layouts"]))


if __name__ == "__main__":
    main()
