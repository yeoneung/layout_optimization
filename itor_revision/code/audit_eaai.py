"""Independent audit of an EAAI phase directory (any method set).

Adapted from the archive's analyze_extended_study.audit: regenerates every
cell from the frozen protocol, checks instance identity, cell seeds, search
seeds, common initialization, trace chronology and eligibility, re-verifies
every stored budget layout with the independent verifier, recomputes every
objective, and writes audit.json with a manifest of file hashes.  Missing or
unexpected files fail the audit.  Run only on a directory with complete.json.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "code_submission" / "rl_layout" / "experiments"
sys.path.insert(0, str(ARCHIVE))
sys.path.insert(0, str(HERE))

from extended_protocol import cell_seed, digest, generate_cell, run_key, search_seed, task_key  # noqa: E402
from run_eaai_study import atomic_json, eaai_source_hashes  # noqa: E402


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(directory, check_sources=True, output=None):
    from completion_search import LayoutProblem
    from layout_verifier import verify_complete_layout
    import bench

    frozen = read(directory / "protocol.json")
    protocol = frozen["settings"]
    require((directory / "complete.json").exists(), "phase is incomplete")
    require(frozen["protocol_sha256"] == digest(protocol), "protocol digest mismatch")
    if check_sources:
        require(frozen["source_sha256"] == eaai_source_hashes(), "source differs from the frozen study")
    inits, runs = [], []
    expected_inits, expected_runs, manifest = set(), set(), {}
    verified_layouts, max_error = 0, 0.0
    by_reason = {}
    for cell in protocol["cells"]:
        generated = generate_cell(protocol, cell)
        source = bench.verify_witness(generated)
        require(source["all_inside"] and source["max_overlap"] <= 1e-9, "generated packing infeasible")
        for instance in range(cell["instances"]):
            identity = task_key(cell, instance)
            init_path = directory / "initializations" / (identity + ".json")
            expected_inits.add(init_path.name)
            init = read(init_path)
            manifest[str(init_path.relative_to(directory))] = hashlib.sha256(init_path.read_bytes()).hexdigest()
            require(all(init[k] == cell[k] for k in ("geometry", "n", "fill")) and init["instance"] == instance,
                    "initialization identity mismatch: " + identity)
            require(init["protocol_sha256"] == frozen["protocol_sha256"], "initializer protocol mismatch")
            require(init["cell_seed"] == cell_seed(protocol, cell) == generated["seed"], "cell seed mismatch")
            require(abs(init["achieved_fill"] - generated["achieved_fill"][instance]) < 1e-12, "achieved fill mismatch")
            problem = LayoutProblem(generated["inst"].take([instance]))
            require((init["gw"], init["gh"]) == (problem.gw, problem.gh), "plate size mismatch")
            if protocol.get("objective") == "flow":
                # Objectives under the flow (and obstacle) settings are recomputed with the
                # same modules the study used; geometry checks below stay archive-based.
                import flow_objective
                flow = flow_objective.generate_flow(cell["n"], init["flow_seed"], density=protocol["flow"]["density"],
                                                    sigma=protocol["flow"]["sigma"])
                if protocol.get("constraints") == "obstacles":
                    import constraints
                    blocked_mask = np.zeros((problem.canvas, problem.canvas), dtype=bool)
                    for r, c in init["blocked_cells"]:
                        blocked_mask[r, c] = True
                    anchors = [{"x": a["x"], "y": a["y"], "flow": np.array(a["flow"]), "rect": tuple(a["rect"])}
                               for a in init["anchors"]]
                    problem = constraints.ConstrainedFlowLayoutProblem(generated["inst"].take([instance]), flow,
                                                                        blocked_mask, anchors)
                else:
                    problem = flow_objective.FlowLayoutProblem(generated["inst"].take([instance]), flow)
            deviation = abs(float(generated["achieved_fill"][instance]) - cell["fill"])
            if deviation > protocol["fill_tolerance"]:
                require(not init["initialized"] and init.get("reason") == "fill_deviation",
                        "fill deviation not recorded: " + identity)
            else:
                ok, witness, _ = problem.initial_best_contact()
                require(bool(ok) == init["initialized"], "initialization coverage mismatch: " + identity)
            inits.append(init)
            if not init["initialized"]:
                by_reason[init.get("reason")] = by_reason.get(init.get("reason"), 0) + 1
                require(init["initial_layout"] is None and init["initial_J"] is None, "failed init has a layout")
                continue
            require(sorted(map(tuple, witness)) == sorted(map(tuple, init["initial_layout"])),
                    "initial layout differs from the deterministic initializer: " + identity)
            initial_value = problem.score(witness)
            require(abs(initial_value - init["initial_J"]) < 1e-8, "initial objective mismatch")
            verified_layouts += 1
            for method in protocol["methods"]:
                for seed in range(protocol["evaluation_seeds"]):
                    path = directory / "runs" / (run_key(cell, instance, method, seed) + ".json")
                    expected_runs.add(path.name)
                    run = read(path)
                    manifest[str(path.relative_to(directory))] = hashlib.sha256(path.read_bytes()).hexdigest()
                    require(all(run[k] == cell[k] for k in ("geometry", "n", "fill")) and run["instance"] == instance
                            and run["method"] == method and run["evaluation_seed"] == seed, "run identity mismatch")
                    require(run["search_seed"] == search_seed(protocol, cell, instance, method, seed), "search seed mismatch")
                    require(run["protocol_sha256"] == frozen["protocol_sha256"], "run protocol mismatch")
                    require(run["cpu_id"] == init["cpu_id"], "instance used different CPUs")
                    require(abs(run["initial_J"] - initial_value) < 1e-8, "not a common initialization")
                    trace = run["trace"]
                    times = [e["elapsed_s"] for e in trace]
                    values = [e["J"] for e in trace]
                    require(trace and times == sorted(times) and times[0] == 0, "invalid trace chronology")
                    require(np.isfinite(times).all() and np.isfinite(values).all(), "nonfinite trace")
                    require(abs(values[0] - initial_value) < 1e-8, "trace initial value mismatch")
                    require(times[-1] <= protocol["budgets_s"][-1], "late event admitted")
                    require(run["stats"]["actual_search_elapsed_s"] + 1e-6 >= protocol["budgets_s"][-1],
                            "search stopped before the cutoff")
                    require(run["stats"]["final_verification_failures"] == 0, "recorded verifier failure")
                    require([r["budget_s"] for r in run["rows"]] == protocol["budgets_s"], "budget row mismatch")
                    previous = initial_value
                    blocked_cells = init.get("blocked_cells")
                    blocked = None
                    if blocked_cells:
                        blocked = np.zeros((problem.canvas, problem.canvas), dtype=bool)
                        for r, c in blocked_cells:
                            blocked[r, c] = True
                    for row in run["rows"]:
                        check = verify_complete_layout(problem.iw, problem.ih, problem.gw, problem.gh, row["layout"])
                        require(check.valid, "invalid stored layout: " + check.reason)
                        if blocked is not None:
                            occ = np.zeros((problem.canvas, problem.canvas), dtype=np.int32)
                            for i, r, c in row["layout"]:
                                occ[r:r + problem.ih[i], c:c + problem.iw[i]] += 1
                            require(not np.any(occ[blocked]), "stored layout overlaps a fixed obstacle")
                        value = problem.score(row["layout"], already_verified=True)
                        error = abs(value - row["J"])
                        max_error = max(max_error, error)
                        require(error < 1e-8, "stored objective mismatch")
                        eligible = max(e["J"] for e in trace if e["elapsed_s"] <= row["budget_s"])
                        require(abs(value - eligible) < 1e-8, "budget row is not the best eligible event")
                        require(value >= previous - 1e-8, "incumbent regression")
                        require(abs(row["improvement"] - (value - initial_value)) < 1e-8, "raw improvement mismatch")
                        require(abs(row["improvement_pct"] - 100 * (value - initial_value) / initial_value) < 1e-8,
                                "percent improvement mismatch")
                        previous = value
                        verified_layouts += 1
                    runs.append({"method": method, "cpu_wall_ratio": run["cpu_wall_ratio"],
                                 "overrun_s": run["stats"]["overrun_s"],
                                 "late": run["stats"]["late_completed_events"]})
    require({p.name for p in (directory / "runs").glob("*.json")} == expected_runs, "missing or unexpected run files")
    require({p.name for p in (directory / "initializations").glob("*.json")} == expected_inits,
            "missing or unexpected initialization files")
    complete = read(directory / "complete.json")
    require(complete["runs"] == complete["expected_runs"] == len(runs), "completion count mismatch")
    timing = {}
    for method in protocol["methods"]:
        sel = [r for r in runs if r["method"] == method]
        ratios = np.array([r["cpu_wall_ratio"] for r in sel])
        timing[method] = {"runs": len(sel), "cpu_wall_ratio_mean": float(ratios.mean()),
                          "cpu_wall_ratio_min": float(ratios.min()), "cpu_wall_ratio_median": float(np.median(ratios)),
                          "max_overrun_s": float(max(r["overrun_s"] for r in sel)),
                          "late_completed_events": int(sum(r["late"] for r in sel))}
    report = {"passed": True, "attempted_instances": len(inits),
              "initialized_instances": sum(i["initialized"] for i in inits),
              "not_initialized_by_reason": by_reason, "runs": len(runs),
              "verified_saved_layouts": verified_layouts, "max_objective_error": max_error,
              "protocol_sha256": frozen["protocol_sha256"], "sources_checked": check_sources,
              "results_manifest_sha256": digest(manifest), "timing": timing, "files_sha256": manifest}
    atomic_json(Path(output) if output is not None else directory / "audit.json", report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, help="write the new audit here, preserving the archived audit")
    parser.add_argument("--skip-source-check", action="store_true",
                        help="skip comparison with historical implementation hashes; sources_checked is false")
    args = parser.parse_args()
    report = audit(args.directory.resolve(), check_sources=not args.skip_source_check, output=args.out)
    print("AUDIT PASSED: {} attempted, {} initialized ({}), {} runs, {} saved layouts verified, max objective error {:.2e}".format(
        report["attempted_instances"], report["initialized_instances"], report["not_initialized_by_reason"],
        report["runs"], report["verified_saved_layouts"], report["max_objective_error"]))
    for method, t in report["timing"].items():
        print("  {:<20s} runs {:4d}  cpu/wall mean {:.3f} min {:.3f}  max overrun {:.3f} s  late {}".format(
            method, t["runs"], t["cpu_wall_ratio_mean"], t["cpu_wall_ratio_min"], t["max_overrun_s"], t["late_completed_events"]))


if __name__ == "__main__":
    main()
