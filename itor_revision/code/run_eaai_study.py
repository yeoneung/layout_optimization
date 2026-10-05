"""Resumable, CPU-pinned study runner for the EAAI track.

Differences from the archive runner: any list of method tokens (archive M0/M1,
ALNS:<config>, and the extended tokens in methods_ext), any phase (smoke,
pilot, train, validation, test), explicit CPU pinning to verified fast cores,
and a per-core speed benchmark stored with the frozen protocol.  The frozen
archive sources are imported unchanged; their hashes plus the EAAI sources are
recorded so a resume cannot mix implementations.

The test phase regenerates the published test instances exactly (same split,
tag, base seed and cell design), so every method can be paired instance by
instance with the published M0, M1 and ALNS results while all timings in one
output root come from this machine.
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

# Single-threaded numerics in every process of the study, including the parent
# that runs the per-core benchmark; must precede the first numpy import.
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "code_submission" / "rl_layout" / "experiments"
sys.path.insert(0, str(ARCHIVE))
sys.path.insert(0, str(HERE))

from extended_protocol import (MAX_GRID, BASE_SEED, TARGET_IMPROVEMENTS_PCT,  # noqa: E402
                               digest, generate_cell, run_key, search_seed, source_hashes,
                               task_key)

DEFAULT_CPUS = "0,2,4,6,8,10,12,14"   # one logical CPU per P-core on the i9-14900KF
PHASE_SPLIT = {"smoke": "val", "pilot": "val", "validation": "val", "train": "train", "test": "test"}


def atomic_json(path, data, attempts=40):
    """Atomic replace with retries: file-sync and antivirus tools on Windows can
    hold a fresh file for a moment, which turns os.replace into PermissionError."""
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"), allow_nan=False)
    for attempt in range(attempts):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(0.25 * (attempt + 1))


def eaai_source_hashes():
    hashes = dict(source_hashes())
    for name in ("methods_ext.py", "learned_methods.py", "flow_objective.py", "constraints.py",
                 "alns_configs.py", "run_eaai_study.py"):
        hashes["EAAI/" + name] = hashlib.sha256((HERE / name).read_bytes()).hexdigest()
    return hashes


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def core_benchmark(cpu, repeats=40):
    import numpy as np
    import psutil
    previous = psutil.Process().cpu_affinity()
    psutil.Process().cpu_affinity([cpu])
    rng = np.random.default_rng(0)
    a = rng.random((300, 300))
    started = time.perf_counter()
    for _ in range(repeats):
        occ = np.zeros((100, 100), dtype=np.int32)
        for k in range(200):
            occ[(k % 90):(k % 90) + 8, ((k * 7) % 90):((k * 7) % 90) + 8] += 1
        s = np.cumsum(np.cumsum(occ, 0), 1)
        a @ a
        np.argsort(-s.ravel(), kind="mergesort")[:32]
    elapsed = time.perf_counter() - started
    psutil.Process().cpu_affinity(previous)
    return elapsed


def make_protocol(args):
    phase = args.phase
    sizes = [int(x) for x in args.sizes.split(",")]
    fills = [float(x) for x in args.fills.split(",")]
    geometries = args.geometries.split(",")
    per_fill = {}
    for item in args.instances.split(","):
        if "=" in item:
            fill, count = item.split("=")
            per_fill[round(float(fill), 2)] = int(count)
        else:
            per_fill["default"] = int(item)
    cells = []
    for geometry in geometries:
        for n in sizes:
            for fill in fills:
                count = per_fill.get(round(fill, 2), per_fill.get("default"))
                if count is None:
                    raise ValueError("no instance count for fill {}".format(fill))
                cells.append({"geometry": geometry, "n": n, "fill": fill, "instances": count,
                              "max_grid": MAX_GRID[n], "avg_area": [9.0, 30.0]})
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    import methods_ext
    if args.alns_grid:
        import alns_configs  # noqa: F401  (registers the grid)
    import repair_baseline
    for method in methods:
        if method.startswith("ALNS:"):
            if method.split(":", 1)[1] not in repair_baseline.CONFIGURATIONS:
                raise ValueError("unknown ALNS configuration " + method + " (use --alns-grid for the extended grid)")
        else:
            methods_ext.parse_method(method)
    if args.constraints != "none" and args.objective != "flow":
        raise ValueError("obstacle constraints are defined for the flow objective")
    cpus = [int(x) for x in args.cpus.split(",")]
    if len(set(cpus)) < args.workers:
        raise ValueError("need at least one distinct CPU per worker")
    budgets = [float(x) for x in args.budgets.split(",")]
    tag = args.tag or ("completion_extension_test" if phase == "test" else "eaai_" + phase)
    return {
        "study": "EAAI track: repair-cap family, switching policies and bandit control",
        "phase": phase, "split": PHASE_SPLIT[phase], "tag": tag, "base_seed": BASE_SEED,
        "cells": cells, "methods": methods, "evaluation_seeds": args.seeds,
        "budgets_s": sorted(budgets), "kappa": 32, "repair_cap": 4, "lns_cap": 16,
        "completion_tries": 1, "workers": args.workers, "cpus": cpus[:args.workers],
        "fill_tolerance": args.fill_tolerance,
        "objective": args.objective,
        "flow": ({"density": 0.30, "sigma": 0.8, "distance": "rectilinear centroid",
                  "seed_rule": "cell_seed * 1000 + instance + 7"} if args.objective == "flow" else None),
        "constraints": args.constraints,
        "obstacles": ({"column_fraction": 0.15, "dock_shape": [2, 3], "dock_density": 0.30, "dock_sigma": 0.8,
                       "seed_rule": "cell_seed * 1000 + instance + 11",
                       "source": "cells free in the withheld source packing"}
                      if args.constraints == "obstacles" else None),
        "alns_grid": bool(args.alns_grid),
        "target_improvements_pct": TARGET_IMPROVEMENTS_PCT,
        "initializer": "verified best-contact; source witness is withheld",
        "timing": "all methods for one instance share one pinned CPU; BLAS and OpenMP threads are one",
        "accounting": "post-initialization search; measured initialization also added for total-time reporting",
        "method_order": "deterministically shuffled method/seed jobs within each instance",
        "uninitialized_instances": "reported as initialization failures; never replaced",
        "fill_accounting": "abort if achieved fill deviates from nominal by more than 0.01",
    }


def freeze_protocol(directory, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists():
        stored = load_json(path)
        if stored["settings"] != protocol or stored["source_sha256"] != eaai_source_hashes():
            raise RuntimeError("protocol or code changed; use a separate output directory")
        return stored
    import importlib.metadata
    import psutil
    benchmark = {str(cpu): core_benchmark(cpu) for cpu in protocol["cpus"]}
    stored = {"settings": protocol, "protocol_sha256": digest(protocol),
              "source_sha256": eaai_source_hashes(), "created_unix": time.time(),
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "processor": platform.processor(),
              "physical_cores": psutil.cpu_count(logical=False),
              "logical_cores": psutil.cpu_count(logical=True),
              "packages": {name: importlib.metadata.version(name) for name in ("numpy", "psutil")},
              "cpu_ids": protocol["cpus"], "core_benchmark_s": benchmark}
    atomic_json(path, stored)
    return stored


def tasks(protocol):
    return [(cell, i) for cell in protocol["cells"] for i in range(cell["instances"])]


def worker(directory, worker_id):
    import numpy as np
    import psutil
    import bench
    from completion_search import LayoutProblem
    from repair_baseline import run_adaptive_repair
    import methods_ext

    frozen = load_json(directory / "protocol.json")
    protocol = frozen["settings"]
    if protocol.get("alns_grid"):
        import alns_configs  # noqa: F401  (registers the extended ALNS grid in this worker)
    process = psutil.Process()
    cpu = frozen["cpu_ids"][worker_id]
    process.cpu_affinity([cpu])
    progress = directory / ("worker_{}.json".format(worker_id))
    results = directory / "runs"
    initializations = directory / "initializations"
    results.mkdir(exist_ok=True)
    initializations.mkdir(exist_ok=True)
    cached = {}
    total_runs = 0
    for ordinal, (cell, instance) in enumerate(tasks(protocol)):
        if ordinal % protocol["workers"] != worker_id:
            continue
        if (directory.parent / "STOP").exists():
            return 2
        if frozen["source_sha256"] != eaai_source_hashes():
            raise RuntimeError("source changed during the study")
        identity = (cell["geometry"], cell["n"], cell["fill"])
        if identity not in cached:
            generated = generate_cell(protocol, cell)
            check = bench.verify_witness(generated)
            if not check["all_inside"] or check["max_overlap"] > 1e-9:
                raise RuntimeError("invalid generated source packing")
            cached[identity] = generated
        generated = cached[identity]
        flow_seed = None
        obstacle_record = None
        if protocol.get("objective") == "flow":
            import flow_objective
            flow_seed = int(generated["seed"]) * 1000 + instance + 7
            flow = flow_objective.generate_flow(cell["n"], flow_seed, density=protocol["flow"]["density"],
                                                sigma=protocol["flow"]["sigma"])
            single = generated["inst"].take([instance])
            if protocol.get("constraints") == "obstacles":
                import constraints
                constraints.activate()
                settings = protocol["obstacles"]
                canvas = max(int(round(single.gw[0])), int(round(single.gh[0])))
                obstacle_seed = int(generated["seed"]) * 1000 + instance + 11
                blocked, anchors, info = constraints.make_constraints(
                    single, generated["witness"][instance], canvas, obstacle_seed,
                    column_fraction=settings["column_fraction"], dock_shape=tuple(settings["dock_shape"]),
                    dock_density=settings["dock_density"], dock_sigma=settings["dock_sigma"])
                problem = constraints.ConstrainedFlowLayoutProblem(single, flow, blocked, anchors)
                obstacle_record = {"obstacle_seed": obstacle_seed,
                                   "blocked_cells": [[int(r), int(c)] for r, c in np.argwhere(blocked)],
                                   "anchors": [{"x": float(a["x"]), "y": float(a["y"]),
                                                "flow": [float(v) for v in a["flow"]],
                                                "rect": [int(v) for v in a["rect"]]} for a in anchors],
                                   "obstacle_info": {k: (v if not isinstance(v, tuple) else list(v))
                                                     for k, v in info.items()}}
            else:
                flow_objective.activate()
                problem = flow_objective.FlowLayoutProblem(single, flow)
        else:
            problem = LayoutProblem(generated["inst"].take([instance]))
        init_path = initializations / (task_key(cell, instance) + ".json")
        if init_path.exists():
            init = load_json(init_path)
            witness = init["initial_layout"]
        else:
            # An instance whose achieved fill deviates from the nominal fill by
            # more than the tolerance is recorded as not initialized (reason
            # fill_deviation) and stays in the attempted count; it is never
            # replaced by another draw.
            deviation = abs(float(generated["achieved_fill"][instance]) - cell["fill"])
            reason = None
            if deviation > protocol["fill_tolerance"]:
                ok, witness, elapsed, value = False, None, 0.0, None
                reason = "fill_deviation"
            else:
                ok, witness, elapsed = problem.initial_best_contact()
                value = None
                if ok:
                    scored = time.perf_counter()
                    value = problem.score(witness)
                    elapsed += time.perf_counter() - scored
                else:
                    reason = "initializer_failed"
            init = {"geometry": cell["geometry"], "n": cell["n"], "fill": cell["fill"],
                    "instance": instance, "initialized": bool(ok), "reason": reason,
                    "initialization_s": elapsed, "initial_J": value, "initial_layout": witness,
                    "cell_seed": int(generated["seed"]),
                    "achieved_fill": float(generated["achieved_fill"][instance]),
                    "fill_deviation": float(deviation),
                    "gw": problem.gw, "gh": problem.gh,
                    "mean_facility_area": float(np.mean(problem.iw * problem.ih)),
                    "cpu_id": cpu, "protocol_sha256": frozen["protocol_sha256"]}
            if flow_seed is not None:
                init.update({"objective": "flow", "flow_seed": flow_seed,
                             "flow_constant": float(problem.objective.constant[0]),
                             "initial_cost": (problem.cost_of(witness) if ok else None)})
            if obstacle_record is not None:
                init.update(obstacle_record)
            atomic_json(init_path, init)
        if init["protocol_sha256"] != frozen["protocol_sha256"]:
            raise RuntimeError("initialization belongs to another protocol")
        if not init["initialized"]:
            print(task_key(cell, instance), "not initialized:", init.get("reason"), flush=True)
            continue
        if abs(problem.score(witness) - init["initial_J"]) > 1e-8:
            raise RuntimeError("stored initial objective mismatch")
        jobs = [(method, seed) for method in protocol["methods"]
                for seed in range(protocol["evaluation_seeds"])]
        order_rng = np.random.default_rng(search_seed(protocol, cell, instance, "order", 0))
        order_rng.shuffle(jobs)
        for method, seed in jobs:
            if (directory.parent / "STOP").exists():
                return 2
            key = run_key(cell, instance, method, seed)
            destination = results / (key + ".json")
            if destination.exists():
                if load_json(destination)["protocol_sha256"] != frozen["protocol_sha256"]:
                    raise RuntimeError("result belongs to another protocol")
                continue
            seed_value = search_seed(protocol, cell, instance, method, seed)
            atomic_json(progress, {"status": "running", "worker": worker_id, "pid": os.getpid(),
                                   "cpu_id": cpu, "run": key, "started_unix": time.time(),
                                   "completed_here": total_runs})
            cpu_started = process.cpu_times()
            wall_started = time.perf_counter()
            if method.startswith("ALNS:"):
                output = run_adaptive_repair(problem, witness, protocol["budgets_s"], seed_value,
                                             method.split(":", 1)[1], initial_value=init["initial_J"])
            elif method.startswith("LRN:"):
                import learned_methods
                output = learned_methods.run_learned_anytime(
                    problem, [tuple(item) for item in witness], method.split(":", 1)[1],
                    protocol["budgets_s"], seed=seed_value, initial_value=init["initial_J"])
            else:
                output = methods_ext.run_anytime_ext(problem, witness, method, protocol["budgets_s"],
                                                     seed=seed_value, kappa=protocol["kappa"],
                                                     completion_tries=protocol["completion_tries"],
                                                     initial_value=init["initial_J"])
            wall_elapsed = time.perf_counter() - wall_started
            cpu_finished = process.cpu_times()
            cpu_elapsed = (cpu_finished.user + cpu_finished.system
                           - cpu_started.user - cpu_started.system)
            output.update({"method": method, "geometry": cell["geometry"], "n": cell["n"],
                           "fill": cell["fill"], "instance": instance, "evaluation_seed": seed,
                           "search_seed": seed_value, "initialization_s": init["initialization_s"],
                           "cpu_id": cpu, "process_cpu_s": cpu_elapsed, "call_wall_s": wall_elapsed,
                           "cpu_wall_ratio": cpu_elapsed / max(wall_elapsed, 1e-12),
                           "numpy": np.__version__, "protocol_sha256": frozen["protocol_sha256"]})
            atomic_json(destination, output)
            total_runs += 1
            print(key, "J={:.4f} imp={:.2f}%".format(output["rows"][-1]["J"],
                                                     output["rows"][-1]["improvement_pct"]), flush=True)
    atomic_json(progress, {"status": "complete", "worker": worker_id, "pid": os.getpid(),
                           "cpu_id": cpu, "finished_unix": time.time(), "completed_here": total_runs})
    return 0


def run_phase(root, protocol):
    directory = root / protocol["phase"]
    frozen = freeze_protocol(directory, protocol)
    if (directory / "complete.json").exists():
        return directory
    children, logs = [], []
    environment = os.environ.copy()
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        environment[name] = "1"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PYTHONUTF8"] = "1"
    try:
        for index in range(protocol["workers"]):
            log = (directory / ("worker_{}.log".format(index))).open("a", encoding="utf-8")
            logs.append(log)
            command = [sys.executable, "-u", str(Path(__file__).resolve()), "--out", str(root),
                       "--worker", str(index), "--phase", protocol["phase"]]
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             env=environment, cwd=str(HERE),
                                             creationflags=(subprocess.CREATE_NO_WINDOW
                                                            if os.name == "nt" else 0)))
        while any(child.poll() is None for child in children):
            failed = [child.returncode for child in children
                      if child.returncode is not None and child.returncode != 0]
            if failed:
                raise RuntimeError("worker failed; inspect worker logs: {}".format(failed))
            atomic_json(root / "status.json", {
                "phase": protocol["phase"], "status": "running", "updated_unix": time.time(),
                "pid": os.getpid(), "worker_pids": [child.pid for child in children],
                "completed_runs": len(list((directory / "runs").glob("*.json"))),
                "expected_runs_upper": len(tasks(protocol)) * len(protocol["methods"]) * protocol["evaluation_seeds"],
                "completed_initializations": len(list((directory / "initializations").glob("*.json"))),
                "attempted_instances": len(tasks(protocol)), "cpu_ids": frozen["cpu_ids"]})
            time.sleep(5)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("one or more workers did not finish")
        initializations = [load_json(p) for p in sorted((directory / "initializations").glob("*.json"))]
        if len(initializations) != len(tasks(protocol)):
            raise RuntimeError("incomplete initialization record")
        expected = (sum(row["initialized"] for row in initializations) * len(protocol["methods"])
                    * protocol["evaluation_seeds"])
        actual = len(list((directory / "runs").glob("*.json")))
        if expected != actual:
            raise RuntimeError("missing runs: expected {}, got {}".format(expected, actual))
        atomic_json(directory / "complete.json", {
            "runs": actual, "expected_runs": expected,
            "initialized": sum(row["initialized"] for row in initializations),
            "attempted": len(initializations), "completed_unix": time.time(),
            "protocol_sha256": frozen["protocol_sha256"]})
        return directory
    finally:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait()
        for log in logs:
            log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", choices=sorted(PHASE_SPLIT), default="pilot")
    parser.add_argument("--methods", default="M0,M1")
    parser.add_argument("--sizes", default="64,128,256")
    parser.add_argument("--fills", default="0.90,0.95")
    parser.add_argument("--geometries", default="guillotine,nonslicing")
    parser.add_argument("--instances", default="3", help="count, or per-fill list like 0.90=10,0.95=20")
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--budgets", default="0,0.1,0.3,1,3,10,30,60")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cpus", default=DEFAULT_CPUS)
    parser.add_argument("--tag")
    parser.add_argument("--fill-tolerance", dest="fill_tolerance", type=float, default=0.01,
                        help="instances whose achieved fill deviates more are recorded as not initialized")
    parser.add_argument("--objective", choices=["preference", "flow"], default="preference",
                        help="preference: the published objective; flow: material-handling cost (flow_objective.py)")
    parser.add_argument("--constraints", choices=["none", "obstacles"], default="none",
                        help="obstacles: fixed blocked cells plus a pre-placed dock with flows (constraints.py)")
    parser.add_argument("--alns-grid", dest="alns_grid", action="store_true",
                        help="register the extended ALNS configuration grid (alns_configs.py)")
    parser.add_argument("--worker", type=int)
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker is not None:
        return worker(root / args.phase, args.worker)
    root.mkdir(parents=True, exist_ok=True)
    import psutil
    lock = root / "runner.lock"
    if lock.exists():
        owner = load_json(lock)
        try:
            process = psutil.Process(owner["pid"])
            alive = abs(process.create_time() - owner["created"]) < 0.01
        except psutil.NoSuchProcess:
            alive = False
        if alive:
            raise RuntimeError("another study runner owns this output directory")
        lock.unlink()
    with lock.open("x", encoding="utf-8") as handle:
        json.dump({"pid": os.getpid(), "created": psutil.Process().create_time()}, handle)
    try:
        protocol = make_protocol(args)
        run_phase(root, protocol)
        atomic_json(root / "status.json", {"status": "complete", "phase": args.phase,
                                           "finished_unix": time.time(), "pid": os.getpid()})
        print("PHASE COMPLETE:", args.phase, flush=True)
    except BaseException as error:
        atomic_json(root / "status.json", {"status": "failed", "phase": args.phase,
                                           "updated_unix": time.time(), "pid": os.getpid(),
                                           "error": repr(error)})
        raise
    finally:
        lock.unlink()
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print("STUDY FAILED:", repr(error), flush=True)
        raise
