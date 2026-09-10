"""Resumable, CPU-pinned validation and test runs for the extended comparison.

Each result is saved atomically after one run. A frozen protocol and source
hashes prevent silently combining different implementations on resume. The
pipeline fixes the ALNS configuration using validation before it creates test
tasks. Historical result files are never overwritten.
"""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import importlib.metadata
import subprocess
import sys
import time

from extended_protocol import (atomic_json, cell_seed, digest, generate_cell,
                               make_protocol, run_key, search_seed, source_hashes,
                               task_key)


def physical_cpu_ids(workers):
    import psutil
    allowed = psutil.Process().cpu_affinity()
    cores = []
    if os.name == "nt":
        class Details(ctypes.Union):
            _fields_ = [("reserved", ctypes.c_ulonglong * 2)]

        class ProcessorInfo(ctypes.Structure):
            _fields_ = [("mask", ctypes.c_size_t), ("relationship", ctypes.c_int),
                        ("details", Details)]

        size = ctypes.c_ulong(0)
        function = ctypes.windll.kernel32.GetLogicalProcessorInformation
        function(None, ctypes.byref(size))
        buffer = ctypes.create_string_buffer(size.value)
        if not function(buffer, ctypes.byref(size)):
            raise OSError("cannot read physical CPU topology")
        for offset in range(0, size.value, ctypes.sizeof(ProcessorInfo)):
            entry = ProcessorInfo.from_buffer_copy(buffer.raw, offset)
            if entry.relationship == 0:
                members = [cpu for cpu in allowed if entry.mask & (1 << cpu)]
                if members:
                    cores.append(members[0])
    else:
        for cpu in allowed:
            topology = Path("/sys/devices/system/cpu/cpu{}/topology/thread_siblings_list".format(cpu))
            if topology.exists():
                first = int(topology.read_text().strip().split(",")[0].split("-")[0])
                if first == cpu:
                    cores.append(cpu)
    if len(cores) < workers:
        raise RuntimeError("insufficient verified physical cores for {} workers".format(workers))
    # Spread the four workers across physical cores, retaining others for the UI.
    preferred = cores[1::2] + cores[::2]
    return preferred[:workers]


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def freeze_protocol(directory, protocol):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "protocol.json"
    if path.exists():
        stored = load_json(path)
        if stored["settings"] != protocol or stored["source_sha256"] != source_hashes():
            raise RuntimeError("protocol or code changed; use a separate output directory")
        return stored
    stored = {"settings": protocol, "protocol_sha256": digest(protocol),
              "source_sha256": source_hashes(), "created_unix": time.time(),
              "python": sys.version, "executable": sys.executable,
              "platform": platform.platform(), "processor": platform.processor(),
              "packages": {name: importlib.metadata.version(name)
                           for name in ("numpy", "torch", "psutil")},
              "cpu_ids": physical_cpu_ids(protocol["workers"])}
    atomic_json(path, stored)
    return stored


def tasks(protocol):
    return [(cell, i) for cell in protocol["cells"] for i in range(cell["instances"])]


def worker(directory, worker_id):
    import numpy as np
    import psutil
    import bench
    from completion_search import LayoutProblem, run_anytime
    from repair_baseline import run_adaptive_repair

    frozen = load_json(directory / "protocol.json")
    protocol = frozen["settings"]
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
        if frozen["source_sha256"] != source_hashes():
            raise RuntimeError("source changed during the study")
        cell_identity = (cell["geometry"], cell["n"], cell["fill"])
        if cell_identity not in cached:
            generated = generate_cell(protocol, cell)
            check = bench.verify_witness(generated)
            if not check["all_inside"] or check["max_overlap"] > 1e-9:
                raise RuntimeError("invalid generated source packing")
            if abs(generated["achieved_fill"] - cell["fill"]).max() > 0.01:
                raise RuntimeError("generated fill deviates from the specified tolerance")
            cached[cell_identity] = generated
        generated = cached[cell_identity]
        problem = LayoutProblem(generated["inst"].take([instance]))
        init_path = initializations / (task_key(cell, instance) + ".json")
        if init_path.exists():
            init = load_json(init_path)
            witness = init["initial_layout"]
        else:
            ok, witness, elapsed = problem.initial_best_contact()
            value = None
            if ok:
                scored = time.perf_counter()
                value = problem.score(witness)
                elapsed += time.perf_counter() - scored
            init = {"geometry": cell["geometry"], "n": cell["n"], "fill": cell["fill"],
                    "instance": instance, "initialized": bool(ok),
                    "initialization_s": elapsed, "initial_J": value,
                    "initial_layout": witness, "cell_seed": int(generated["seed"]),
                    "achieved_fill": float(generated["achieved_fill"][instance]),
                    "gw": problem.gw, "gh": problem.gh,
                    "mean_facility_area": float(np.mean(problem.iw * problem.ih)),
                    "cpu_id": cpu, "protocol_sha256": frozen["protocol_sha256"]}
            atomic_json(init_path, init)
        if init["protocol_sha256"] != frozen["protocol_sha256"]:
            raise RuntimeError("initialization belongs to another protocol")
        if not init["initialized"]:
            print(task_key(cell, instance), "initialization failed", flush=True)
            continue
        # Verify a resumed initializer and its objective independently.
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
                existing = load_json(destination)
                if existing["protocol_sha256"] != frozen["protocol_sha256"]:
                    raise RuntimeError("result belongs to another protocol")
                continue
            seed_value = search_seed(protocol, cell, instance, method, seed)
            atomic_json(progress, {"status": "running", "worker": worker_id,
                                   "pid": os.getpid(), "cpu_id": cpu, "run": key,
                                   "started_unix": time.time(), "completed_here": total_runs})
            cpu_started = process.cpu_times()
            wall_started = time.perf_counter()
            if method.startswith("ALNS:"):
                output = run_adaptive_repair(problem, witness, protocol["budgets_s"],
                                             seed_value, method.split(":", 1)[1],
                                             initial_value=init["initial_J"])
            else:
                output = run_anytime(problem, witness, method, protocol["budgets_s"],
                                      seed=seed_value, kappa=protocol["kappa"],
                                      repair_cap=protocol["repair_cap"],
                                      lns_cap=protocol["lns_cap"],
                                      completion_tries=protocol["completion_tries"],
                                      initial_value=init["initial_J"])
            wall_elapsed = time.perf_counter() - wall_started
            cpu_finished = process.cpu_times()
            cpu_elapsed = (cpu_finished.user + cpu_finished.system
                           - cpu_started.user - cpu_started.system)
            output.update({"method": method, "geometry": cell["geometry"], "n": cell["n"],
                           "fill": cell["fill"], "instance": instance,
                           "evaluation_seed": seed, "search_seed": seed_value,
                           "initialization_s": init["initialization_s"],
                           "cpu_id": cpu, "process_cpu_s": cpu_elapsed,
                           "call_wall_s": wall_elapsed,
                           "cpu_wall_ratio": cpu_elapsed / max(wall_elapsed, 1e-12),
                           "numpy": np.__version__,
                           "protocol_sha256": frozen["protocol_sha256"]})
            atomic_json(destination, output)
            total_runs += 1
            print(key, "J={:.4f}".format(output["rows"][-1]["J"]), flush=True)
    atomic_json(progress, {"status": "complete", "worker": worker_id,
                           "pid": os.getpid(), "cpu_id": cpu,
                           "finished_unix": time.time(), "completed_here": total_runs})
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
            command = [sys.executable, "-u", str(Path(__file__).resolve()),
                       "--out", str(root), "--worker", str(index),
                       "--phase", protocol["phase"]]
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                             env=environment, cwd=str(Path(__file__).resolve().parent),
                                             creationflags=(subprocess.CREATE_NO_WINDOW
                                                            if os.name == "nt" else 0)))
        while any(child.poll() is None for child in children):
            failed = [child.returncode for child in children
                      if child.returncode is not None and child.returncode != 0]
            if failed:
                raise RuntimeError("worker failed; inspect worker logs: {}".format(failed))
            atomic_json(root / "status.json", {"phase": protocol["phase"], "status": "running",
                        "updated_unix": time.time(), "pid": os.getpid(),
                        "worker_pids": [child.pid for child in children],
                        "completed_runs": len(list((directory / "runs").glob("*.json"))),
                        "completed_initializations": len(list((directory / "initializations").glob("*.json"))),
                        "attempted_instances": len(tasks(protocol)),
                        "cpu_ids": frozen["cpu_ids"]})
            time.sleep(5)
        if any(child.returncode != 0 for child in children):
            raise RuntimeError("one or more workers did not finish")
        initializations = [load_json(path) for path in sorted((directory / "initializations").glob("*.json"))]
        if len(initializations) != len(tasks(protocol)):
            raise RuntimeError("incomplete initialization record")
        expected = sum(row["initialized"] for row in initializations) * len(protocol["methods"]) * protocol["evaluation_seeds"]
        actual = len(list((directory / "runs").glob("*.json")))
        if expected != actual:
            raise RuntimeError("missing runs: expected {}, got {}".format(expected, actual))
        atomic_json(directory / "complete.json", {"runs": actual, "expected_runs": expected,
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


def select_configuration(directory):
    from collections import defaultdict
    import numpy as np
    by_instance = defaultdict(list)
    for path in sorted((directory / "runs").glob("*.json")):
        run = load_json(path)
        key = (run["method"], run["geometry"], run["n"], run["fill"], run["instance"])
        by_instance[key].append(run["rows"][-1]["improvement_pct"])
    methods = defaultdict(list)
    for key, values in by_instance.items():
        methods[key[0]].append(float(np.mean(values)))
    scores = {method: float(np.mean(values)) for method, values in methods.items()}
    candidates = sorted(method for method in scores if method.startswith("ALNS:"))
    if not candidates:
        raise RuntimeError("no initialized validation cases")
    chosen = max(candidates, key=lambda method: scores[method])
    report = {"selected_configuration": chosen.split(":", 1)[1],
              "selection_metric": "mean within-instance seed-averaged improvement percent at 10 seconds",
              "scores": scores, "instances_per_method": {k: len(v) for k, v in methods.items()},
              "selected_unix": time.time(), "test_results_used": False}
    atomic_json(directory.parent / "selection.json", report)
    print("VALIDATION SELECTION", json.dumps(report), flush=True)
    return report["selected_configuration"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--phase", choices=["validation", "test", "smoke"], default="validation")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument("--configuration")
    args = parser.parse_args()
    root = args.out.resolve()
    if args.worker is not None:
        return worker(root / args.phase, args.worker)
    root.mkdir(parents=True, exist_ok=True)
    # Exclusive ownership prevents two resumptions from writing the same shard.
    # An interrupted process leaves a lock that can be reclaimed after checking
    # that its PID and creation time no longer identify a live process.
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
        if args.pipeline:
            validation = run_phase(root, make_protocol("validation", workers=args.workers))
            selected = select_configuration(validation)
            run_phase(root, make_protocol("test", selected, workers=args.workers))
        else:
            directory = run_phase(root, make_protocol(args.phase, args.configuration, args.workers))
            if args.phase == "validation":
                select_configuration(directory)
        atomic_json(root / "status.json", {"status": "complete", "phase": args.phase,
                                           "finished_unix": time.time(), "pid": os.getpid()})
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
