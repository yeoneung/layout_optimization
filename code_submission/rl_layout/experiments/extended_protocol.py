"""Frozen settings shared by the extended-study runner and independent audit."""

import hashlib
import json
from pathlib import Path


MAX_GRID = {32: 44, 64: 44, 128: 72, 256: 100}
BASE_SEED = 20260812
TARGET_IMPROVEMENTS_PCT = [1.0, 5.0, 10.0]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def make_protocol(phase, selected_configuration=None, workers=4):
    if phase not in ("validation", "test", "smoke"):
        raise ValueError("unknown study phase")
    validation = phase == "validation"
    rooms = [64, 128] if validation else [32, 64, 128, 256]
    cells = []
    for geometry in ["guillotine", "nonslicing"]:
        for n in rooms:
            for fill in [0.90, 0.95]:
                count = 2 if validation else (20 if fill == 0.95 else 10)
                if phase == "smoke":
                    count = 1
                cells.append({"geometry": geometry, "n": n, "fill": fill,
                              "instances": count, "max_grid": MAX_GRID[n],
                              "avg_area": [9.0, 30.0]})
    methods = (["ALNS:small_beam", "ALNS:medium_beam", "ALNS:wide_beam",
                "ALNS:large_beam", "B2", "B2S"] if validation
               else ["M0", "M1", "ALNS:" + (selected_configuration or "small_beam")])
    if phase == "test" and selected_configuration is None:
        raise ValueError("test requires a validation-selected ALNS configuration")
    return {
        "study": "long-horizon and larger-layout completion comparison",
        "phase": phase, "split": "val" if validation or phase == "smoke" else "test",
        "tag": "completion_extension_" + phase,
        "base_seed": BASE_SEED, "cells": cells, "methods": methods,
        "evaluation_seeds": 2 if validation else (1 if phase == "smoke" else 3),
        "budgets_s": ([1.0, 10.0] if validation else
                      ([0.0, 0.02] if phase == "smoke" else
                       [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0])),
        "kappa": 32, "repair_cap": 4, "lns_cap": 16, "completion_tries": 1,
        "workers": workers, "target_improvements_pct": TARGET_IMPROVEMENTS_PCT,
        "primary_contrasts": "M1 minus M0 and ALNS at 10 and 60 seconds, by size and fill",
        "selection": "highest validation mean percent improvement at 10 seconds, seeds averaged within instance",
        "initializer": "verified best-contact; source witness is withheld",
        "timing": "all methods for one instance share one pinned CPU; BLAS and OpenMP threads are one",
        "accounting": "post-initialization search; measured initialization also added for total-time reporting",
        "method_order": "deterministically shuffled method/seed jobs within each instance",
        "source_packing": "verified, never passed to a search method",
        "fill_accounting": "nominal fill is approximate on the integer lattice; every achieved fill is stored; abort if absolute deviation exceeds 0.01",
        "uninitialized_instances": "reported as initialization failures; never replaced or omitted from coverage",
        "non_hits": "time-to-target is right-censored at the run cutoff; do not average successful runs alone",
        "historical_results": "stored separately; short-budget results are reread from these new long runs",
    }


def cell_seed(protocol, cell):
    tag = protocol["tag"] + "_" + cell["geometry"]
    split_offset = {"train": 0, "val": 4000000, "test": 8000000}[protocol["split"]]
    return (protocol["base_seed"] + split_offset + cell["n"] * 10000
            + int(round(cell["fill"] * 100))
            + sum(ord(c) for c in tag) * 1000003)


def task_key(cell, instance):
    return "{}_n{}_f{}_i{:03d}".format(cell["geometry"], cell["n"],
                                      int(round(cell["fill"] * 100)), instance)


def run_key(cell, instance, method, seed):
    return task_key(cell, instance) + "_" + method.replace(":", "_") + "_s" + str(seed)


def search_seed(protocol, cell, instance, method, evaluation_seed):
    identity = [cell_seed(protocol, cell), instance, method, evaluation_seed]
    return int(digest(identity)[:12], 16)


def generate_cell(protocol, cell):
    import bench
    return bench.suite_cell(protocol["split"], cell["n"], cell["fill"],
                            cell["instances"], base_seed=protocol["base_seed"],
                            tag=protocol["tag"] + "_" + cell["geometry"],
                            geometry=cell["geometry"], max_grid=cell["max_grid"],
                            avg_area=tuple(cell["avg_area"]))


def source_hashes():
    root = Path(__file__).resolve().parent
    files = ["completion_search.py", "repair_baseline.py", "extended_protocol.py",
             "run_extended_study.py", "bench.py", "core.py", "greedy_construct.py",
             "certified_decode.py", "witness_frontier.py", "layout_verifier.py"]
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in files}


def atomic_json(path, data):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, separators=(",", ":"), allow_nan=False)
    temporary.replace(path)
