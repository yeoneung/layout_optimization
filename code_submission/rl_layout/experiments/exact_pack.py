"""
Time-limited CP-SAT decision packing as an initial-witness generator.

Runs in the `layout_exact` environment (ortools; no torch needed since the
suite is rebuilt from coordinates via `bench`).  For each instance of the hard
cells the solver decides pure feasibility, no objective, with a wall-clock
limit; the per-instance solve time gives coverage at any smaller budget too.
Together with `witness_frontier.py` this is the coverage frontier: heuristics
answer in milliseconds, the complete solver certifies feasibility or exhausts
its budget.
"""

import argparse
import json
import time

import numpy as np
from ortools.sat.python import cp_model

import bench

# (suite, tag, split, n, fill, n_inst): kept in sync with witness_frontier.py,
# not imported from it, because that module pulls torch and this script runs in
# the ortools-only environment.
CELLS = [
    ("full", "", "test", 32, 0.85, 20),
    ("full", "", "test", 32, 0.90, 20),
    ("full", "", "test", 32, 0.95, 20),
    ("full", "", "test", 64, 0.80, 20),
    ("full", "", "test", 64, 0.95, 20),
    ("dense", "dense100_v3", "test", 32, 0.90, 100),
    ("dense", "dense100_v3", "test", 32, 0.95, 100),
    ("dense", "dense100_v3", "test", 64, 0.90, 100),
    ("dense", "dense100_v3", "test", 64, 0.95, 100),
]


def pack_once(w, h, gw, gh, limit_s):
    m = cp_model.CpModel()
    xs, ys, ix, iy = [], [], [], []
    for i in range(len(w)):
        x = m.NewIntVar(0, int(gw - w[i]), f"x{i}")
        y = m.NewIntVar(0, int(gh - h[i]), f"y{i}")
        ix.append(m.NewFixedSizeIntervalVar(x, int(w[i]), f"ix{i}"))
        iy.append(m.NewFixedSizeIntervalVar(y, int(h[i]), f"iy{i}"))
        xs.append(x)
        ys.append(y)
    m.AddNoOverlap2D(ix, iy)
    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = float(limit_s)
    s.parameters.num_search_workers = 1
    t0 = time.time()
    status = s.Solve(m)
    wall = time.time() - t0
    ok = status in (cp_model.OPTIMAL, cp_model.FEASIBLE)
    sol = ([(int(s.Value(xs[i])), int(s.Value(ys[i])))
            for i in range(len(w))] if ok else None)
    return ok, wall, sol


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=float, default=10.0)
    p.add_argument("--out", default="exact_pack.json")
    a = p.parse_args()

    rows = []
    for suite, tag, split, n, fill, n_inst in CELLS:
        cell = bench.suite_cell(split, n, fill, n_inst, tag=tag)
        inst = cell["inst"]
        iw = np.rint(inst.w).astype(np.int64)
        ih = np.rint(inst.h).astype(np.int64)
        times, oks = [], []
        for b in range(inst.B):
            ok, wall, sol = pack_once(iw[b], ih[b], int(round(inst.gw[b])),
                                      int(round(inst.gh[b])), a.limit)
            if ok:   # independent overlap/inside check on the solution
                occ = np.zeros((int(round(inst.gh[b])),
                                int(round(inst.gw[b]))), dtype=np.int32)
                for i, (x, y) in enumerate(sol):
                    occ[y:y + ih[b][i], x:x + iw[b][i]] += 1
                if occ.max() > 1:
                    raise RuntimeError("CP-SAT packing overlaps")
            oks.append(ok)
            times.append(wall)
        times = np.array(times)
        oks = np.array(oks)
        row = {"suite": suite, "n": n, "fill": fill, "limit_s": a.limit,
               "coverage": float(oks.mean()),
               "coverage_1s": float((oks & (times <= 1.0)).mean()),
               "mean_s": float(times[oks].mean()) if oks.any() else None,
               "max_s": float(times[oks].max()) if oks.any() else None}
        rows.append(row)
        print(f"{suite:5s} n={n:2d} f={fill:.2f}  cov={row['coverage']:.2f} "
              f"(<=1s: {row['coverage_1s']:.2f})  "
              f"mean {row['mean_s'] or 0:.2f}s max {row['max_s'] or 0:.2f}s")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
