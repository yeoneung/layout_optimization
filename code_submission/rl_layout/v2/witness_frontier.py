"""
Coverage of alternative initial-witness generators on the hard cells.

Certified coverage equals the witness generator's success rate at the empty
plate, so the generator, not the decoder, is the component to improve.  The
deterministic baseline of the paper is already bottom-left-fill in decreasing
area (attempt 0 of `FirstFitWitness.certify` scans row-major, i.e. bottom-left,
over a largest-first order).  This script prices the obvious alternatives a
packing reader would ask for, on every cell of the study whose coverage is
below 1: bottom-left-fill under three other sort keys, a best-contact rule
(most shared boundary with placed rectangles and walls, the skyline-style
waste heuristic for fixed plates), and the randomized-restart generator at
effort 128.  `exact_pack.py` adds the time-limited CP-SAT decision packer for
the same cells; together they give the coverage frontier.

Every generator's witness is verified with the same deterministic checker
before it is counted.
"""

import argparse
import json
import time

import numpy as np

import bench
from certified_decode import FirstFitWitness

# (suite, tag, split, n, fill, n_inst): every study cell whose baseline
# coverage is below 1 at ANY effort.  In three of them (full 32/0.85, full
# 64/0.80, dense 64/0.90) the effort-1 gap already closes by effort 8 to 128;
# they are included so the file is complete, and the paper's table shows the
# six cells that remain below 1 at effort 128.
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


def blf(gen, order):
    """Bottom-left fill in the given order; returns (ok, witness)."""
    work = np.zeros((gen.GH, gen.GW), dtype=np.int32)
    wit = []
    for i in order:
        m = gen._legal(work, i)
        idx = np.argwhere(m)
        if not len(idx):
            return False, None
        r, c = int(idx[0][0]), int(idx[0][1])
        work[r:r + int(gen.ih[i]), c:c + int(gen.iw[i])] += 1
        wit.append((int(i), r, c))
    return True, wit


def best_contact(gen, order):
    """Largest shared boundary with placed rectangles and plate walls."""
    work = np.zeros((gen.GH, gen.GW), dtype=np.int32)
    wit = []
    for i in order:
        m = gen._legal(work, i)
        if not m.any():
            return False, None
        w, h = int(gen.iw[i]), int(gen.ih[i])
        occ = (work > 0).astype(np.int64)
        # padded prefix sums along each axis for side-strip occupancy counts
        C = np.zeros((gen.GH + 1, gen.GW + 2), dtype=np.int64)
        C[1:, 1:-1] = occ.cumsum(axis=0)
        R = np.zeros((gen.GH + 2, gen.GW + 1), dtype=np.int64)
        R[1:-1, 1:] = occ.cumsum(axis=1)
        rows = np.arange(gen.GH)[:, None]
        cols = np.arange(gen.GW)[None, :]
        r1 = np.minimum(rows + h, gen.GH)
        c1 = np.minimum(cols + w, gen.GW)
        left = C[r1, cols] - C[rows, cols]              # column c-1 (shifted pad)
        right = C[r1, np.minimum(c1 + 1, gen.GW + 1)] - \
            C[rows, np.minimum(c1 + 1, gen.GW + 1)]
        bottom = R[rows, c1] - R[rows, cols]            # row r-1 (shifted pad)
        top = R[np.minimum(r1 + 1, gen.GH + 1), c1] - \
            R[np.minimum(r1 + 1, gen.GH + 1), cols]
        contact = (left + right + bottom + top
                   + np.where(cols == 0, h, 0)
                   + np.where(cols + w == gen.gw, h, 0)
                   + np.where(rows == 0, w, 0)
                   + np.where(rows + h == gen.gh, w, 0))
        score = np.where(m, contact, -1)
        flat = int(score.argmax())                       # ties: bottom-left
        r, c = flat // gen.GW, flat % gen.GW
        work[r:r + h, c:c + w] += 1
        wit.append((int(i), r, c))
    return True, wit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--out", default="witness_frontier.json")
    a = p.parse_args()

    keys = {
        "blf area (paper baseline)": lambda W, H: -(W * H),
        "blf height": lambda W, H: -H,
        "blf width": lambda W, H: -W,
        "blf perimeter": lambda W, H: -(W + H),
        "best contact, area order": lambda W, H: -(W * H),
        "randomized restarts x128": lambda W, H: -(W * H),
    }
    rows = []
    for suite, tag, split, n, fill, n_inst in CELLS:
        cell = bench.suite_cell(split, n, fill, n_inst, tag=tag)
        inst = cell["inst"]
        iw = np.rint(inst.w).astype(np.int64)
        ih = np.rint(inst.h).astype(np.int64)
        gens = [FirstFitWitness(iw[b], ih[b], a.canvas, a.canvas,
                                int(round(inst.gw[b])), int(round(inst.gh[b])))
                for b in range(inst.B)]
        for name, key in keys.items():
            rng = np.random.default_rng(0)
            t0 = time.time()
            n_ok = 0
            for b, gen in enumerate(gens):
                order = sorted(range(inst.n),
                               key=lambda i: key(iw[b][i], ih[b][i]))
                if name.startswith("randomized"):
                    ok, wit = gen.certify(
                        np.zeros((a.canvas, a.canvas), dtype=np.int32),
                        list(range(inst.n)), tries=128, rng=rng)
                elif name.startswith("best contact"):
                    ok, wit = best_contact(gen, order)
                else:
                    ok, wit = blf(gen, order)
                if ok:
                    if not gen.verify(np.zeros((a.canvas, a.canvas),
                                               dtype=np.int32),
                                      list(range(inst.n)), wit):
                        raise RuntimeError(f"{name}: invalid witness")
                    n_ok += 1
            rows.append({"suite": suite, "n": n, "fill": fill,
                         "generator": name, "coverage": n_ok / inst.B,
                         "ms_per_instance":
                             1000.0 * (time.time() - t0) / inst.B})
            print(f"{suite:5s} n={n:2d} f={fill:.2f} {name:28s} "
                  f"{n_ok:3d}/{inst.B:3d} = {n_ok / inst.B:.2f} "
                  f"({rows[-1]['ms_per_instance']:6.1f} ms)")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
