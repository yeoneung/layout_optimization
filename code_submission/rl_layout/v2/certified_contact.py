"""
End-to-end certified decoding with the best-contact initial witness.

`witness_frontier.py` shows the best-contact generator lifting empty-plate
coverage to 0.95 or higher in every hard cell.  This script closes the loop:
it runs the witness-certified greedy decoder (`certified_greedy.decode_cell`)
with that generator as the initial witness on the dense cells, so the coverage
gain is demonstrated inside the actual mechanism rather than argued from the
theorem.  Step-level certificates and the carried-witness fallback are
unchanged; every returned layout is re-verified feasible.
"""

import json

import numpy as np

import bench
import core
from certified_greedy import decode_cell

CELLS = [
    ("full", "", "test", 32, 0.95, 20),
    ("full", "", "test", 64, 0.95, 20),
    ("dense", "dense100_v3", "test", 32, 0.90, 100),
    ("dense", "dense100_v3", "test", 32, 0.95, 100),
    ("dense", "dense100_v3", "test", 64, 0.95, 100),
]


def main():
    rows = []
    for suite, tag, split, n, fill, n_inst in CELLS:
        cell = bench.suite_cell(split, n, fill, n_inst, tag=tag)
        obj = core.BatchObjective(cell["inst"])
        res = decode_cell(cell, effort=1, init="contact")
        ret = res["certified"]
        tot, comp = obj.evaluate(res["x"], res["y"], components=True)
        if np.any(ret & ~(comp["overlap_area"] <= 1e-9)):
            raise RuntimeError("infeasible certified return")
        row = {"suite": suite, "n": n, "fill": fill,
               "returned": int(ret.sum()), "instances": n_inst,
               "coverage": float(ret.mean()),
               "mean_J": float(tot[ret].mean()) if ret.any() else None,
               "witness_used": res["witness_used"], "rejects": res["rejects"],
               "fallback_share": (res["witness_used"] / (ret.sum() * n)
                                  if ret.any() else None),
               "ms_per_instance": 1000.0 * res["wall_time"] / n_inst}
        rows.append(row)
        print(f"{suite:5s} n={n:2d} f={fill:.2f}  cov={row['coverage']:.2f} "
              f"({row['returned']}/{n_inst})  J={row['mean_J'] or 0:7.1f}  "
              f"fb={row['witness_used']:4d} ({100 * (row['fallback_share'] or 0):.1f}%)  "
              f"rej={row['rejects']:5d}")

    with open("certified_contact.json", "w", encoding="utf-8") as f:
        json.dump({"rows": rows}, f, indent=1)
    print("wrote certified_contact.json")


if __name__ == "__main__":
    main()
