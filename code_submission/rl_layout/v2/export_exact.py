"""
Export small benchmark instances as plain JSON for the exact CP-SAT solver.

The exact solver runs in a separate interpreter (ortools is installed only
there), so the instance has to cross an environment boundary.  Everything the
model needs is written explicitly, including the per-pair d_max computed by
`core.BatchObjective` -- reimplementing that normalization on the other side is
exactly the kind of silent divergence that would make the reported optimality
gaps meaningless.
"""

import argparse
import json

import numpy as np

import bench
import core


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--rooms", type=int, default=8)
    p.add_argument("--fills", nargs="+", type=float, default=[0.40, 0.60, 0.75])
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--take", type=int, default=None,
                   help="export only the first K instances of each cell")
    p.add_argument("--tag", default="exact",
                   help="seed tag; keeps this family disjoint from the main suite")
    p.add_argument("--avg-area", dest="avg_area", nargs=2, type=float,
                   default=[6.0, 10.0],
                   help="mean room area range; controls plate size and hence "
                        "whether CP-SAT can prove optimality")
    p.add_argument("--out", default="exact_specs.json")
    a = p.parse_args()

    specs = []
    for fl in a.fills:
        cell = bench.suite_cell(a.split, a.rooms, fl, a.instances,
                                tag=a.tag, avg_area=tuple(a.avg_area))
        inst = cell["inst"]
        obj = core.BatchObjective(inst)
        take = a.take or inst.B
        for b in range(min(take, inst.B)):
            dmax = {f"{int(i)},{int(j)}": float(obj.dmax[b, k])
                    for k, (i, j) in enumerate(zip(obj.iu0, obj.iu1))}
            specs.append({
                "split": a.split, "n_rooms": a.rooms, "fill": fl, "instance": b,
                "n": int(inst.n),
                "gw": int(round(inst.gw[b])), "gh": int(round(inst.gh[b])),
                "w": [int(round(v)) for v in inst.w[b]],
                "h": [int(round(v)) for v in inst.h[b]],
                "adj": inst.adj[b].tolist(),
                "dmax": dmax,
                "edge_side": inst.edge_side[b].tolist(),
                "edge_bottom": inst.edge_bottom[b].tolist(),
                "edge_top": inst.edge_top[b].tolist(),
                "adj_weight": float(inst.weights.adj_weight),
                "distance_power": float(inst.weights.distance_power),
                "no_overlap_bonus": float(inst.weights.no_overlap_bonus),
            })

    with open(a.out, "w") as f:
        json.dump({"specs": specs}, f)
    sizes = [s["gw"] * s["gh"] for s in specs]
    print(f"wrote {a.out}: {len(specs)} specs, plate cells "
          f"{min(sizes)}-{max(sizes)}")


if __name__ == "__main__":
    main()
