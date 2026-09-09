"""
How small is the feasible set?  Measured, not asserted.

An earlier draft of this paper claimed that zero overlap is a "measure-zero
event" in a continuous state space.  That is false: non-overlap of axis-aligned
rectangles is a conjunction of *strict* inequalities such as
x_i + w_i/2 < x_j - w_j/2, so the feasible set is open and has positive Lebesgue
measure whenever it is non-empty.  The claim has been removed from the
manuscript.

What is true, and what actually explains why a reward penalty struggles to
deliver feasibility, is quantitative rather than topological: the feasible
fraction of the placement space decays sharply in the number of rooms and in the
fill ratio.  This script measures that fraction directly by uniform sampling, so
the manuscript can state a number instead of a wrong theorem.

Reported for continuous centres (the improvement MDP's state space) and for
lattice-aligned placements (the constructive MDP's), because the second is the
comparison the paper's argument turns on.
"""

import argparse
import json

import numpy as np

import core


def feasible_fraction(inst1, n_samples, rng, chunk=2048, grid=False):
    """Fraction of uniformly sampled layouts of `inst1` with zero total overlap."""
    hits = 0
    done = 0
    while done < n_samples:
        B = min(chunk, n_samples - done)
        rep = inst1.take(np.zeros(B, dtype=np.int64))
        obj = core.BatchObjective(rep)
        x, y = core.random_layout(obj, rng, grid_aligned=grid)
        _, c = obj.evaluate(x, y, components=True)
        hits += int(np.sum(c["overlap_area"] <= 1e-9))
        done += B
    return hits / n_samples


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rooms", nargs="+", type=int, default=[2, 3, 4, 6, 8, 10, 12, 16])
    p.add_argument("--fills", nargs="+", type=float, default=[0.35, 0.50, 0.65])
    p.add_argument("--samples", type=int, default=200_000)
    p.add_argument("--instances", type=int, default=8)
    p.add_argument("--seed", type=int, default=20260812)
    p.add_argument("--out", default="feasible_volume.json")
    a = p.parse_args()

    rows = []
    for fill in a.fills:
        for n in a.rooms:
            rng = np.random.default_rng(a.seed + n * 1000 + int(fill * 100))
            fr_c, fr_g = [], []
            for k in range(a.instances):
                batch = core.generate_instances(1, n, rng, fill=(fill, fill))
                one = batch.take(np.zeros(1, dtype=np.int64))
                fr_c.append(feasible_fraction(one, a.samples, rng, grid=False))
                fr_g.append(feasible_fraction(one, a.samples, rng, grid=True))
            row = {"n_rooms": n, "fill": fill,
                   "samples_per_instance": a.samples, "instances": a.instances,
                   "feasible_frac_continuous": float(np.mean(fr_c)),
                   "feasible_frac_lattice": float(np.mean(fr_g)),
                   "min_continuous": float(np.min(fr_c)),
                   "max_continuous": float(np.max(fr_c))}
            rows.append(row)
            print(f"fill {fill:.2f}  n={n:3d}   continuous {row['feasible_frac_continuous']:.3e}"
                  f"   lattice {row['feasible_frac_lattice']:.3e}", flush=True)

    with open(a.out, "w") as f:
        json.dump({"rows": rows}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
