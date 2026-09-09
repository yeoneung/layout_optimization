"""
A harder benchmark: how the certificate story scales with facility count.

The suite of Section 7 stops at 64 facilities because that is where a fixed-$n$
policy can still be trained and compared.  The certificate question does not have
that restriction, and it is the question most likely to change with scale:
stranding is a coordination failure among facilities, so more of them should make
one-step legality a looser relaxation of viability.  This script tests that on
instances of 96 to 256 facilities, all with a provably feasible packing.

Cost forces a split, and it is reported rather than hidden.  The rollout
certificate is $O(n^2WH)$ per query and is invoked up to `n_cand` times per
commitment, so it is run only up to 128 facilities; beyond that only the cheap
methods are evaluated, and the omission is recorded in the output.
"""

import argparse
import json
import time

import numpy as np

import bench
from viability import ViabilityConstructor

# (n_rooms, max_grid): the plate has to grow with the facility count or the
# facilities shrink to the 2x2 minimum and the instance stops being a layout
# problem
GRID = {64: 52, 96: 64, 128: 72, 192: 88, 256: 100}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--rooms", nargs="+", type=int, default=[96, 128, 192, 256])
    p.add_argument("--fills", nargs="+", type=float, default=[0.85, 0.95])
    p.add_argument("--instances", type=int, default=15)
    p.add_argument("--rollout-max-n", dest="roll_max", type=int, default=128,
                   help="above this facility count the rollout certificate is "
                        "too slow to run and is recorded as skipped")
    p.add_argument("--n-cand", dest="n_cand", type=int, default=8)
    p.add_argument("--tag", default="big")
    p.add_argument("--out", default="bigbench.json")
    a = p.parse_args()

    rows = []
    for n in a.rooms:
        mg = GRID.get(n, 100)
        for fl in a.fills:
            cell = bench.suite_cell(a.split, n, fl, a.instances, tag=a.tag,
                                    max_grid=mg, avg_area=(9.0, 26.0))
            inst = cell["inst"]
            w = bench.verify_witness(cell)
            assert w["max_overlap"] <= 1e-9 and w["all_inside"], \
                f"infeasible witness at n={n} fill={fl}"
            print(f"\n=== n={n} fill={fl:.2f} plate {inst.gw[0]:.0f}x"
                  f"{inst.gh[0]:.0f}, {inst.B} instances ===", flush=True)

            methods = [("no certificate", dict(filter_mode="none")),
                       ("first-fit", dict(filter_mode="pack"))]
            if n <= a.roll_max:
                methods.append(("rollout", dict(filter_mode="rollout")))
            else:
                rows.append({"split": a.split, "n_rooms": n, "fill": fl,
                             "method": "rollout", "skipped": True})
                print("  rollout: SKIPPED (too slow at this size)", flush=True)

            for label, kw in methods:
                t0 = time.time()
                J, feas = [], []
                for b in range(inst.B):
                    r = ViabilityConstructor(inst.take(np.array([b])),
                                             n_cand=a.n_cand, room_rule="area",
                                             **kw).solve()
                    J.append(r["best"])
                    feas.append(r["feasible"])
                wall = time.time() - t0
                rows.append({"split": a.split, "n_rooms": n, "fill": fl,
                             "method": label, "skipped": False,
                             "J": float(np.mean(J)),
                             "feasible": float(np.mean(feas)),
                             "s_per_instance": float(wall / inst.B),
                             "per_instance_J": [float(v) for v in J],
                             "per_instance_feasible": [bool(v) for v in feas]})
                print(f"  {label:16s} feas={np.mean(feas):4.2f}  "
                      f"J={np.mean(J):9.1f}  {wall / inst.B:7.1f} s/inst",
                      flush=True)
                with open(a.out, "w") as f:
                    json.dump({"rows": rows}, f, indent=1)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
