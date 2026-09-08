"""
Does viability, rather than objective lookahead, explain the learned policy?

The experiment is a ladder of rejection filters bolted onto the *same* greedy
rule, differing only in how hard they try to answer "will this commitment leave a
later facility with nowhere to go".  Each rung is training-free:

    none   the one-step mask alone; reproduces the greedy rule exactly
    n1     necessary: every remaining facility must retain a legal position
    area   n1, plus the largest remaining facility must retain a legal position
    pack   sufficient: an explicit largest-first packing of every remaining
           facility must succeed, which certifies the state is viable

The prediction under test is quantitative and was fixed before running: at fill
0.95 the greedy rule reaches 0.20 feasibility and the learned policy 0.80, so if
the policy's advantage is viability estimation, the filters should move greedy
toward 0.80 rather than leaving it at 0.20 -- and the cost of doing so should
rise steeply along the ladder, since exact viability is a packing-feasibility
question.

What the ladder cannot do is settle whether a *learned* filter buys the same
feasibility at one-step cost; that needs the critic and is a separate run.  This
run establishes the reference curve it would have to match.
"""

import argparse
import json
import time

import numpy as np

import bench
from viability import ViabilityConstructor

MODES = ["none", "n1", "area", "pack"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--rooms", type=int, default=32)
    p.add_argument("--fills", nargs="+", type=float,
                   default=[0.75, 0.80, 0.85, 0.90, 0.95])
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--modes", nargs="+", default=MODES)
    p.add_argument("--n-cand", dest="n_cand", type=int, default=16)
    p.add_argument("--room-rule", dest="room_rule", default="greedy",
                   choices=["greedy", "area"],
                   help="which facility to commit: by immediate gain, or "
                        "largest-first. Crossing this with the filter separates "
                        "the contribution of the commit order from that of "
                        "viability checking.")
    p.add_argument("--out", default="viability.json")
    a = p.parse_args()

    rows = []
    for fl in a.fills:
        cell = bench.suite_cell(a.split, a.rooms, fl, a.instances)
        inst = cell["inst"]
        for mode in a.modes:
            t0 = time.time()
            J, feas, cand, calls, rej = [], [], [], [], []
            for b in range(inst.B):
                r = ViabilityConstructor(inst.take(np.array([b])),
                                         filter_mode=mode,
                                         n_cand=a.n_cand,
                                         room_rule=a.room_rule).solve()
                J.append(r["best"])
                feas.append(r["feasible"])
                cand.append(r["candidates"])
                calls.append(r["filter_calls"])
                rej.append(r["rejects"])
            wall = time.time() - t0
            row = {"split": a.split, "n_rooms": a.rooms, "fill": fl,
                   "mode": mode, "n_cand": a.n_cand,
                   "room_rule": a.room_rule,
                   "J_mean": float(np.mean(J)),
                   "feasible_rate": float(np.mean(feas)),
                   "candidates": float(np.mean(cand)),
                   "filter_calls": float(np.mean(calls)),
                   "rejects": float(np.mean(rej)),
                   "ms_per_instance": float(wall / inst.B * 1000),
                   "per_instance_J": [float(v) for v in J],
                   "per_instance_feasible": [bool(v) for v in feas]}
            rows.append(row)
            print(f"fill={fl:.2f}  {a.room_rule:6s} {mode:5s}  feas={row['feasible_rate']:4.2f}  "
                  f"J={row['J_mean']:9.2f}  rejects={row['rejects']:6.1f}  "
                  f"{row['ms_per_instance']:9.1f} ms", flush=True)
            with open(a.out, "w") as f:
                json.dump({"rows": rows}, f, indent=1)

    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
