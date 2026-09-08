"""
Can a learned predictive filter buy rollout-level feasibility at first-fit cost?

The ladder compared here is one of *certificates*, all bolted onto the identical
largest-first constructor, so the only thing that varies is how admissibility is
decided:

    pack       one deterministic first-fit completion   -- cheap, conservative
    oracle-8   eight randomized completions             -- tighter, 8x the cost
    oracle-32  thirty-two randomized completions        -- tighter still
    learned    O(nWH) features plus one forward pass    -- unsound

The hypothesis under test is that the learned filter tracks the expensive oracle
rather than the cheap one, at a cost that does not grow with the oracle's budget.
Feasibility is the primary axis because that is what a certificate is for;
objective and wall-clock are reported alongside.

The threshold is swept, because a certificate is a trade-off and reporting one
operating point would hide it.
"""

import argparse
import json
import time

import numpy as np
import torch

import bench
from viability import ViabilityConstructor
from viability_net import ViabilityNet


def load_net(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = ViabilityNet()
    net.load_state_dict(ck["state"])
    net.eval()
    return (net, ck["mu"], ck["sd"]), ck.get("metadata", {})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="test")
    p.add_argument("--rooms", type=int, default=32)
    p.add_argument("--fills", nargs="+", type=float,
                   default=[0.90, 0.93, 0.95, 0.97])
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--net", default="runs/viability_net.pt")
    p.add_argument("--thresholds", nargs="+", type=float,
                   default=None)
    p.add_argument("--n-cand", dest="n_cand", type=int, default=32)
    p.add_argument("--out", default="learned_cert.json")
    a = p.parse_args()

    net, validation_metadata = load_net(a.net)
    if a.thresholds is None:
        if "thresholds" not in validation_metadata:
            raise ValueError("checkpoint has no validation-derived thresholds")
        thresholds = validation_metadata["thresholds"]
        threshold_source = "instance-disjoint validation quantiles"
    else:
        thresholds = a.thresholds
        threshold_source = "command line"
    rows = []

    def run(cell, label, **kw):
        inst = cell["inst"]
        t0 = time.time()
        J, feas, calls = [], [], []
        for b in range(inst.B):
            vc = ViabilityConstructor(inst.take(np.array([b])),
                                      n_cand=a.n_cand, room_rule="area", **kw)
            r = vc.solve()
            J.append(r["best"])
            feas.append(r["feasible"])
            calls.append(vc.n_filter_calls)
        wall = time.time() - t0
        row = {"fill": cell["fill"], "method": label,
               "J": float(np.mean(J)), "feasible": float(np.mean(feas)),
               "filter_calls": float(np.mean(calls)),
               "ms": float(wall / inst.B * 1000),
               "per_instance_J": [float(v) for v in J],
               "per_instance_feasible": [bool(v) for v in feas]}
        rows.append(row)
        print(f"fill={cell['fill']:.2f}  {label:22s} feas={row['feasible']:4.2f}  "
              f"J={row['J']:8.1f}  calls={row['filter_calls']:6.0f}  "
              f"{row['ms']:8.1f} ms", flush=True)

    for fl in a.fills:
        cell = bench.suite_cell(a.split, a.rooms, fl, a.instances)
        run(cell, "no certificate", filter_mode="none")
        run(cell, "first-fit", filter_mode="pack")
        run(cell, "oracle-32", filter_mode="oracle", oracle_tries=32)
        run(cell, "rollout (exact)", filter_mode="rollout")
        for th in thresholds:
            run(cell, f"learned t={th:+.3f}", filter_mode="learned",
                net=net, threshold=th)
        print(flush=True)

    with open(a.out, "w") as f:
        json.dump({"metadata": {
            "split": a.split, "rooms": a.rooms, "fills": a.fills,
            "instances_per_fill": a.instances, "n_candidates": a.n_cand,
            "threshold_source": threshold_source,
            "thresholds": [float(v) for v in thresholds],
            "validation": validation_metadata,
        }, "rows": rows}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
