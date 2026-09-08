"""
Train the constructive policy on the benchmark's *train* split, several seeds.

The zero-shot policy evaluated in `run_bench_policy.py` was trained on
`core.generate_instances`, while the suite is built by guillotine partition.
Evaluating it on the suite therefore measures distribution shift and the value of
learning at the same time, and cannot separate them.  This script removes the
confound by training on the suite's own train split, from disjoint seeds, so the
only thing left between train and test is the instance draw.

Runs are serialized rather than parallel: this machine's GPU is shared, and a
single training run at a time keeps it usable for anything else.
"""

import argparse
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--steps", type=int, default=1_200_000)
    p.add_argument("--n-rooms", dest="n_rooms", type=int, default=32)
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--order", default="random")
    p.add_argument("--prefix", default="S_bench")
    p.add_argument("--bench-fills", dest="bench_fills", nargs="+", type=float,
                   default=[0.40, 0.60, 0.75],
                   help="fill ratios sampled during training; must be forwarded "
                        "to train_construct or the run silently trains on the "
                        "default range")
    a = p.parse_args()

    for s in a.seeds:
        tag = f"{a.prefix}_seed{s}"
        cmd = [sys.executable, "-u", "train_construct.py",
               "--instances", "bench", "--n-rooms", str(a.n_rooms),
               "--canvas", str(a.canvas), "--order", a.order,
               "--tag", tag, "--seed", str(s),
               "--total-steps", str(a.steps),
               "--bench-fills", *[str(f) for f in a.bench_fills]]
        print(f"\n{'=' * 70}\n{tag}\n{'=' * 70}", flush=True)
        t = time.time()
        r = subprocess.run(cmd)
        print(f"[{tag}] exit={r.returncode} in {time.time() - t:.0f}s", flush=True)
        if r.returncode != 0:
            print(f"[{tag}] FAILED -- stopping", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
