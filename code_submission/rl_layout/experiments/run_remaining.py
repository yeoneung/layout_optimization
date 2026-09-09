"""
Everything still outstanding, run strictly one at a time.

The machine is shared with other GPU work, so this deliberately serializes
instead of fanning out: one training process at a time, modest batch and channel
counts.  Order is by how much each result carries in the argument.
"""

import subprocess
import sys
import time

PY = sys.executable
STEPS = [
    # Track E mechanisms (small floor plate, cheapest)
    ["run_stage3.py", "--n-envs", "64", "--log-every", "30", "--out", "runs"],
    # Track A ablation + Track C, all five rows at one matched budget
    ["run_track_ac.py", "--total-steps", "700000", "--n-envs", "64",
     "--rollout", "64", "--horizon", "512", "--log-every", "40", "--out", "runs"],
    # Track B on the clinic, then Track D on the instance distribution
    ["run_stage2.py", "--n-envs", "64", "--log-every", "30", "--out", "runs"],
]


def main():
    for cmd in STEPS:
        print(f"\n{'#'*78}\n# {' '.join(cmd)}\n{'#'*78}", flush=True)
        t = time.time()
        r = subprocess.run([PY, "-u"] + cmd)
        print(f"# exit {r.returncode} after {time.time()-t:.0f}s", flush=True)
        if r.returncode != 0:
            print("# non-zero exit; continuing with the next step", flush=True)


if __name__ == "__main__":
    main()
