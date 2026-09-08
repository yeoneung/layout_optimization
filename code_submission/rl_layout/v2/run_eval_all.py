"""
Remaining evaluations, one at a time.

  1. the improvement-MDP ablation, evaluated properly on held-out starts rather
     than read off a training curve,
  2. the clinic scenario, where v1 reported annealing tying a no-search
     heuristic and reaching zero overlap in only 20% of runs,
  3. transfer: the instance-distribution policy against solvers run from scratch
     on briefs it has never seen.
"""

import subprocess
import sys
import time

PY = sys.executable
IMPROVE = ["runs/A0_v1_delta_g995.pt", "runs/A0_v1_delta_g0.pt",
           "runs/A1_best_translate.pt", "runs/A2_best_allops.pt",
           "runs/C1_neural_sa.pt"]

STEPS = [
    ["compare_main.py", "--instances", "64", "--skip-classical",
     "--improve"] + IMPROVE + ["--out", "compare_improve.json"],
    ["run_baselines.py", "--scenario", "hospital", "--out", "baselines_hospital.json"],
    ["run_construct_baselines.py", "--scenario", "hospital", "--instances", "64",
     "--k", "8", "--out", "constr_baselines_hospital.json"],
    ["compare_main.py", "--scenario", "hospital", "--instances", "64",
     "--construct", "runs/B2_constr_hospital.pt", "--out", "compare_hospital.json"],
    ["compare_transfer.py", "--construct", "runs/D1_constr_random.pt",
     "--instances", "48", "--canvas", "44", "--out", "transfer.json"],
]


def main():
    for cmd in STEPS:
        print(f"\n{'#'*78}\n# {' '.join(cmd[:4])} ...\n{'#'*78}", flush=True)
        t = time.time()
        r = subprocess.run([PY, "-u"] + cmd)
        print(f"# exit {r.returncode} after {time.time()-t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
