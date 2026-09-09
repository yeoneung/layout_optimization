"""Throwaway check that the evaluation paths run end to end."""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                              # noqa: E402
import core                                                    # noqa: E402
import evaluate2 as E                                          # noqa: E402

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
spec = build_spec("comb_high", flip_adj=True)
N = 16
inst = core.from_spec(spec, N)
obj = core.BatchObjective(inst)
rng = np.random.default_rng(1234)
n = len(spec.room_order)
X0 = rng.uniform(spec.w / 2, spec.grid_w - spec.w / 2, size=(N, n))
Y0 = rng.uniform(spec.h / 2, spec.grid_h - spec.h / 2, size=(N, n))

m, nr, env, a = E.load_improve("runs_smoke/smoke.pt", inst, dev)
r = E.run_improve(m, nr, env, dev, 512, X0, Y0, deterministic=False)
E.print_row(E.summarize("improve smoke H512", obj, r["x"], r["y"], r["evals"],
                        r["wall_time"], N))

m2, nr2, env2, a2 = E.load_construct("runs_smoke/csmoke.pt", inst, dev)
r2 = E.run_construct(m2, nr2, env2, dev, deterministic=True)
E.print_row(E.summarize("construct smoke greedy", obj, r2["x"], r2["y"], r2["evals"],
                        r2["wall_time"], N))
runs = [E.run_construct(m2, nr2, env2, dev) for _ in range(4)]
bk = E.best_of_k(runs)
E.print_row(E.summarize("construct smoke x4", obj, bk["x"], bk["y"], bk["evals"],
                        bk["wall_time"], N))
print("diversity of one sampled batch:", round(E.diversity(inst, runs[0]["x"],
                                                           runs[0]["y"]), 3))
print("EVAL PATHS OK")
