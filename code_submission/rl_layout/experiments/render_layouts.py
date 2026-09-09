"""
Qualitative figure: what the layouts actually look like.

The objective is a scalar and hides where the difference comes from, so the
panels are annotated with the decomposition (adjacency / wall bonus / overlap)
rather than only the total.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
import torch                                                   # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                              # noqa: E402

import baselines2 as B2                                        # noqa: E402
import core                                                    # noqa: E402
import evaluate2 as E                                          # noqa: E402
import figures as F                                            # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", default="comb_high")
    p.add_argument("--construct", default="runs/B1_constr_fixed.pt")
    p.add_argument("--instances", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--name", default="fig_layouts")
    a = p.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = build_spec(a.scenario, flip_adj=True)
    N = a.instances
    inst = core.from_spec(spec, N)
    obj = core.BatchObjective(inst)
    rng = np.random.default_rng(a.seed)
    n = len(spec.room_order)
    X0 = rng.uniform(spec.w / 2, spec.grid_w - spec.w / 2, size=(N, n))
    Y0 = rng.uniform(spec.h / 2, spec.grid_h - spec.h / 2, size=(N, n))

    panels = []
    panels.append(("random initialization", X0, Y0))

    xs, ys = B2.shelf_pack(obj, rng)
    panels.append(("shelf packing (no search)", xs, ys))

    r = B2.simulated_annealing(obj, X0, Y0, 45000, greedy_every=5000, seed=a.seed)
    panels.append(("SA 45 000 + polish", r["x"], r["y"]))

    rg = B2.simulated_annealing(obj, X0, Y0, 45000, greedy_every=5000, seed=a.seed,
                                grid=True)
    panels.append(("SA 45 000, lattice", rg["x"], rg["y"]))

    if os.path.exists(a.construct):
        model, nrm, env, _ = E.load_construct(a.construct, inst, dev)
        runs = [E.run_construct(model, nrm, env, dev) for _ in range(8)]
        bk = E.best_of_k(runs)
        panels.append(("RL, constructive (best of 8)", bk["x"], bk["y"]))

    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    ar = spec.grid_h / spec.grid_w
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, (4.6 * ar + 0.9) * nrow))
    axes = np.atleast_1d(axes).ravel()
    names = list(spec.room_order)
    for ax, (title, x, y) in zip(axes, panels):
        tot, c = obj.evaluate(x, y, components=True)
        b = int(np.argmax(tot))
        sub = (f"{title}\n$J$={tot[b]:.0f}   adjacency {c['adj'][b]:.0f} · "
               f"wall {c['edge'][b]:.0f} · overlap {c['overlap_area'][b]:.1f}")
        F.draw_layout(ax, inst, x, y, b, sub, names=names)
    for ax in axes[len(panels):]:
        ax.axis("off")
    fig.suptitle("Best of 32 runs per method — same objective, same scenario",
                 fontsize=11, fontweight="bold", color=F.INK, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    F.save(fig, a.name)


if __name__ == "__main__":
    main()
