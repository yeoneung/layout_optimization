"""
Check that the closed-form marginal used by `greedy_construct` really is the
objective increment.  If it is not, the greedy baseline is optimizing something
other than the objective and the comparison against it means nothing.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from layout_env import build_spec                              # noqa: E402

import core                                                    # noqa: E402
from greedy_construct import GreedyConstructor                 # noqa: E402


def check(scenario="comb_high", B=4, seed=0):
    spec = build_spec(scenario, flip_adj=True)
    inst = core.from_spec(spec, B)
    G = GreedyConstructor(inst)
    obj = G.obj
    rng = np.random.default_rng(seed)
    n = inst.n

    x = np.zeros((B, n))
    y = np.zeros((B, n))
    placed = np.zeros((B, n), dtype=bool)
    occ = np.zeros((B, G.GH, G.GW), dtype=np.int32)
    prev = obj.evaluate_masked(x, y, placed)

    worst = 0.0
    order = rng.permutation(n)
    for t, i in enumerate(order):
        legal, S, valid = G._legal(occ, i)
        gain = G._gain(i, x, y, placed)
        flat = np.where(legal, gain, -np.inf).reshape(B, -1).argmax(axis=1)
        pyy, pxx = flat // G.GW, flat % G.GW
        predicted = gain.reshape(B, -1)[np.arange(B), flat]

        b = np.arange(B)
        x[b, i] = pxx + inst.w[b, i] / 2.0
        y[b, i] = pyy + inst.h[b, i] / 2.0
        placed[b, i] = True
        for k in range(B):
            occ[k, pyy[k]:pyy[k] + G.ih[k, i], pxx[k]:pxx[k] + G.iw[k, i]] += 1

        now = obj.evaluate_masked(x, y, placed)
        actual = now - prev
        prev = now
        e = float(np.abs(actual - predicted).max())
        worst = max(worst, e)
        if t < 3 or t == n - 1:
            print(f"  step {t:2d} room {i:2d}: predicted {predicted[0]:9.4f}  "
                  f"actual {actual[0]:9.4f}  max|err| {e:.2e}")
    print(f"  worst |predicted - actual| over all {n} placements: {worst:.3e}")
    assert worst < 1e-9, "closed-form marginal disagrees with the objective"
    tot = obj.evaluate(x, y)
    print(f"  final J = {tot.mean():.1f}  (all rooms placed, overlap "
          f"{obj.evaluate(x, y, components=True)[1]['overlap_area'].max():.3f})\n")


if __name__ == "__main__":
    for s in ("comb_high", "hospital"):
        print(f"[{s}]")
        check(s)
    print("OK - the greedy constructive baseline optimizes the exact objective.")
