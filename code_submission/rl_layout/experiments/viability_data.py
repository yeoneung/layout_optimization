"""
Supervised data for a learned predictor of the rollout verdict.

An earlier version of this script labelled states by whether *some* completion
exists, using randomized restarts as the oracle.  The experiments showed that
target to be the wrong one: a strictly more permissive viability test is a worse
filter, because it admits states from which completion exists only along paths
the constructor never takes.  What predicts the constructor's success is the
constructor's own base heuristic.

So the label here is the rollout verdict --- does largest-first greedy placement,
the base heuristic, complete without overlap from this state --- which is exactly
the certificate whose witness the constructor can follow.  Computing it costs a
full O(n^2 WH) rollout per query; a predictor that reproduces it from
free-space features adds one forward pass after O(nWH) feature extraction, and
that asymptotic gap is the point.

Both the rollout label and the cheap first-fit verdict are recorded on the same
states, so the paper can report what each certificate would have admitted.
"""

import argparse
import time

import numpy as np

import bench
import viability as V
from viability import ViabilityConstructor


def walk(inst, b, rng, n_cand=6):
    """Walk one largest-first rollout, yielding candidate successor states."""
    vc = ViabilityConstructor(inst.take(np.array([b])), filter_mode="none")
    n, GW = vc.n, vc.GW
    occ = np.zeros((vc.GH, vc.GW), dtype=np.int32)
    placed = np.zeros(n, dtype=bool)
    x = np.zeros((1, n))
    y = np.zeros((1, n))
    order = np.argsort(-(inst.w[b] * inst.h[b]))

    for t in range(n):
        i = int(order[t])
        lg, _, _ = vc.gc._legal(occ[None], i)
        g = vc.gc._gain(i, x, y, placed[None])
        s = np.where(lg[0], g[0], -np.inf)
        if not np.isfinite(s).any():
            return
        flat = np.argsort(-s, axis=None, kind="stable")
        k = min(n_cand, int(np.isfinite(s).sum()))
        picks = list(flat[:k])
        # a few random legal positions as well, so the predictor sees states a
        # purely greedy walk would never propose
        rest = [f for f in flat[k:k + 300] if np.isfinite(s.reshape(-1)[f])]
        if rest:
            picks += list(rng.choice(rest, size=min(2, len(rest)), replace=False))

        for f in picks:
            f = int(f)
            r, c = f // GW, f % GW
            trial = occ.copy()
            trial[r:r + vc.ih[i], c:c + vc.iw[i]] += 1
            pl = placed.copy()
            pl[i] = True
            xa, ya = x.copy(), y.copy()
            xa[0, i] = c + inst.w[b, i] / 2.0
            ya[0, i] = r + inst.h[b, i] / 2.0
            rem = [j for j in range(n) if not pl[j]]
            if rem:
                yield vc, trial, rem, pl, xa, ya

        f = int(flat[0])
        r, c = f // GW, f % GW
        occ[r:r + vc.ih[i], c:c + vc.iw[i]] += 1
        placed[i] = True
        x[0, i] = c + inst.w[b, i] / 2.0
        y[0, i] = r + inst.h[b, i] / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", default="train")
    p.add_argument("--rooms", type=int, default=32)
    p.add_argument("--fills", nargs="+", type=float, default=[0.93, 0.95, 0.97])
    p.add_argument("--instances", type=int, default=40)
    p.add_argument("--n-cand", dest="n_cand", type=int, default=6)
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="viability_train.npz")
    a = p.parse_args()

    X, Y, F, G, GROUP = [], [], [], [], []
    t0 = time.time()
    for fill_index, fl in enumerate(a.fills):
        cell = bench.suite_cell(a.split, a.rooms, fl, a.instances)
        inst = cell["inst"]
        rng = np.random.default_rng(a.seed + int(fl * 100))
        n0 = len(X)
        for b in range(inst.B):
            # Every state descended from the same generated layout instance
            # receives the same group id.  The trainer uses this id to prevent
            # near-duplicate states from one trajectory appearing on both
            # sides of the validation split.
            group_id = fill_index * a.instances + b
            for vc, occ, rem, pl, xa, ya in walk(inst, b, rng, a.n_cand):
                lab = vc._complete_greedy(occ, rem, pl, xa, ya)
                ff = V.complete_random(occ, rem, vc.iw, vc.ih, vc.GH, vc.GW,
                                       vc.gw, vc.gh, 1, rng)
                X.append(V.state_features(occ, rem, vc.iw, vc.ih,
                                          vc.GH, vc.GW, vc.gw, vc.gh))
                Y.append(1.0 if lab else 0.0)
                F.append(1.0 if ff else 0.0)
                G.append(fl)
                GROUP.append(group_id)
        y = np.array(Y[n0:])
        f = np.array(F[n0:])
        agree = float(np.mean(y == f)) if len(y) else 0.0
        print(f"fill={fl:.2f}  {len(X) - n0:6d} samples  rollout+={y.mean():.3f}  "
              f"first-fit+={f.mean():.3f}  agree={agree:.3f}  "
              f"{time.time() - t0:.0f}s", flush=True)

    X = np.stack(X)
    np.savez_compressed(a.out, X=X, Y=np.array(Y, np.float32),
                        FF=np.array(F, np.float32), fill=np.array(G, np.float32),
                        group=np.array(GROUP, np.int64))
    print(f"\nwrote {a.out}: {X.shape[0]} samples, rollout positive "
          f"{np.mean(Y):.3f}")


if __name__ == "__main__":
    main()
