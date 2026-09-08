"""
Certificate-guided heuristic decoding.

This is the algorithm the paper proposes, and it is a composition of two parts
that solve two different problems:

  * a certificate filters the policy's leading candidates;

  * quality is handled by the *policy*.  Among admitted placements the learned
    position head chooses, which is where a trained scorer beats the hand-designed
    rules by a margin we measure.

Concretely: rank the policy's position logits, walk down the ranking, and commit
the first position whose successor state carries a certificate.  If none of the
top `n_cand` does, fall back to the policy's own argmax.  That fallback is useful
engineering but voids the completion theorem.  This module is therefore
deliberately named and reported as a heuristic; the witness-preserving algorithm
is implemented in `certified_decode.py`.

The certificate is evaluated one instance at a time because it is a sequential
packing attempt, so this evaluator loops over the batch rather than vectorizing
it.  That costs wall-clock and is reported honestly; it is not inherent to the
method, only to this implementation.
"""

import time

import numpy as np
import torch

import core
import evaluate2 as E
from viability import ViabilityConstructor


@torch.no_grad()
def run_guided_heuristic(model, nrm, env, dev, inst, n_cand=32, temperature=1.0,
                         deterministic=False, seed=0, certificate="rollout"):
    """One certificate-guided heuristic rollout per instance."""
    t0 = time.time()
    B, n = env.B, env.n
    obs, glob = env.reset()
    T = torch.as_tensor
    rng = np.random.default_rng(seed)

    # One ViabilityConstructor per instance, used only for its certificate test.
    # It must be built on the *environment's* canvas, not on the instance's own
    # plate: the policy's position index is flat over the canvas grid, so a
    # certificate grid of a different width would decode every position wrongly.
    vcs = [ViabilityConstructor(inst.take(np.array([b])), canvas=env.GW,
                                filter_mode=certificate) for b in range(B)]
    assert all(vc.GW == env.GW and vc.GH == env.GH for vc in vcs)
    occ = [np.zeros((env.GH, env.GW), dtype=np.int32) for b in range(B)]
    placed = np.zeros((B, n), dtype=bool)
    # the rollout certificate scores candidate completions with the exact
    # marginal, so it needs the centres of what is already placed
    cx = [np.zeros((1, n)) for _ in range(B)]
    cy = [np.zeros((1, n)) for _ in range(B)]
    n_reject = 0
    n_fallback = 0

    for _ in range(n):
        cache = {}

        def planes_fn(room_np):
            P, legal, dead, fb = env.planes(room_np)
            cache.update(P=P, legal=legal, dead=dead, fb=fb)
            return T(P, device=dev), T(legal, device=dev), T(dead, device=dev)

        tok, pooled, _, rcat = model.room_dist(T(nrm(obs), device=dev),
                                               T(glob, device=dev),
                                               T(~env.placed, device=dev))
        fr = env.forced_room()
        if fr is not None:
            room = T(fr, device=dev)
        elif deterministic:
            room = rcat.logits.argmax(-1)
        else:
            room = torch.distributions.Categorical(
                logits=rcat.logits / temperature).sample()
        room_np = room.cpu().numpy()
        pl, lg_mask, dead = planes_fn(room_np)
        logits = model.pos_logits(tok, pooled, room, pl, lg_mask, dead)

        # rank positions once, then walk the ranking under the certificate
        order = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
        pos = np.empty(B, dtype=np.int64)
        for b in range(B):
            i = int(room_np[b])
            vc = vcs[b]
            rem = [j for j in range(n) if not placed[b, j] and j != i]
            chosen = None
            for k in range(min(n_cand, order.shape[1])):
                f = int(order[b, k])
                if not np.isfinite(logits[b, f].item()):
                    break
                r, c = f // env.GW, f % env.GW
                trial = occ[b].copy()
                trial[r:r + vc.ih[i], c:c + vc.iw[i]] += 1
                pa = placed[b].copy()
                pa[i] = True
                xa, ya = cx[b].copy(), cy[b].copy()
                xa[0, i] = c + inst.w[b, i] / 2.0
                ya[0, i] = r + inst.h[b, i] / 2.0
                if vc._feasible_after(trial, rem, pa, xa, ya):
                    chosen = (f, trial, xa, ya)
                    break
                n_reject += 1
            if chosen is None:
                n_fallback += 1
                f = int(order[b, 0])
                r, c = f // env.GW, f % env.GW
                trial = occ[b].copy()
                trial[r:r + vc.ih[i], c:c + vc.iw[i]] += 1
                xa, ya = cx[b].copy(), cy[b].copy()
                xa[0, i] = c + inst.w[b, i] / 2.0
                ya[0, i] = r + inst.h[b, i] / 2.0
                chosen = (f, trial, xa, ya)
            pos[b], occ[b], cx[b], cy[b] = chosen
            placed[b, i] = True

        obs, glob, _, _ = env.step(room_np, pos)

    x, y, tot, c = env.result()
    return {"x": x, "y": y, "best": tot, "overlap_area": c["overlap_area"],
            "adj": c["adj"], "edge": c["edge"], "evals": n * B,
            "rejects": n_reject, "fallbacks": n_fallback,
            "wall_time": time.time() - t0}


# Backward compatibility for old analysis notebooks.  New experiment scripts
# must import `run_guided_heuristic` so an unsafe fallback cannot be mistaken for
# the certified decoder.
run_certified = run_guided_heuristic


def best_of_k(runs):
    return E.best_of_k(runs)
