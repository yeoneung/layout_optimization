"""
Training-free non-myopic search inside the *same* masked constructive state space.

This is the control that decides whether the learned policy is doing anything a
search procedure could not.  The greedy constructor answers "what does one-step
optimality get you inside the mask"; it leaves open the obvious next question,
which any operations-research reader will ask immediately: what does *lookahead*
get you inside the mask, still without learning anything?

Two standard answers are implemented here.

  BeamConstructor   keeps the B highest-scoring partial layouts at every commit
                    step instead of the single best.  Width 1 reproduces the
                    joint room+position greedy rule exactly, which is the
                    correctness check for the implementation.

  regret-k          commits the room that would suffer most from being deferred,
                    measured as the gap between its best and its k-th best legal
                    position.  This is the classical non-myopic insertion order
                    from vehicle routing, transplanted to placement.

Both reuse `GreedyConstructor`'s exact marginal gain and integral-image legality
test, so the action set is identical to the policy's by construction rather than
by reimplementation.  The beam is carried in the batch dimension: a beam of width
B is an InstanceBatch holding B copies of one instance, each copy tracking its own
occupancy grid.  That makes the whole search one vectorized numpy program.

Cost is recorded in the two units the paper reports separately -- full objective
evaluations, and scored room/position candidates -- because a marginal score over
a grid of positions is not the same amount of work as an objective call and
collapsing them into one number is how these comparisons go wrong.
"""

import time

import numpy as np

import core
from greedy_construct import GreedyConstructor


def replicate(inst, B, which=0):
    """B copies of one instance of `inst`, as a fresh InstanceBatch."""
    return inst.take(np.full(B, which, dtype=np.int64))


class BeamConstructor:
    """Beam search over commit sequences in the masked constructive space.

    `order` selects what the beam is allowed to decide.  With "free" it chooses
    both the next facility and its position, which maximizes the objective but,
    at high fill ratios, spends the plate greedily and strands a later facility.
    With "area" the commit order is fixed to largest-first --- the classical
    packing order --- and the beam searches positions only.  The second is the
    control the density experiment needs: without it, a learned policy that
    orders well would be compared only against procedures that order badly, and
    the comparison would flatter it.
    """

    def __init__(self, inst, width, canvas=None, order="free"):
        self.width = width
        self.order = order
        self.I1 = inst                                  # the single instance
        self.beam_inst = replicate(inst, width)
        self.gc = GreedyConstructor(self.beam_inst, canvas=canvas)
        self.obj1 = core.BatchObjective(inst)
        self.area_order = np.argsort(-(inst.w[0] * inst.h[0]))

    def solve(self):
        t0 = time.time()
        gc, B, n = self.gc, self.width, self.gc.n
        GH, GW = gc.GH, gc.GW

        placed = np.zeros((B, n), dtype=bool)
        occ = np.zeros((B, GH, GW), dtype=np.int32)
        x = np.zeros((B, n))
        y = np.zeros((B, n))
        px = np.zeros((B, n), dtype=np.int64)
        py = np.zeros((B, n), dtype=np.int64)
        score = np.full(B, -np.inf)
        score[0] = 0.0            # only one live hypothesis before the first commit
        cands = 0                 # scored (room, position) pairs
        NEG = -1e18

        for t in range(n):
            # Each (parent, room) pair contributes its single best legal position.
            # Without this restriction the beam degenerates: the B highest-scoring
            # successors are B adjacent cells for the same room in the same
            # parent, so widening the beam buys near-duplicates instead of
            # alternatives.  Reducing over positions first makes the beam search
            # the space of *insertion decisions*, which is the standard
            # construction and still reproduces the greedy rule at width 1.
            if self.order == "free":
                # one candidate per (parent, room): its single best position
                best = np.full((n, B), NEG)
                pos = np.zeros((n, B), dtype=np.int64)
                for i in range(n):
                    legal, _, _ = gc._legal(occ, i)
                    g = gc._gain(i, x, y, placed)
                    cands += int(legal.sum())
                    ok = (legal & (~placed[:, i])[:, None, None]
                          & np.isfinite(score)[:, None, None])
                    s = np.where(ok, score[:, None, None] + g, NEG).reshape(B, -1)
                    pos[i] = s.argmax(axis=1)           # first maximum: reproducible
                    best[i] = s[np.arange(B), pos[i]]
                shape = (n, B)
                flat = best.reshape(-1)
            else:
                # the room is forced, so the alternatives are its positions.
                # Reducing to one position per parent here would leave each
                # parent with a single successor and the beam could never widen.
                i = int(self.area_order[t])
                legal, _, _ = gc._legal(occ, i)
                g = gc._gain(i, x, y, placed)
                cands += int(legal.sum())
                ok = (legal & (~placed[:, i])[:, None, None]
                      & np.isfinite(score)[:, None, None])
                s = np.where(ok, score[:, None, None] + g, NEG).reshape(B, -1)
                shape = s.shape
                flat = s.reshape(-1)

            k = min(B, int(np.sum(flat > NEG / 2)))
            if k == 0:                      # no legal successor anywhere: stop early
                break
            # Ties broken toward the lowest flat index so that width 1 reproduces
            # the greedy rule exactly and every width is reproducible; a bare
            # argpartition returns an arbitrary maximum.
            part = np.argpartition(-flat, k - 1)[:k]
            thresh = flat[part].min()
            tied = np.flatnonzero(flat >= thresh)
            top = tied[np.lexsort((tied, -flat[tied]))][:k]

            if self.order == "free":
                ri, pb = np.unravel_index(top, shape)
                rr, rc = pos[ri, pb] // GW, pos[ri, pb] % GW
            else:
                pb, flatpos = np.unravel_index(top, shape)
                ri = np.full(k, int(self.area_order[t]), dtype=np.int64)
                rr, rc = flatpos // GW, flatpos % GW

            # rebuild the beam; parents are read before any child overwrites them
            n_placed, n_occ = placed[pb].copy(), occ[pb].copy()
            n_x, n_y = x[pb].copy(), y[pb].copy()
            n_px, n_py = px[pb].copy(), py[pb].copy()
            n_score = flat[top].copy()
            for s in range(k):
                i = ri[s]
                n_placed[s, i] = True
                n_px[s, i], n_py[s, i] = rc[s], rr[s]
                n_x[s, i] = rc[s] + self.beam_inst.w[s, i] / 2.0
                n_y[s, i] = rr[s] + self.beam_inst.h[s, i] / 2.0
                n_occ[s, rr[s]:rr[s] + gc.ih[s, i], rc[s]:rc[s] + gc.iw[s, i]] += 1

            pad = B - k
            if pad:                          # keep the arrays rectangular
                n_placed = np.concatenate([n_placed, placed[:pad] * False])
                n_occ = np.concatenate([n_occ, np.zeros((pad, GH, GW), np.int32)])
                n_x = np.concatenate([n_x, np.zeros((pad, n))])
                n_y = np.concatenate([n_y, np.zeros((pad, n))])
                n_px = np.concatenate([n_px, np.zeros((pad, n), np.int64)])
                n_py = np.concatenate([n_py, np.zeros((pad, n), np.int64)])
                n_score = np.concatenate([n_score, np.full(pad, -np.inf)])

            placed, occ, x, y, px, py, score = \
                n_placed, n_occ, n_x, n_y, n_px, n_py, n_score

        live = np.isfinite(score)
        tot, c = self.gc.obj.evaluate(x, y, components=True)
        tot = np.where(live, tot, -np.inf)
        b = int(np.argmax(tot))
        return {"x": x[b:b + 1].copy(), "y": y[b:b + 1].copy(),
                "best": float(tot[b]), "overlap_area": c["overlap_area"][b],
                "adj": c["adj"][b], "edge": c["edge"][b],
                "obj_calls": B,               # one full evaluation per final beam element
                "candidates": cands,
                "wall_time": time.time() - t0}


def regret_construct(inst, k=2, canvas=None, rng=None):
    """Regret-k insertion: commit the room with the largest best-vs-kth-best gap.

    Vectorized over the batch dimension of `inst`, so it costs the same as the
    greedy rule and can be run over a whole benchmark at once.
    """
    t0 = time.time()
    gc = GreedyConstructor(inst, canvas=canvas)
    B, n, GH, GW = gc.B, gc.n, gc.GH, gc.GW
    placed = np.zeros((B, n), dtype=bool)
    occ = np.zeros((B, GH, GW), dtype=np.int32)
    x = np.zeros((B, n))
    y = np.zeros((B, n))
    cands = 0
    NEG = -1e18

    for _ in range(n):
        best_g = np.full((B, n), NEG)         # best legal gain per room
        best_flat = np.zeros((B, n), dtype=np.int64)
        regret = np.full((B, n), NEG)
        for i in range(n):
            legal, _, _ = gc._legal(occ, i)
            g = gc._gain(i, x, y, placed)
            cands += int(legal.sum())
            s = np.where(legal, g, NEG).reshape(B, -1)
            kk = min(k, s.shape[1])
            idx = np.argpartition(-s, kk - 1, axis=1)[:, :kk]
            vals = np.take_along_axis(s, idx, axis=1)
            order = np.argsort(-vals, axis=1)
            vals = np.take_along_axis(vals, order, axis=1)
            idx = np.take_along_axis(idx, order, axis=1)
            alive = ~placed[:, i] & (vals[:, 0] > NEG / 2)
            best_g[:, i] = np.where(alive, vals[:, 0], NEG)
            best_flat[:, i] = idx[:, 0]
            # a room with no k-th option is maximally urgent, not minimally
            kth = np.where(vals[:, -1] > NEG / 2, vals[:, -1], vals[:, 0] - 1e6)
            regret[:, i] = np.where(alive, vals[:, 0] - kth, NEG)

        pick = regret.argmax(axis=1)
        b = np.arange(B)
        flat = best_flat[b, pick]
        rr, cc = flat // GW, flat % GW
        placed[b, pick] = True
        x[b, pick] = cc + inst.w[b, pick] / 2.0
        y[b, pick] = rr + inst.h[b, pick] / 2.0
        for m in range(B):
            i = pick[m]
            occ[m, rr[m]:rr[m] + gc.ih[m, i], cc[m]:cc[m] + gc.iw[m, i]] += 1

    tot, c = gc.obj.evaluate(x, y, components=True)
    return {"x": x, "y": y, "best": tot, "overlap_area": c["overlap_area"],
            "adj": c["adj"], "edge": c["edge"],
            "obj_calls": B, "candidates": cands, "evals": cands,
            "wall_time": time.time() - t0}
