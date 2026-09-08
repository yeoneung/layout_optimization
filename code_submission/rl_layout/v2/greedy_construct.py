"""
The classical counterpart of the learned constructive policy.

This is the control that decides what the Track B result actually shows.  The
learned policy gets two things at once: an action set masked to non-overlapping
placements, and a trained way of choosing inside it.  `greedy_construct` keeps
the first and replaces the second with one-step-optimal selection -- at every
step it evaluates the exact marginal objective of every unplaced room at every
legal lattice position and commits the best one.

If the learned policy only matched this, the Track B result would be about
masking and nothing would have been learned.  Any margin over it is what
lookahead beyond one step buys, which is the part a policy can learn and a greedy
rule cannot.

The marginal objective is computed in closed form rather than by re-evaluating
the whole layout: placing room i at p adds exactly its pairwise adjacency terms
against the already-placed rooms plus its own wall bonus, because a masked
placement never creates overlap and therefore never disturbs an existing term.
"""

import time

import numpy as np

import core
from core import _pair_distance


class GreedyConstructor:
    def __init__(self, inst, canvas=None):
        self.I = inst
        self.obj = core.BatchObjective(inst)
        self.B, self.n = inst.B, inst.n
        self.GW = canvas or int(inst.gw.max())
        self.GH = canvas or int(inst.gh.max())
        self.iw = np.rint(inst.w).astype(np.int64)
        self.ih = np.rint(inst.h).astype(np.int64)
        self.igw = np.rint(inst.gw).astype(np.int64)
        self.igh = np.rint(inst.gh).astype(np.int64)
        pi = np.zeros((self.n, self.n), dtype=np.int64)
        pi[self.obj.iu0, self.obj.iu1] = np.arange(self.obj.P)
        pi[self.obj.iu1, self.obj.iu0] = np.arange(self.obj.P)
        self.pair_idx = pi
        self._rows = np.arange(self.GH)[None, :, None]
        self._cols = np.arange(self.GW)[None, None, :]

    # ------------------------------------------------------------------
    def _legal(self, occ, i):
        """(B, GH, GW) True where room i fits with no overlap."""
        ii = np.zeros((self.B, self.GH + 1, self.GW + 1), dtype=np.int32)
        np.cumsum(np.cumsum(occ, axis=1), axis=2, out=ii[:, 1:, 1:])
        w = self.iw[:, i][:, None, None]
        h = self.ih[:, i][:, None, None]
        r1 = np.minimum(self._rows + h, self.GH)
        c1 = np.minimum(self._cols + w, self.GW)
        b = np.arange(self.B)[:, None, None]
        S = ii[b, r1, c1] - ii[b, self._rows, c1] - ii[b, r1, self._cols] + \
            ii[b, self._rows, self._cols]
        valid = ((self._cols + w <= self.igw[:, None, None])
                 & (self._rows + h <= self.igh[:, None, None]))
        return (S == 0) & valid, S, valid

    def _gain(self, i, x, y, placed, integral=None):
        """(B, GH, GW) exact objective added by placing room i at each position."""
        w = self.I.w[:, i][:, None, None]
        h = self.I.h[:, i][:, None, None]
        cx = self._cols + w / 2.0
        cy = self._rows + h / 2.0
        wt = self.I.weights

        gain = np.zeros((self.B, self.GH, self.GW))
        for j in range(self.n):
            m = placed[:, j]
            if not m.any():
                continue
            a = self.I.adj[:, i, j][:, None, None]
            dmax = self.obj.dmax[:, self.pair_idx[i, j]][:, None, None]
            d = _pair_distance(cx, cy, w, h,
                               x[:, j][:, None, None], y[:, j][:, None, None],
                               self.I.w[:, j][:, None, None],
                               self.I.h[:, j][:, None, None])
            nd = np.clip(d / np.maximum(dmax, 1e-12), 0.0, 1.0) ** wt.distance_power
            c = np.where(a >= 0, a * nd, (-a) * (1.0 - nd))
            gain += np.where(m[:, None, None], c, 0.0)
        gain *= wt.adj_weight

        touch_side = (self._cols == 0) | (self._cols + self.iw[:, i][:, None, None]
                                          == self.igw[:, None, None])
        gain = gain + (touch_side * self.I.edge_side[:, i][:, None, None]
                       + (self._rows == 0) * self.I.edge_bottom[:, i][:, None, None]
                       + (self._rows + self.ih[:, i][:, None, None]
                          == self.igh[:, None, None]) * self.I.edge_top[:, i][:, None, None])
        return gain

    # ------------------------------------------------------------------
    def solve(self, rng=None, jitter=0.0, room_rule="greedy", order_noise=1.0):
        """One layout per instance.

        `jitter` adds noise to the greedy score so repeated runs differ, which is
        what makes a fair best-of-K comparison against a sampled policy possible.
        `room_rule='area'` fixes the commit order to largest-first and only the
        position is chosen greedily -- the cheaper, more common heuristic.
        `room_rule='random'` draws the commit order uniformly per instance, which
        is the training-free counterpart of the policy's diversity mechanism
        (Section 6.5 of the paper) and therefore the control the diversity claim
        has to be measured against.  `order_noise` interpolates between the two:
        it is the fraction of commit slots that get reshuffled, so 0 reproduces
        the area order exactly and 1 is a uniform permutation.  Sweeping it
        traces the heuristic's own quality/diversity curve continuously, which is
        what decides whether the policy holds any region of that curve alone.
        """
        t0 = time.time()
        rng = rng or np.random.default_rng(0)
        B, n = self.B, self.n
        px = np.zeros((B, n), dtype=np.int64)
        py = np.zeros((B, n), dtype=np.int64)
        placed = np.zeros((B, n), dtype=bool)
        occ = np.zeros((B, self.GH, self.GW), dtype=np.int32)
        x = np.zeros((B, n))
        y = np.zeros((B, n))
        evals = 0
        if room_rule == "random":
            order = np.argsort(-(self.I.w * self.I.h), axis=1)
            m = int(round(np.clip(order_noise, 0.0, 1.0) * n))
            if m > 1:
                # reshuffle m of the n commit slots; the rest keep the area order
                for k in range(B):
                    slots = rng.choice(n, m, replace=False)
                    order[k, slots] = order[k, rng.permutation(slots)]
        else:
            order = np.argsort(-(self.I.w * self.I.h), axis=1)

        for t in range(n):
            best_score = np.full(B, -np.inf)
            best_room = np.zeros(B, dtype=np.int64)
            best_flat = np.zeros(B, dtype=np.int64)
            cand_rooms = range(n)
            forced = order[:, t] if room_rule in ("area", "random") else None

            for i in cand_rooms:
                act = ~placed[:, i]
                if forced is not None:
                    act = act & (forced == i)
                if not act.any():
                    continue
                legal, S, valid = self._legal(occ, i)
                g = self._gain(i, x, y, placed)
                evals += B                      # one marginal-objective sweep
                if jitter > 0:
                    g = g + rng.normal(0, jitter, g.shape)
                # illegal positions are only considered if nothing else is left
                score = np.where(legal, g, -1e9 + np.where(valid, -S, -1e6))
                flat = score.reshape(B, -1).argmax(axis=1)
                sc = score.reshape(B, -1)[np.arange(B), flat]
                take = act & (sc > best_score)
                best_score = np.where(take, sc, best_score)
                best_room = np.where(take, i, best_room)
                best_flat = np.where(take, flat, best_flat)

            r = best_room
            pyy, pxx = best_flat // self.GW, best_flat % self.GW
            b = np.arange(B)
            px[b, r], py[b, r] = pxx, pyy
            placed[b, r] = True
            x[b, r] = pxx + self.I.w[b, r] / 2.0
            y[b, r] = pyy + self.I.h[b, r] / 2.0
            for k in range(B):
                i = r[k]
                occ[k, pyy[k]:pyy[k] + self.ih[k, i], pxx[k]:pxx[k] + self.iw[k, i]] += 1

        tot, c = self.obj.evaluate(x, y, components=True)
        return {"x": x, "y": y, "best": tot, "overlap_area": c["overlap_area"],
                "adj": c["adj"], "edge": c["edge"],
                "evals": evals, "wall_time": time.time() - t0}
