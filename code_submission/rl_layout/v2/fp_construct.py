"""
Masked constructive placement on the floorplanning benchmarks.

This is the same algorithm as `viability.ViabilityConstructor` with the
adjacency objective replaced by HPWL: commit one block at a time to a lattice
position that provably overlaps nothing, choose among candidates by the exact
marginal, and optionally require that a candidate leave the remaining blocks
placeable.

Only two things change relative to the generated-instance version.  The marginal
is now the HPWL increment of `fp_core.HPWL`, which is minimized rather than
maximized, so scores are negated.  And blocks may be rotated, since the
floorplanning literature treats hard blocks as rotatable and forbidding it would
handicap every method here against published results; a rotation is simply a
second (w, h) option per block, so it costs one more legality mask per candidate
and nothing conceptually.
"""

import time

import numpy as np

import fp_core as FP


def _integral(occ):
    ii = np.zeros((occ.shape[0] + 1, occ.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(occ, axis=0), axis=1, out=ii[1:, 1:])
    return ii


def _legal(ii, GH, GW, w, h, W, H):
    rows = np.arange(GH)[:, None]
    cols = np.arange(GW)[None, :]
    r1 = np.minimum(rows + h, GH)
    c1 = np.minimum(cols + w, GW)
    S = ii[r1, c1] - ii[rows, c1] - ii[r1, cols] + ii[rows, cols]
    inside = (cols + w <= W) & (rows + h <= H)
    return (S == 0) & inside


class FPConstructor:
    """Constructive floorplanner with an optional viability certificate.

    `room_rule`: "area" commits blocks largest-first; "greedy" picks the block
    and position with the best marginal jointly.
    `filter_mode`: "none", "pack" (one first-fit completion), or "rollout" (the
    constructor's own base heuristic run to completion).
    `rotate`: allow the 90-degree rotation of each block as a second option.
    """

    def __init__(self, inst: FP.FloorplanInstance, filter_mode="none",
                 room_rule="area", n_cand=8, rotate=True, positions="corner"):
        self.I = inst
        self.obj = FP.HPWL(inst)
        self.n = inst.n
        self.GW, self.GH = inst.W, inst.H
        self.mode = filter_mode
        self.rule = room_rule
        self.n_cand = n_cand
        self.rotate = rotate
        self.positions = positions        # "corner" or "grid"
        self.area_order = np.argsort(-(inst.w * inst.h))
        self.n_filter_calls = 0

    # ------------------------------------------------------------------
    def _orients(self, i):
        w, h = int(self.I.w[i]), int(self.I.h[i])
        return [(w, h)] if (not self.rotate or w == h) else [(w, h), (h, w)]

    def _complete_firstfit(self, occ, remaining):
        work = occ.copy()
        for i in sorted(remaining, key=lambda j: -(self.I.w[j] * self.I.h[j])):
            ii = _integral(work)
            placed = False
            for (w, h) in self._orients(i):
                m = _legal(ii, self.GH, self.GW, w, h, self.I.W, self.I.H)
                idx = np.argwhere(m)
                if len(idx):
                    r, c = idx[0]
                    work[r:r + h, c:c + w] += 1
                    placed = True
                    break
            if not placed:
                return False
        return True

    def _complete_greedy(self, occ, remaining, box):
        """The base heuristic run to completion: largest-first, best marginal,
        over the same candidate positions the constructor itself uses."""
        work = occ.copy()
        bx = box.copy()
        ex, ey = [], []
        ys, xs = np.nonzero(work)
        for i in sorted(remaining, key=lambda j: -(self.I.w[j] * self.I.h[j])):
            ii = _integral(work)
            cands = self._candidates(i, ii, bx, np.array(ex, dtype=np.int64),
                                     np.array(ey, dtype=np.int64))
            if not cands:
                return False
            _, _, r, c, w, h = min(cands, key=lambda z: z[0])
            work[r:r + h, c:c + w] += 1
            bx = self.obj.commit(i, bx, c + w / 2.0, r + h / 2.0)
            ex.append(c + w)
            ey.append(r + h)
        return True

    def _centres(self, w, h):
        rows = np.arange(self.GH)[:, None]
        cols = np.arange(self.GW)[None, :]
        return cols + w / 2.0, rows + h / 2.0

    def corner_points(self, xs, ys):
        """Candidate lower-left positions: the classical corner-point set.

        Scoring every lattice cell is what makes this constructor pack loosely.
        The top few cells by wirelength marginal are all neighbours of one
        another, so widening the plate narrows the effective search and density
        gets worse rather than better --- we measured exactly that.  The standard
        remedy in packing is to consider only positions where the new block would
        sit flush against something: abscissas are the plate edge and the right
        edges of placed blocks, ordinates the plate edge and their top edges.
        That is O(k^2) candidates after k placements instead of O(WH), and every
        one of them is a position a tight packing could use.
        """
        cx = np.unique(np.concatenate([[0], xs]))
        cy = np.unique(np.concatenate([[0], ys]))
        return cx.astype(np.int64), cy.astype(np.int64)

    def _legal_points(self, ii, w, h, rs, cs):
        """Vectorized legality test at an explicit list of lower-left points."""
        inside = (cs + w <= self.I.W) & (rs + h <= self.I.H)
        r1 = np.minimum(rs + h, self.GH)
        c1 = np.minimum(cs + w, self.GW)
        S = ii[r1, c1] - ii[rs, c1] - ii[r1, cs] + ii[rs, cs]
        return (S == 0) & inside

    def _candidates(self, i, ii, box, ex, ey):
        """(score, r, c, w, h) for the admissible positions of block `i`."""
        out = []
        for (w, h) in self._orients(i):
            if self.positions == "corner":
                cx, cy = self.corner_points(ex, ey)
                cs, rs = np.meshgrid(cx, cy)
                cs, rs = cs.ravel(), rs.ravel()
            else:
                rows = np.arange(self.GH)
                cols = np.arange(self.GW)
                cs, rs = np.meshgrid(cols, rows)
                cs, rs = cs.ravel(), rs.ravel()
            ok = self._legal_points(ii, w, h, rs, cs)
            if not ok.any():
                continue
            rs, cs = rs[ok], cs[ok]
            g = self.obj.marginal_grid(i, box, cs + w / 2.0, rs + h / 2.0)
            k = min(self.n_cand, len(g))
            sel = np.argsort(g, kind="stable")[:k]
            out += [(float(g[j]), i, int(rs[j]), int(cs[j]), w, h) for j in sel]
        return out

    def _admissible(self, occ, remaining, box):
        if self.mode == "none" or not remaining:
            return True
        self.n_filter_calls += 1
        if self.mode == "pack":
            return self._complete_firstfit(occ, remaining)
        if self.mode == "rollout":
            return self._complete_greedy(occ, remaining, box)
        raise ValueError(self.mode)

    # ------------------------------------------------------------------
    def solve(self):
        t0 = time.time()
        n = self.n
        occ = np.zeros((self.GH, self.GW), dtype=np.int32)
        placed = np.zeros(n, dtype=bool)
        X = np.zeros(n)
        Y = np.zeros(n)
        box = self.obj.fresh()
        ex, ey = [], []              # right and top edges of placed blocks
        fallbacks = 0

        for t in range(n):
            ii = _integral(occ)
            todo = ([int(self.area_order[t])] if self.rule == "area"
                    else [i for i in range(n) if not placed[i]])
            cands = []
            for i in todo:
                if placed[i]:
                    continue
                cands += self._candidates(i, ii, box,
                                          np.array(ex, dtype=np.int64),
                                          np.array(ey, dtype=np.int64))
            if not cands:
                break
            cands.sort(key=lambda z: z[0])
            cands = cands[:self.n_cand]

            chosen = None
            for sc, i, r, c, w, h in cands:
                trial = occ.copy()
                trial[r:r + h, c:c + w] += 1
                rem = [j for j in range(n) if not placed[j] and j != i]
                bx = self.obj.commit(i, box.copy(), c + w / 2.0, r + h / 2.0)
                if self._admissible(trial, rem, bx):
                    chosen = (i, r, c, w, h, trial, bx)
                    break
            if chosen is None:
                fallbacks += 1
                sc, i, r, c, w, h = cands[0]
                trial = occ.copy()
                trial[r:r + h, c:c + w] += 1
                bx = self.obj.commit(i, box.copy(), c + w / 2.0, r + h / 2.0)
                chosen = (i, r, c, w, h, trial, bx)

            i, r, c, w, h, occ, box = chosen
            placed[i] = True
            X[i], Y[i] = c + w / 2.0, r + h / 2.0
            ex.append(c + w)
            ey.append(r + h)

        done = bool(placed.all())
        hpwl = self.obj.total(X, Y) if done else float("inf")
        used = float(occ[:self.I.H, :self.I.W].sum()) / float(self.I.W * self.I.H)
        return {"x": X, "y": Y, "placed": placed, "complete": done,
                "hpwl": hpwl, "fallbacks": fallbacks, "utilization": used,
                "filter_calls": self.n_filter_calls,
                "wall_time": time.time() - t0}
