"""
Viability filters for the masked constructive MDP.

The mask used everywhere else in this work answers a one-step question: does this
placement overlap anything already placed?  A state can pass that test and still
be doomed, because some facility placed later will have nowhere legal to go.  The
set of states from which a *complete* feasible layout still exists is the
viability kernel V; the one-step mask only certifies membership in the strictly
larger set M of states with at least one legal action.

Deciding membership in V exactly is a packing-feasibility question and therefore
NP-hard, so this module brackets it instead, which is both cheaper and cleaner to
reason about:

    N1, N_area   necessary conditions -- if they fail, the state is certainly
                 outside V, so rejecting is always sound;
    S            a sufficient condition -- an explicit packing of every
                 remaining facility, so passing it certifies membership in V.

True viability lies between them.  The gap between what the necessary tests
reject and what the sufficient test accepts is exactly the region where a cheap
exact answer is unavailable, and it is the region a learned filter has to cover.

Each filter is used as a rejection rule: candidate placements are considered in
decreasing order of exact marginal gain and the first one whose resulting state
passes the filter is committed.  Only the top `n_cand` candidates are tested,
because the sufficient test costs a packing attempt per candidate; that cap is a
reported parameter, not a hidden one.
"""

import time

import numpy as np

import core
from greedy_construct import GreedyConstructor


def _integral(occ):
    ii = np.zeros((occ.shape[0] + 1, occ.shape[1] + 1), dtype=np.int64)
    np.cumsum(np.cumsum(occ, axis=0), axis=1, out=ii[1:, 1:])
    return ii


def _legal_mask(ii, GH, GW, w, h, gw, gh):
    """(GH, GW) True where a w x h box fits with no overlap and inside the plate."""
    rows = np.arange(GH)[:, None]
    cols = np.arange(GW)[None, :]
    r1 = np.minimum(rows + h, GH)
    c1 = np.minimum(cols + w, GW)
    S = ii[r1, c1] - ii[rows, c1] - ii[r1, cols] + ii[rows, cols]
    inside = (cols + w <= gw) & (rows + h <= gh)
    return (S == 0) & inside


def complete_random(occ, remaining, iw, ih, GH, GW, gw, gh, tries, rng):
    """Try `tries` randomized largest-first completions; True if any succeeds.

    This is the randomized multistart certificate used as a comparison method.
    The learned diagnostic is instead labelled by `_complete_greedy` below so
    that its target is the constructor's own base-policy rollout.  Deciding
    viability exactly is NP-hard, but a completion procedure with
    restarts is a computable lower bound on it that gets tighter with `tries`,
    and unlike the single first-fit certificate it is not systematically
    conservative.  Its cost is `tries` sequential packing attempts, which is
    exactly the expense a one-shot predictor would amortize.
    """
    order = sorted(remaining, key=lambda i: -(iw[i] * ih[i]))
    for t in range(tries):
        work = occ.copy()
        ok = True
        for i in order:
            m = _legal_mask(_integral(work), GH, GW, iw[i], ih[i], gw, gh)
            idx = np.argwhere(m)
            if not len(idx):
                ok = False
                break
            # first attempt is deterministic first-fit; later ones randomize the
            # position, which is where the extra coverage comes from
            r, c = idx[0] if t == 0 else idx[rng.integers(len(idx))]
            work[r:r + ih[i], c:c + iw[i]] += 1
        if ok:
            return True
    return False


def state_features(occ, remaining, iw, ih, GH, GW, gw, gh):
    """A fixed-length description of how constrained a partial layout is.

    Everything here is computable from integral images in O(|remaining| * WH),
    the same order as a *single* completion attempt and far below `tries` of
    them.  The features deliberately describe free space and its usability
    rather than the objective: this predictor's only job is feasibility.
    """
    ii = _integral(occ)
    plate = float(gw * gh)
    free = plate - float(occ[:gh, :gw].sum())
    need = float(sum(iw[i] * ih[i] for i in remaining)) if remaining else 0.0
    counts = []
    for i in remaining:
        m = _legal_mask(ii, GH, GW, iw[i], ih[i], gw, gh)
        counts.append(float(m.sum()))
    counts = np.array(counts) if counts else np.zeros(1)
    areas = np.array([iw[i] * ih[i] for i in remaining], dtype=float) \
        if remaining else np.zeros(1)
    big = int(np.argmax(areas))
    norm = max(plate, 1.0)
    eps = 1e-9
    return np.array([
        len(remaining) / max(len(iw), 1),
        free / norm,
        need / max(free, eps),                      # slack: >1 means impossible
        np.log1p(counts.min()) / 10.0,
        np.log1p(counts.mean()) / 10.0,
        np.log1p(counts.max()) / 10.0,
        np.log1p(counts[big]) / 10.0,               # room for the largest piece
        float((counts == 0).sum()) / max(len(counts), 1),
        areas.max() / norm,
        areas.mean() / norm,
        float(np.std(counts)) / (counts.mean() + eps),
        min(free / norm, 1.0) * (1.0 - min(need / max(free, eps), 1.0)),
    ], dtype=np.float32)


N_FEATURES = 12


class ViabilityConstructor:
    """Greedy placement with a viability rejection filter.

    `filter_mode`:
      "none"  -- the one-step mask only (reproduces the greedy rule exactly)
      "n1"    -- reject if any remaining facility would have no legal position
      "area"  -- n1, plus reject if the largest free axis-aligned gap cannot hold
                 the largest remaining facility
      "pack"  -- reject unless an explicit first-fit packing of every remaining
                 facility into the remaining free space succeeds
    """

    def __init__(self, inst, canvas=None, filter_mode="none", n_cand=16,
                 room_rule="greedy", oracle_tries=8, net=None, threshold=0.0,
                 seed=0):
        assert inst.B == 1, "one instance at a time: the filters are sequential"
        self.I = inst
        self.gc = GreedyConstructor(inst, canvas=canvas)
        self.obj = core.BatchObjective(inst)
        self.n = inst.n
        self.GW, self.GH = self.gc.GW, self.gc.GH
        self.gw = int(round(inst.gw[0]))
        self.gh = int(round(inst.gh[0]))
        self.iw = self.gc.iw[0]
        self.ih = self.gc.ih[0]
        self.mode = filter_mode
        self.n_cand = n_cand
        self.room_rule = room_rule
        self.area_order = np.argsort(-(inst.w[0] * inst.h[0]))
        self.oracle_tries = oracle_tries
        self.net = net                    # (module, mu, sd) for filter_mode="learned"
        self.threshold = threshold
        self.rng = np.random.default_rng(seed)
        self.n_filter_calls = 0

    # ------------------------------------------------------------------
    def _complete_greedy(self, occ, remaining, placed, x, y):
        """Run the constructor's own base heuristic to completion; True if it
        finishes without overlap.

        This is the certificate whose witness the constructor can actually
        follow, which the experiments show matters more than how permissive the
        certificate is.  It is a rollout of the base policy in the sense of
        Bertsekas: the value of a candidate is the outcome the base heuristic
        obtains from it.  Cost is O(n^2 WH) per query -- an order more than
        first-fit -- which is what a learned predictor amortizes.
        """
        work = occ.copy()
        pl = placed.copy()
        xx, yy = x.copy(), y.copy()
        order = sorted(remaining, key=lambda i: -(self.iw[i] * self.ih[i]))
        for i in order:
            lg, _, _ = self.gc._legal(work[None], i)
            if not lg[0].any():
                return False
            g = self.gc._gain(i, xx, yy, pl[None])
            s = np.where(lg[0], g[0], -np.inf)
            f = int(np.argmax(s.reshape(-1)))
            r, c = f // self.GW, f % self.GW
            work[r:r + self.ih[i], c:c + self.iw[i]] += 1
            pl[i] = True
            xx[0, i] = c + self.I.w[0, i] / 2.0
            yy[0, i] = r + self.I.h[0, i] / 2.0
        return True

    def _feasible_after(self, occ, remaining, placed=None, x=None, y=None):
        """Does `occ` pass the configured filter, given the remaining facilities?"""
        if self.mode == "none" or not len(remaining):
            return True
        self.n_filter_calls += 1

        if self.mode == "rollout":
            return self._complete_greedy(occ, remaining, placed, x, y)

        if self.mode == "learned":
            # one forward pass on the free-space description; no packing attempt
            import torch
            net, mu, sd = self.net
            f = state_features(occ, remaining, self.iw, self.ih,
                               self.GH, self.GW, self.gw, self.gh)
            with torch.no_grad():
                z = torch.tensor((f - mu) / sd, dtype=torch.float32)[None]
                return bool(net(z).item() >= self.threshold)

        if self.mode.startswith("oracle"):
            # the expensive reference: randomized completion with restarts
            return complete_random(occ, remaining, self.iw, self.ih,
                                   self.GH, self.GW, self.gw, self.gh,
                                   self.oracle_tries, self.rng)

        ii = _integral(occ)
        legals = {}
        for i in remaining:
            m = _legal_mask(ii, self.GH, self.GW, self.iw[i], self.ih[i],
                            self.gw, self.gh)
            if not m.any():
                return False                     # N1 fails: certainly outside V
            legals[i] = m
        if self.mode == "n1":
            return True

        if self.mode == "area":
            # a necessary condition that N1 misses: free space may be fragmented
            # into pieces all too small for the largest remaining facility
            big = max(remaining, key=lambda i: self.iw[i] * self.ih[i])
            return bool(legals[big].any())

        # "pack": an explicit certificate.  Place the remaining facilities
        # largest-first at the first legal position; success proves the state is
        # in V, failure proves nothing but is used as the rejection signal.
        work = occ.copy()
        order = sorted(remaining, key=lambda i: -(self.iw[i] * self.ih[i]))
        for i in order:
            jj = _integral(work)
            m = _legal_mask(jj, self.GH, self.GW, self.iw[i], self.ih[i],
                            self.gw, self.gh)
            if not m.any():
                return False
            r, c = np.argwhere(m)[0]
            work[r:r + self.ih[i], c:c + self.iw[i]] += 1
        return True

    # ------------------------------------------------------------------
    def solve(self, rng=None):
        t0 = time.time()
        n = self.n
        placed = np.zeros(n, dtype=bool)
        occ = np.zeros((self.GH, self.GW), dtype=np.int32)
        x = np.zeros((1, n))
        y = np.zeros((1, n))
        cands_scored = 0
        filter_calls = 0
        rejects = 0

        for t in range(n):
            score = {}
            rooms = ([int(self.area_order[t])] if self.room_rule == "area"
                     else [i for i in range(n) if not placed[i]])
            for i in rooms:
                if placed[i]:
                    continue
                lg, S, valid = self.gc._legal(occ[None], i)
                g = self.gc._gain(i, x, y, placed[None])
                cands_scored += int(lg.sum())
                # scored exactly as GreedyConstructor does, including its
                # fallback: a facility with no legal position is still placed, at
                # the least-overlapping spot, so that stranding shows up as
                # overlap in the objective rather than as an unplaced facility.
                # Without this the no-filter case would not reproduce the greedy
                # rule on precisely the instances the experiment is about.
                score[i] = np.where(lg[0], g[0],
                                    -1e9 + np.where(valid[0], -S[0], -1e6))

            # Stable sorts throughout, so ties resolve to the lowest position
            # index and then the lowest facility index -- which is what
            # GreedyConstructor's argmax-then-strictly-greater comparison does.
            # With an unstable sort the no-filter case picks a different member
            # of a tied set and stops reproducing the greedy rule.
            flat = []
            for i, si in score.items():
                idx = np.argsort(-si, axis=None, kind="stable")[:self.n_cand]
                flat += [(float(si.reshape(-1)[f]), i, int(f)) for f in idx]
            if not flat:
                break
            flat.sort(key=lambda z: -z[0])
            flat = flat[:self.n_cand]

            chosen = None
            for v, i, f in flat:
                r, c = f // self.GW, f % self.GW
                trial = occ.copy()
                trial[r:r + self.ih[i], c:c + self.iw[i]] += 1
                rem = [j for j in range(n) if not placed[j] and j != i]
                filter_calls += 1
                pl_after = placed.copy()
                pl_after[i] = True
                x_after, y_after = x.copy(), y.copy()
                x_after[0, i] = c + self.I.w[0, i] / 2.0
                y_after[0, i] = r + self.I.h[0, i] / 2.0
                if self._feasible_after(trial, rem, pl_after, x_after, y_after):
                    chosen = (i, r, c, trial)
                    break
                rejects += 1
            if chosen is None:                   # nothing survives: take the best
                v, i, f = flat[0]
                r, c = f // self.GW, f % self.GW
                trial = occ.copy()
                trial[r:r + self.ih[i], c:c + self.iw[i]] += 1
                chosen = (i, r, c, trial)

            i, r, c, occ = chosen
            placed[i] = True
            x[0, i] = c + self.I.w[0, i] / 2.0
            y[0, i] = r + self.I.h[0, i] / 2.0

        tot, comp = self.obj.evaluate(x, y, components=True)
        return {"x": x, "y": y, "best": float(tot[0]),
                "overlap_area": float(comp["overlap_area"][0]),
                "feasible": bool(comp["overlap_area"][0] <= 1e-9),
                "candidates": cands_scored, "filter_calls": filter_calls,
                "rejects": rejects, "wall_time": time.time() - t0}
