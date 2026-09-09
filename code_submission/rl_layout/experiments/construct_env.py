"""
Constructive placement: build the layout one room at a time, on the lattice,
choosing only from positions that are already known not to overlap anything.

Why this and not the improvement MDP
------------------------------------
In the improvement formulation almost the whole sample budget is spent
discovering that rooms must not overlap -- a hard constraint the reward can only
hint at through a penalty.  legacy's PPO never got past 38% feasibility even after
3M steps and a greedy polish, and the +400 no-overlap bonus it was chasing is a
discontinuity that policy gradients cannot climb.

Here feasibility is a property of the action set: the candidate positions are
masked by an integral-image occupancy test, so *every* rollout ends
overlap-free and collects the bonus for free.  What the policy has to learn is
the part that is actually a decision problem -- which room to commit next and
where, given that the choice permanently constrains everything after it.

Cost: one placement per step, n steps per layout, and the objective is evaluated
n times in total.  The classical solvers need tens of thousands of evaluations
for one layout, which is the comparison this whole exercise is about.
"""

import numpy as np

import core

N_PLANES = 7
N_GLOB_C = 4


class ConstructEnv:
    """B instances; each episode places all n rooms, one per step."""

    def __init__(self, inst, seed=0, resample=None, order="policy", canvas=None):
        self.canvas = canvas          # fixed canvas so buffers survive resampling
        self.set_instance(inst)
        self.rng = np.random.default_rng(seed)
        self.resample = resample
        self.order = order                 # "policy" | "area" | "random"

    def set_instance(self, inst):
        self.inst = inst
        self.obj = core.BatchObjective(inst)
        self.B, self.n = inst.B, inst.n
        self.GW = self.canvas or int(inst.gw.max())
        self.GH = self.canvas or int(inst.gh.max())
        assert self.GW >= inst.gw.max() and self.GH >= inst.gh.max(), \
            "canvas smaller than an instance boundary"
        self.iw = np.rint(inst.w).astype(np.int64)
        self.ih = np.rint(inst.h).astype(np.int64)
        self.igw = np.rint(inst.gw).astype(np.int64)
        self.igh = np.rint(inst.gh).astype(np.int64)
        w = inst.weights
        self.ceil = np.maximum(self.obj.adj_ceiling + self.obj.edge_ceiling
                               + w.no_overlap_bonus, 1.0)
        self.amax = np.maximum(np.abs(inst.adj).max(axis=(1, 2)), 1.0)
        self.feat_dim = 8 + 3 * self.n
        # canvas coordinate planes, shared by every instance
        yy, xx = np.meshgrid(np.arange(self.GH), np.arange(self.GW), indexing="ij")
        self._cx = (xx / max(self.GW - 1, 1)).astype(np.float32)
        self._cy = (yy / max(self.GH - 1, 1)).astype(np.float32)

    # ------------------------------------------------------------------
    def reset(self, new_instance=False):
        if new_instance and self.resample is not None:
            self.set_instance(self.resample(self.rng))
        B, n = self.B, self.n
        self.px = np.zeros((B, n), dtype=np.int64)
        self.py = np.zeros((B, n), dtype=np.int64)
        self.placed = np.zeros((B, n), dtype=bool)
        self.owner = np.full((B, self.GH, self.GW), -1, dtype=np.int16)
        self.occ = np.zeros((B, self.GH, self.GW), dtype=np.int32)
        self.t = 0
        self.evals = 0
        self.J = self._objective()
        self.J0 = self.J.copy()
        if self.order != "policy":
            key = (self.inst.w * self.inst.h) if self.order == "area" else \
                  self.rng.random((B, n))
            self.fixed_order = np.argsort(-key, axis=1)
        return self.room_obs(), self.globals()

    def forced_room(self):
        """Which room the environment commits next, or None if the policy decides."""
        return None if self.order == "policy" else self.fixed_order[:, self.t]

    def centers(self):
        return (self.px + self.inst.w / 2.0, self.py + self.inst.h / 2.0)

    def _objective(self):
        x, y = self.centers()
        self.evals += self.B
        return self.obj.evaluate_masked(x, y, self.placed)

    # ------------------------------------------------------------------
    def room_obs(self):
        """(B, n, 8 + 3n) per-room features for the room-selection head."""
        B, n = self.B, self.n
        gw = self.inst.gw[:, None]
        gh = self.inst.gh[:, None]
        pl = self.placed.astype(np.float32)
        es = self.inst.edge_side / np.maximum(self.inst.edge_side.max(axis=1, keepdims=True), 1e-6)
        eb = self.inst.edge_bottom / np.maximum(self.inst.edge_bottom.max(axis=1, keepdims=True), 1e-6)
        et = self.inst.edge_top / np.maximum(self.inst.edge_top.max(axis=1, keepdims=True), 1e-6)
        x, y = self.centers()
        per = np.stack([self.inst.w / gw, self.inst.h / gh, pl,
                        np.where(self.placed, x / gw, 0.0),
                        np.where(self.placed, y / gh, 0.0),
                        es, eb, et], axis=-1)
        dx = x[:, None, :] - x[:, :, None]
        dy = y[:, None, :] - y[:, :, None]
        D = np.sqrt(self.inst.gw ** 2 + self.inst.gh ** 2)[:, None, None]
        both = (self.placed[:, None, :] & self.placed[:, :, None]).astype(np.float32)
        rel = np.stack([self.inst.adj / self.amax[:, None, None],
                        np.broadcast_to(pl[:, None, :], (B, n, n)),
                        np.sqrt(dx * dx + dy * dy) / D * both], axis=-1).reshape(B, n, 3 * n)
        return np.concatenate([per, rel], axis=-1).astype(np.float32)

    def globals(self):
        area = (self.inst.w * self.inst.h)
        return np.stack([
            np.full(self.B, self.t / self.n),
            self.placed.mean(axis=1),
            (area * self.placed).sum(axis=1) / (self.inst.gw * self.inst.gh),
            self.J / self.ceil,
        ], axis=-1).astype(np.float32)

    # ------------------------------------------------------------------
    def _window_sums(self, room):
        """(B, GH, GW) occupied area of the w_i x h_i window with lower-left at
        (px, py), via an integral image.  Positions that would fall outside the
        instance's own boundary are marked invalid."""
        B = self.B
        ii = np.zeros((B, self.GH + 1, self.GW + 1), dtype=np.int32)
        np.cumsum(np.cumsum(self.occ, axis=1), axis=2, out=ii[:, 1:, 1:])
        b = np.arange(B)
        w = self.iw[b, room]
        h = self.ih[b, room]
        r0 = np.arange(self.GH)[None, :, None]
        c0 = np.arange(self.GW)[None, None, :]
        r1 = np.minimum(r0 + h[:, None, None], self.GH)
        c1 = np.minimum(c0 + w[:, None, None], self.GW)
        bb = b[:, None, None]
        S = (ii[bb, r1, c1] - ii[bb, r0, c1] - ii[bb, r1, c0] + ii[bb, r0, c0])
        valid = ((c0 + w[:, None, None] <= self.igw[b][:, None, None])
                 & (r0 + h[:, None, None] <= self.igh[b][:, None, None]))
        return S, valid

    def planes(self, room):
        """(B, N_PLANES, GH, GW) input for the position head, plus the legal mask."""
        B = self.B
        b = np.arange(B)
        S, valid = self._window_sums(room)
        legal = (S == 0) & valid

        # affinity of the room being placed towards whoever owns each cell
        occupied = self.owner >= 0
        own = np.clip(self.owner, 0, self.n - 1).astype(np.int64).reshape(B, -1)
        aff_row = self.inst.adj[b, room] / self.amax[:, None]          # (B, n)
        aff = (np.take_along_axis(aff_row, own, axis=1).reshape(B, self.GH, self.GW)
               * occupied).astype(np.float32)

        w = self.iw[b, room][:, None, None]
        h = self.ih[b, room][:, None, None]
        c0 = np.arange(self.GW)[None, None, :]
        r0 = np.arange(self.GH)[None, :, None]
        gw = self.igw[b][:, None, None]
        gh = self.igh[b][:, None, None]
        touch_side = (c0 == 0) | (c0 + w == gw)
        edge = (touch_side * self.inst.edge_side[b, room][:, None, None]
                + (r0 == 0) * self.inst.edge_bottom[b, room][:, None, None]
                + (r0 + h == gh) * self.inst.edge_top[b, room][:, None, None])
        escale = np.maximum(np.abs(self.inst.edge_side).max(axis=1)
                            + np.abs(self.inst.edge_top).max(axis=1), 1e-6)[:, None, None]

        P = np.stack([
            occupied.astype(np.float32),
            legal.astype(np.float32),
            aff,
            (edge / escale).astype(np.float32),
            valid.astype(np.float32),
            np.broadcast_to(self._cx, (B, self.GH, self.GW)),
            np.broadcast_to(self._cy, (B, self.GH, self.GW)),
        ], axis=1).astype(np.float32)
        # rooms with no legal position fall back to the least-overlapping one
        dead = ~legal.reshape(B, -1).any(axis=1)
        fallback = np.where(valid, -S, -10 ** 6)
        return P, legal, dead, fallback

    # ------------------------------------------------------------------
    def step(self, room, flat_pos):
        B = self.B
        b = np.arange(B)
        py, px = flat_pos // self.GW, flat_pos % self.GW
        self.px[b, room] = px
        self.py[b, room] = py
        self.placed[b, room] = True
        for k in range(B):
            i = room[k]
            self.owner[k, py[k]:py[k] + self.ih[k, i], px[k]:px[k] + self.iw[k, i]] = i
            self.occ[k, py[k]:py[k] + self.ih[k, i], px[k]:px[k] + self.iw[k, i]] += 1

        prev = self.J
        self.J = self._objective()
        rew = ((self.J - prev) / self.ceil * 100.0).astype(np.float32)
        self.t += 1
        done = self.t >= self.n
        return self.room_obs(), self.globals(), rew, done

    def result(self):
        x, y = self.centers()
        tot, c = self.obj.evaluate(x, y, components=True)
        return x, y, tot, c
