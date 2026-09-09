"""
The improvement MDP, with the three things legacy's formulation was missing.

1. Reward.  legacy used r_t = J(s_{t+1}) - J(s_t).  That telescopes to J(s_T) - J(s_0),
   so the return is the *final* state's objective -- while evaluation reports the
   best state ever visited.  Worse, any move that temporarily worsens the layout
   must be paid for in full, so with gamma < 1 exploration is strictly
   net-negative and the optimal policy collapses to one-step greedy ascent.
   legacy's own discount sweep shows exactly that: gamma=0 scores 2409, gamma=0.995
   scores 1330.  Here r_t = max(0, J(s_t) - max_{k<t} J(s_k)) instead.  It
   telescopes to the quantity actually reported, and a move that goes downhill
   now to reach a new record later costs nothing, so escaping a local optimum is
   something the policy can learn rather than something it is punished for.

2. Move set.  legacy's agent could only translate one room by a fixed step, while the
   SA it was compared against also had a 20%-probability teleport.  Here the
   policy gets translate / jump / swap / snap.  Swap matters most: two rooms that
   want to exchange places are separated by a large-overlap barrier that no
   sequence of single-cell translations can cross at a profit.

3. Accept/reject.  In legacy every sampled action was applied unconditionally, so an
   imperfect policy random-walks downhill, whereas SA keeps a monotone incumbent.
   `accept="metropolis"` wraps the policy in the same accept/reject ratchet SA
   uses, turning the agent into a learned proposal distribution for annealing.

Objective evaluations per environment step is exactly 1 in every mode, so budgets
remain directly comparable with the classical solvers.
"""

import numpy as np

import core
from nets import OP_TRANSLATE, OP_JUMP, OP_SWAP, OP_SNAP

N_GLOB = 5


class ImproveEnv:
    """B layout instances stepped in lock-step."""

    def __init__(self, inst, horizon, seed=0, action_mode="lattice", delta_max=3.0,
                 reward="best", accept="always", t0=8.0, t1=0.05, resample=None,
                 soft_start=0.0, init_grid=None):
        self.inst = inst
        self.obj = core.BatchObjective(inst)
        self.B, self.n = inst.B, inst.n
        self.horizon = horizon
        self.rng = np.random.default_rng(seed)
        self.action_mode = action_mode
        self.lattice = action_mode == "lattice"
        self.delta_max = delta_max
        self.reward_mode = reward
        self.accept = accept
        self.t0, self.t1 = t0, t1
        self.resample = resample                 # callable(rng) -> InstanceBatch
        self.soft = soft_start
        self.init_grid = self.lattice if init_grid is None else init_grid
        # per-instance normalizer so instances of different size are comparable
        self._set_scale()
        self.t = 0

    def _set_scale(self):
        w = self.inst.weights
        self.ceil = (self.obj.adj_ceiling + self.obj.edge_ceiling + w.no_overlap_bonus)
        self.ceil = np.maximum(self.ceil, 1.0)

    @property
    def ops(self):
        return (OP_TRANSLATE, OP_JUMP, OP_SWAP) if self.lattice else \
               (OP_TRANSLATE, OP_JUMP, OP_SWAP, OP_SNAP)

    # ------------------------------------------------------------------
    def reset(self, x=None, y=None, new_instance=False):
        if new_instance and self.resample is not None:
            self.inst = self.resample(self.rng)
            self.obj = core.BatchObjective(self.inst)
            self._set_scale()
        if x is None:
            x, y = core.random_layout(self.obj, self.rng, grid_aligned=self.init_grid)
        self.x, self.y = x.copy(), y.copy()
        self.cur = self.obj.evaluate(self.x, self.y, soft_overlap=self.soft)
        self.best = self.cur.copy()
        self.best_x, self.best_y = self.x.copy(), self.y.copy()
        self.t = 0
        self.evals = self.B
        return self.observation()

    def temperature(self):
        frac = self.t / max(1, self.horizon - 1)
        return self.t0 * ((self.t1 / self.t0) ** frac)

    def globals(self):
        gap = (self.best - self.cur) / self.ceil
        # overlap area directly, not via evaluate(): calling the objective here
        # would double the evaluation count the method is charged for, and the
        # budget is the axis the whole comparison is reported on
        ov = self.obj.overlap_pairs(self.x, self.y).sum(axis=1)
        ovf = ov / (self.inst.w * self.inst.h).sum(axis=1)
        return np.stack([
            np.full(self.B, self.t / self.horizon),
            np.clip(gap, 0, 1),
            np.clip(self.best / self.ceil, -1, 1),
            np.clip(ovf, 0, 1),
            np.full(self.B, np.log(max(self.temperature(), 1e-6)) / 5.0
                    if self.accept == "metropolis" else 0.0),
        ], axis=-1).astype(np.float32)

    def observation(self):
        return self.obj.observation(self.x, self.y), self.globals()

    # ------------------------------------------------------------------
    def _apply(self, a):
        """Returns candidate (x, y) after applying the factorized action."""
        b = np.arange(self.B)
        r = a["room"]
        cx, cy = self.x.copy(), self.y.copy()
        lo_x, hi_x = self.obj.xmin[b, r], self.obj.xmax[b, r]
        lo_y, hi_y = self.obj.ymin[b, r], self.obj.ymax[b, r]

        # --- translate ---
        d = np.clip(a["delta"], -1.0, 1.0) * self.delta_max
        if self.lattice:
            d = np.rint(d)
            d = np.where(np.all(d == 0, axis=1, keepdims=True),
                         np.array([[1.0, 0.0]]), d)     # never a pure no-op
        tx = np.clip(self.x[b, r] + d[:, 0], lo_x, hi_x)
        ty = np.clip(self.y[b, r] + d[:, 1], lo_y, hi_y)

        # --- jump: absolute target in normalized box coordinates ---
        u = 1.0 / (1.0 + np.exp(-np.clip(a["jump"], -8, 8)))
        jx = lo_x + u[:, 0] * (hi_x - lo_x)
        jy = lo_y + u[:, 1] * (hi_y - lo_y)
        if self.lattice:
            jx = np.clip(np.rint(jx - lo_x) + lo_x, lo_x, hi_x)
            jy = np.clip(np.rint(jy - lo_y) + lo_y, lo_y, hi_y)

        # --- swap the two rooms' centres, each clipped to its own feasible box ---
        j = a["swap"]
        sxi = np.clip(self.x[b, j], lo_x, hi_x)
        syi = np.clip(self.y[b, j], lo_y, hi_y)
        sxj = np.clip(self.x[b, r], self.obj.xmin[b, j], self.obj.xmax[b, j])
        syj = np.clip(self.y[b, r], self.obj.ymin[b, j], self.obj.ymax[b, j])

        # --- snap to the lattice (continuous mode only) ---
        nx = np.clip(np.rint(self.x[b, r] - lo_x) + lo_x, lo_x, hi_x)
        ny = np.clip(np.rint(self.y[b, r] - lo_y) + lo_y, lo_y, hi_y)

        op = a["op"]
        newx = np.select([op == OP_TRANSLATE, op == OP_JUMP, op == OP_SWAP, op == OP_SNAP],
                         [tx, jx, sxi, nx])
        newy = np.select([op == OP_TRANSLATE, op == OP_JUMP, op == OP_SWAP, op == OP_SNAP],
                         [ty, jy, syi, ny])
        cx[b, r] = newx
        cy[b, r] = newy
        is_swap = op == OP_SWAP
        cx[b, j] = np.where(is_swap, sxj, cx[b, j])
        cy[b, j] = np.where(is_swap, syj, cy[b, j])
        return cx, cy

    def step(self, a):
        cx, cy = self._apply(a)
        cand = self.obj.evaluate(cx, cy, soft_overlap=self.soft)
        self.evals += self.B
        prev_cur = self.cur
        prev_best = self.best.copy()

        if self.accept == "metropolis":
            temp = max(self.temperature(), 1e-9)
            delta = (cand - prev_cur) / self.ceil * 100.0
            ok = (delta >= 0) | (self.rng.random(self.B)
                                 < np.exp(np.clip(delta / temp, -50, 0)))
        else:
            ok = np.ones(self.B, dtype=bool)

        self.x = np.where(ok[:, None], cx, self.x)
        self.y = np.where(ok[:, None], cy, self.y)
        self.cur = np.where(ok, cand, prev_cur)

        imp = self.cur > self.best
        self.best = np.where(imp, self.cur, self.best)
        self.best_x[imp] = self.x[imp]
        self.best_y[imp] = self.y[imp]

        if self.reward_mode == "best":
            rew = (self.best - prev_best) / self.ceil
        else:                                    # legacy's delta-of-objective reward
            rew = (self.cur - prev_cur) / self.ceil

        self.t += 1
        done = self.t >= self.horizon
        obs, glob = self.observation()
        return obs, glob, (rew * 100.0).astype(np.float32), done

    # ------------------------------------------------------------------
    def exact_best(self):
        """Objective of the incumbent under the *exact* objective (soft off)."""
        tot, c = self.obj.evaluate(self.best_x, self.best_y, components=True)
        return tot, c
