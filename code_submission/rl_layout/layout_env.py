"""
Vectorized (numpy) re-implementation of the combinatorial-optimization objective
used in comb_opti_layout/{comb_high.py, hospital.py}.

The point is that RL and SA optimize *exactly* the same scalar objective, so the
comparison is apples-to-apples.  All constants (room list, sizes, affinity table,
weights, grid size) are imported from the original scripts rather than copied.
"""

import importlib.util
import math
import os
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

COMB_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "comb_opti_layout",
    "comb_opti_layout",
)


def load_reference_module(name: str):
    """Import comb_high.py / hospital.py as a module (their main() is guarded)."""
    path = os.path.join(COMB_DIR, f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"_ref_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@dataclass
class Spec:
    name: str
    room_order: list
    room_types: list
    w: np.ndarray          # (n,)
    h: np.ndarray          # (n,)
    adj: np.ndarray        # (n, n) affinity, 0 diagonal
    grid_w: float
    grid_h: float
    step_size: float
    adj_weight: float
    edge_weight: float
    overlap_weight: float
    no_overlap_bonus: float
    no_overlap_middle_bonus: float
    overlap_bonus_threshold: float
    distance_power: float
    # per-room wall bonuses, probed directly from the legacy scenario implementation
    edge_side: np.ndarray   # (n,) bonus for touching the left or right wall
    edge_bottom: np.ndarray  # (n,) bonus for touching the bottom wall
    edge_top: np.ndarray     # (n,) bonus for touching the top wall
    flip_adj: bool = False   # True -> affinity signs negated (see build_spec)


def _probe_edge_bonuses(m, room_order):
    """Recover the per-room wall bonuses by calling the reference function on
    isolated placements, so the vectorized version can never drift from it."""
    n = len(room_order)
    side = np.zeros(n)
    bottom = np.zeros(n)
    top = np.zeros(n)

    for i, name in enumerate(room_order):
        w, h = m.ROOM_SIZES[name]
        w, h = float(w), float(h)
        # every other room parked far outside the domain with zero area
        far = {o: m.Room(-1e4, -1e4, 0.0, 0.0) for o in room_order if o != name}

        def bonus(x, y):
            r = m.Room(x, y, w, h)
            return m.edge_bonus_for_room(name, r, {**far, name: r})

        cx, cy = m.GRID_W / 2.0, m.GRID_H / 2.0
        side[i] = bonus(w / 2.0, cy)                     # touching left only
        bottom[i] = bonus(cx, h / 2.0)                   # touching bottom only
        top[i] = bonus(cx, m.GRID_H - h / 2.0)           # touching top only
    return side, bottom, top


def build_spec(module_name: str, flip_adj: bool = False) -> Spec:
    """`flip_adj=True` negates the affinity matrix.

    `adjacency_term()` rewards `wij * nd` for `wij >= 0`, where `nd` is the
    normalized *distance* — so as written a positive weight pushes a pair apart,
    the opposite of what the scenario files' own comments say
    ("positive -> prefer close").  Negating the matrix restores the intended
    reading without touching the objective code: an intended-close pair becomes
    negative and is scored `|wij| * (1 - nd)`, which is maximized when close.
    The ceiling `2 * sum |wij|` is unchanged, so scores stay comparable.
    """
    m = load_reference_module(module_name)
    room_order = list(m.ROOM_ORDER)
    n = len(room_order)
    room_types = [m.ROOM_TYPES[r] for r in room_order]

    w = np.array([float(m.ROOM_SIZES[r][0]) for r in room_order], dtype=np.float64)
    h = np.array([float(m.ROOM_SIZES[r][1]) for r in room_order], dtype=np.float64)

    adj = np.zeros((n, n), dtype=np.float64)
    for i, ri in enumerate(room_order):
        for j, rj in enumerate(room_order):
            if i != j:
                adj[i, j] = float(m.ADJ[ri][rj])
    if flip_adj:
        adj = -adj

    edge_side, edge_bottom, edge_top = _probe_edge_bonuses(m, room_order)

    return Spec(
        name=module_name,
        room_order=room_order,
        room_types=room_types,
        w=w, h=h, adj=adj,
        grid_w=float(m.GRID_W), grid_h=float(m.GRID_H),
        step_size=float(m.STEP_SIZE),
        adj_weight=float(m.ADJ_WEIGHT),
        edge_weight=float(m.EDGE_WEIGHT),
        overlap_weight=float(m.OVERLAP_WEIGHT),
        no_overlap_bonus=float(m.NO_OVERLAP_BONUS),
        no_overlap_middle_bonus=float(m.NO_OVERLAP_MIDDLE_BONUS),
        overlap_bonus_threshold=float(m.OVERLAP_BONUS_THRESHOLD),
        distance_power=float(m.DISTANCE_POWER),
        edge_side=edge_side, edge_bottom=edge_bottom, edge_top=edge_top,
        flip_adj=flip_adj,
    )


# ----------------------------------------------------------------------------
# Vectorized geometry
# ----------------------------------------------------------------------------

def _junk_distance(tan, w, h):
    """Distance from a rectangle centre to its boundary along a ray of slope `tan`."""
    hw = w / 2.0
    hh = h / 2.0
    ratio = h / w
    use_h = tan > ratio
    kw = np.where(use_h, np.where(np.isinf(tan), 0.0, hh / np.where(tan == 0, 1.0, tan)), hw)
    kh = np.where(use_h, hh, hw * np.where(np.isinf(tan), 0.0, tan))
    return np.sqrt(kw * kw + kh * kh)


class LayoutObjective:
    """Batched evaluation of the SA objective. x, y have shape (B, n)."""

    def __init__(self, spec: Spec):
        self.s = spec
        n = len(spec.room_order)
        self.n = n
        self.feat_dim = 8 + 5 * n      # geometry(4) + boundary(4) + 5 relational per room
        self.w = spec.w
        self.h = spec.h
        self.iu = np.triu_indices(n, k=1)

        self.xmin = spec.w / 2.0
        self.xmax = spec.grid_w - spec.w / 2.0
        self.ymin = spec.h / 2.0
        self.ymax = spec.grid_h - spec.h / 2.0

        # d_max for every pair: room i pinned at the lower-left corner, room j at
        # the upper-right corner.  Depends only on the sizes -> precompute once.
        ax = self.xmin[:, None] * np.ones((1, n))
        ay = self.ymin[:, None] * np.ones((1, n))
        bx = np.ones((n, 1)) * self.xmax[None, :]
        by = np.ones((n, 1)) * self.ymax[None, :]
        self.dmax = self._pair_distance(
            ax, ay, self.w[:, None] * np.ones((1, n)), self.h[:, None] * np.ones((1, n)),
            bx, by, np.ones((n, 1)) * self.w[None, :], np.ones((n, 1)) * self.h[None, :],
        )
        self.dmax = np.maximum(self.dmax, 1e-12)

    @staticmethod
    def _pair_distance(x1, y1, w1, h1, x2, y2, w2, h2):
        dx = x2 - x1
        dy = y2 - y1
        adx = np.abs(dx)
        ady = np.abs(dy)
        with np.errstate(divide="ignore", invalid="ignore"):
            tan = np.where(adx <= 1e-12, np.inf, ady / np.where(adx <= 1e-12, 1.0, adx))
        jd1 = _junk_distance(tan, w1, h1)
        jd2 = _junk_distance(tan, w2, h2)
        return np.sqrt(dx * dx + dy * dy) - jd1 - jd2

    def clip(self, x, y):
        return (np.clip(x, self.xmin, self.xmax), np.clip(y, self.ymin, self.ymax))

    def overlap_matrix(self, x, y):
        """(B, n, n) pairwise overlap areas (diagonal is garbage, never used)."""
        hw = self.w / 2.0
        hh = self.h / 2.0
        x0 = x - hw
        x1 = x + hw
        y0 = y - hh
        y1 = y + hh
        ox = np.minimum(x1[:, :, None], x1[:, None, :]) - np.maximum(x0[:, :, None], x0[:, None, :])
        oy = np.minimum(y1[:, :, None], y1[:, None, :]) - np.maximum(y0[:, :, None], y0[:, None, :])
        return np.maximum(ox, 0.0) * np.maximum(oy, 0.0)

    def evaluate(self, x, y, components=False):
        """x, y: (B, n) -> total objective (B,) matching comb_high.objective()."""
        B, n = x.shape
        ov = self.overlap_matrix(x, y)
        iu0, iu1 = self.iu
        ov_pairs = ov[:, iu0, iu1]                       # (B, P)
        overlap_area = ov_pairs.sum(axis=1)

        # --- overlap term (penalty + no-overlap bonus) ---
        penalty = self.s.overlap_weight * overlap_area
        thr = self.s.overlap_bonus_threshold
        bonus = np.where(
            np.abs(penalty) <= 1e-12,
            self.s.no_overlap_bonus,
            np.where(
                (penalty >= thr) & (penalty < 0.0),
                self.s.no_overlap_middle_bonus * (1.0 - penalty / thr) ** 2,
                0.0,
            ),
        )
        overlap_term = penalty + bonus

        # --- adjacency term (skips overlapping pairs) ---
        xi = x[:, iu0]; yi = y[:, iu0]
        xj = x[:, iu1]; yj = y[:, iu1]
        wi = self.w[iu0]; hi = self.h[iu0]
        wj = self.w[iu1]; hj = self.h[iu1]
        d = self._pair_distance(xi, yi, wi, hi, xj, yj, wj, hj)
        nd = np.clip(d / self.dmax[iu0, iu1], 0.0, 1.0) ** self.s.distance_power
        aij = self.s.adj[iu0, iu1]
        contrib = np.where(aij >= 0, aij * nd, (-aij) * (1.0 - nd))
        contrib = np.where(ov_pairs > 0.0, 0.0, contrib)
        adj_term = self.s.adj_weight * contrib.sum(axis=1)

        # --- edge term (rooms that overlap anything get nothing) ---
        ov_self = ov.copy()
        idx = np.arange(n)
        ov_self[:, idx, idx] = 0.0
        clean = ov_self.sum(axis=2) <= 0.0                # (B, n)

        tol = 1e-9
        touch_l = np.abs((x - self.w / 2.0) - 0.0) < tol
        touch_r = np.abs((x + self.w / 2.0) - self.s.grid_w) < tol
        touch_b = np.abs((y - self.h / 2.0) - 0.0) < tol
        touch_t = np.abs((y + self.h / 2.0) - self.s.grid_h) < tol

        bonus_room = ((touch_l | touch_r) * self.s.edge_side[None, :]
                      + touch_b * self.s.edge_bottom[None, :]
                      + touch_t * self.s.edge_top[None, :])
        edge_term = (bonus_room * clean).sum(axis=1)

        total = adj_term + edge_term + overlap_term
        if components:
            return total, {
                "adj": adj_term,
                "edge": edge_term,
                "overlap": overlap_term,
                "overlap_area": overlap_area,
            }
        return total

    # ------------------------------------------------------------------
    # observation
    # ------------------------------------------------------------------
    def observation(self, x, y):
        """Per-room feature blocks, shape (B, n, 8 + 4n), as in the paper."""
        B, n = x.shape
        L = max(self.s.grid_w, self.s.grid_h)
        gx, gy = self.s.grid_w, self.s.grid_h

        geom = np.stack([x / gx, y / gy,
                         np.broadcast_to(self.w / gx, (B, n)),
                         np.broadcast_to(self.h / gy, (B, n))], axis=-1)

        d_left = (x - self.w / 2.0) / gx
        d_right = (gx - (x + self.w / 2.0)) / gx
        d_bot = (y - self.h / 2.0) / gy
        d_top = (gy - (y + self.h / 2.0)) / gy
        bnd = np.stack([d_left, d_right, d_bot, d_top], axis=-1)

        dx = x[:, None, :] - x[:, :, None]
        dy = y[:, None, :] - y[:, :, None]
        dist = np.sqrt(dx * dx + dy * dy)
        denom = math.sqrt(gx * gx + gy * gy)
        ov = self.overlap_matrix(x, y)
        idx = np.arange(n)
        ov[:, idx, idx] = 0.0
        norm = np.maximum(dist, 1e-8)
        aij = np.broadcast_to(self.s.adj / max(np.abs(self.s.adj).max(), 1.0), (B, n, n))
        rel = np.stack([dist / denom,
                        ov / (self.w[:, None] * self.h[:, None]).T.mean(),
                        dx / norm, dy / norm, aij], axis=-1)   # (B, n, n, 5)
        rel = rel.reshape(B, n, 5 * n)

        return np.concatenate([geom, bnd, rel], axis=-1).astype(np.float32)


class LayoutBatchEnv:
    """B independent layout instances stepped in lock-step."""

    def __init__(self, spec: Spec, batch: int, horizon: int, seed: int = 0,
                 action_mode: str = "discrete", delta_max: float = 1.0,
                 reward_scale: float = 10.0, step_scale: float = 0.5):
        self.obj = LayoutObjective(spec)
        self.s = spec
        self.n = len(spec.room_order)
        self.B = batch
        self.horizon = horizon
        self.action_mode = action_mode
        self.delta_max = delta_max
        self.reward_scale = reward_scale
        self.step_scale = step_scale
        self.rng = np.random.default_rng(seed)
        self.x = None
        self.y = None
        self.t = 0

    def reset(self, x=None, y=None):
        if x is None:
            self.x = self.rng.uniform(self.obj.xmin, self.obj.xmax, size=(self.B, self.n))
            self.y = self.rng.uniform(self.obj.ymin, self.obj.ymax, size=(self.B, self.n))
        else:
            self.x = x.copy()
            self.y = y.copy()
        self.t = 0
        self.cur = self.obj.evaluate(self.x, self.y)
        self.best = self.cur.copy()
        self.best_x = self.x.copy()
        self.best_y = self.y.copy()
        return self.obj.observation(self.x, self.y)

    _DIRS = np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])

    def step(self, room, move):
        """room: (B,) int.  move: (B,) int direction for discrete, (B,2) float for continuous."""
        b = np.arange(self.B)
        if self.action_mode == "discrete":
            d = self._DIRS[move] * self.s.step_size * self.step_scale
        else:
            d = np.clip(move, -self.delta_max, self.delta_max)

        self.x[b, room] = np.clip(self.x[b, room] + d[:, 0],
                                  self.obj.xmin[room], self.obj.xmax[room])
        self.y[b, room] = np.clip(self.y[b, room] + d[:, 1],
                                  self.obj.ymin[room], self.obj.ymax[room])

        new = self.obj.evaluate(self.x, self.y)
        reward = (new - self.cur) / self.reward_scale
        self.cur = new

        improved = new > self.best
        self.best = np.where(improved, new, self.best)
        self.best_x[improved] = self.x[improved]
        self.best_y[improved] = self.y[improved]

        self.t += 1
        done = self.t >= self.horizon
        return self.obj.observation(self.x, self.y), reward, done
