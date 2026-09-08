"""
v2 core: a *per-instance* batched layout objective and a procedural instance
generator.

Why this exists
---------------
`../layout_env.py` evaluates B layouts of ONE fixed instance (fixed room set,
sizes, affinity table, boundary).  That is the regime where a metaheuristic must
win: an RL policy trained for 20 GPU-minutes on a single instance has nothing to
amortize over.  To test the only claim that could favour learning -- amortized
inference over a *distribution* of layout briefs -- the objective has to accept a
different instance per batch element.

`BatchObjective` below does exactly that: w, h, adjacency, boundary size and the
per-room wall bonuses are all (B, ...) arrays.  Everything else (the formulas,
the constants, the clipping conventions) is copied verbatim from the reference
implementation, and `verify.py` checks that with B identical instances it agrees
with `layout_env.LayoutObjective` to 0 abs error -- which in turn is checked
against the original pure-Python `objective()`.  So the chain of trust is:

    comb_high.objective()  ==  layout_env.LayoutObjective  ==  v2.BatchObjective
"""

import math
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# objective constants: shared defaults (comb_high); hospital overrides some
# ---------------------------------------------------------------------------


@dataclass
class Weights:
    adj_weight: float = 2.0
    edge_weight: float = 2.0
    overlap_weight: float = -8.0
    no_overlap_bonus: float = 400.0
    no_overlap_middle_bonus: float = 200.0
    overlap_bonus_threshold: float = -200.0
    distance_power: float = 1.0
    step_size: float = 1.0


@dataclass
class InstanceBatch:
    """B layout instances, each with n rooms (n shared, contents may differ)."""

    w: np.ndarray            # (B, n) room widths
    h: np.ndarray            # (B, n) room heights
    adj: np.ndarray          # (B, n, n) affinity, zero diagonal, symmetric
    gw: np.ndarray           # (B,) boundary width
    gh: np.ndarray           # (B,) boundary height
    edge_side: np.ndarray    # (B, n) bonus for touching left OR right wall
    edge_bottom: np.ndarray  # (B, n) bonus for touching the bottom wall
    edge_top: np.ndarray     # (B, n) bonus for touching the top wall
    type_id: np.ndarray      # (B, n) integer room-type id (diversity metric, embeddings)
    weights: Weights = field(default_factory=Weights)
    names: list = None       # optional, only for rendering

    @property
    def B(self):
        return self.w.shape[0]

    @property
    def n(self):
        return self.w.shape[1]

    def take(self, idx):
        """Sub-select instances (idx: array of ints or slice)."""
        return InstanceBatch(
            w=self.w[idx], h=self.h[idx], adj=self.adj[idx],
            gw=self.gw[idx], gh=self.gh[idx],
            edge_side=self.edge_side[idx], edge_bottom=self.edge_bottom[idx],
            edge_top=self.edge_top[idx], type_id=self.type_id[idx],
            weights=self.weights, names=self.names,
        )

    def repeat(self, k):
        """Each instance repeated k times, contiguously (i.e. np.repeat)."""
        return InstanceBatch(
            w=np.repeat(self.w, k, axis=0), h=np.repeat(self.h, k, axis=0),
            adj=np.repeat(self.adj, k, axis=0),
            gw=np.repeat(self.gw, k), gh=np.repeat(self.gh, k),
            edge_side=np.repeat(self.edge_side, k, axis=0),
            edge_bottom=np.repeat(self.edge_bottom, k, axis=0),
            edge_top=np.repeat(self.edge_top, k, axis=0),
            type_id=np.repeat(self.type_id, k, axis=0),
            weights=self.weights, names=self.names,
        )

    def fill_ratio(self):
        return (self.w * self.h).sum(axis=1) / (self.gw * self.gh)


def from_spec(spec, B):
    """Lift a `layout_env.Spec` (one fixed instance) into B identical copies."""
    n = len(spec.room_order)
    types = sorted(set(spec.room_types))
    tid = np.array([types.index(t) for t in spec.room_types])
    ones = np.ones((B, 1))
    return InstanceBatch(
        w=ones * spec.w[None, :], h=ones * spec.h[None, :],
        adj=np.broadcast_to(spec.adj, (B, n, n)).copy(),
        gw=np.full(B, spec.grid_w), gh=np.full(B, spec.grid_h),
        edge_side=ones * spec.edge_side[None, :],
        edge_bottom=ones * spec.edge_bottom[None, :],
        edge_top=ones * spec.edge_top[None, :],
        type_id=np.broadcast_to(tid, (B, n)).copy(),
        weights=Weights(
            adj_weight=spec.adj_weight, edge_weight=spec.edge_weight,
            overlap_weight=spec.overlap_weight,
            no_overlap_bonus=spec.no_overlap_bonus,
            no_overlap_middle_bonus=spec.no_overlap_middle_bonus,
            overlap_bonus_threshold=spec.overlap_bonus_threshold,
            distance_power=spec.distance_power, step_size=spec.step_size,
        ),
        names=list(spec.room_order),
    )


# ---------------------------------------------------------------------------
# geometry, verbatim from the legacy scenario implementation
# ---------------------------------------------------------------------------

def _junk_distance(tan, w, h):
    hw = w / 2.0
    hh = h / 2.0
    ratio = h / w
    use_h = tan > ratio
    kw = np.where(use_h, np.where(np.isinf(tan), 0.0, hh / np.where(tan == 0, 1.0, tan)), hw)
    kh = np.where(use_h, hh, hw * np.where(np.isinf(tan), 0.0, tan))
    return np.sqrt(kw * kw + kh * kh)


def _pair_distance(x1, y1, w1, h1, x2, y2, w2, h2):
    dx = x2 - x1
    dy = y2 - y1
    adx = np.abs(dx)
    ady = np.abs(dy)
    with np.errstate(divide="ignore", invalid="ignore"):
        tan = np.where(adx <= 1e-12, np.inf, ady / np.where(adx <= 1e-12, 1.0, adx))
    return np.sqrt(dx * dx + dy * dy) - _junk_distance(tan, w1, h1) - _junk_distance(tan, w2, h2)


class BatchObjective:
    """Evaluate B layouts, one per instance.  x, y have shape (B, n)."""

    def __init__(self, inst: InstanceBatch):
        self.I = inst
        B, n = inst.B, inst.n
        self.B, self.n = B, n
        self.wt = inst.weights
        self.feat_dim = 8 + 5 * n

        self.xmin = inst.w / 2.0
        self.xmax = inst.gw[:, None] - inst.w / 2.0
        self.ymin = inst.h / 2.0
        self.ymax = inst.gh[:, None] - inst.h / 2.0
        # degenerate instances (a room wider than the boundary) must never occur
        assert np.all(self.xmax >= self.xmin - 1e-9), "room wider than boundary"
        assert np.all(self.ymax >= self.ymin - 1e-9), "room taller than boundary"

        iu = np.triu_indices(n, k=1)
        self.iu0, self.iu1 = iu
        self.P = len(self.iu0)

        # per-pair d_max: room i pinned lower-left, room j pinned upper-right
        wi = inst.w[:, self.iu0]
        hi = inst.h[:, self.iu0]
        wj = inst.w[:, self.iu1]
        hj = inst.h[:, self.iu1]
        self.pw_i, self.pw_j = wi, wj
        self.ph_i, self.ph_j = hi, hj
        self.dmax = np.maximum(_pair_distance(
            self.xmin[:, self.iu0], self.ymin[:, self.iu0], wi, hi,
            self.xmax[:, self.iu1], self.ymax[:, self.iu1], wj, hj), 1e-12)
        self.adj_pairs = inst.adj[:, self.iu0, self.iu1]

        # theoretical ceilings, for reporting only
        self.adj_ceiling = self.wt.adj_weight * np.abs(self.adj_pairs).sum(axis=1)
        self.edge_ceiling = (inst.edge_side + np.maximum(inst.edge_bottom, inst.edge_top)).sum(axis=1)

    # -- geometry ----------------------------------------------------------
    def clip(self, x, y):
        return np.clip(x, self.xmin, self.xmax), np.clip(y, self.ymin, self.ymax)

    def overlap_pairs(self, x, y):
        """(B, P) overlap area for each unordered room pair."""
        hw = self.I.w / 2.0
        hh = self.I.h / 2.0
        x0, x1 = x - hw, x + hw
        y0, y1 = y - hh, y + hh
        ox = (np.minimum(x1[:, self.iu0], x1[:, self.iu1])
              - np.maximum(x0[:, self.iu0], x0[:, self.iu1]))
        oy = (np.minimum(y1[:, self.iu0], y1[:, self.iu1])
              - np.maximum(y0[:, self.iu0], y0[:, self.iu1]))
        return np.maximum(ox, 0.0) * np.maximum(oy, 0.0)

    def overlap_matrix(self, x, y):
        """(B, n, n) overlap areas, zero diagonal."""
        hw = self.I.w / 2.0
        hh = self.I.h / 2.0
        x0, x1 = x - hw, x + hw
        y0, y1 = y - hh, y + hh
        ox = np.minimum(x1[:, :, None], x1[:, None, :]) - np.maximum(x0[:, :, None], x0[:, None, :])
        oy = np.minimum(y1[:, :, None], y1[:, None, :]) - np.maximum(y0[:, :, None], y0[:, None, :])
        m = np.maximum(ox, 0.0) * np.maximum(oy, 0.0)
        idx = np.arange(self.n)
        m[:, idx, idx] = 0.0
        return m

    # -- objective ---------------------------------------------------------
    def evaluate(self, x, y, components=False, soft_overlap=0.0):
        """Total objective (B,).  `soft_overlap` in (0, 1] blends in a smoothed
        feasibility signal for *training only*; 0.0 is the exact objective."""
        wt = self.wt
        ov = self.overlap_pairs(x, y)
        overlap_area = ov.sum(axis=1)

        penalty = wt.overlap_weight * overlap_area
        thr = wt.overlap_bonus_threshold
        bonus = np.where(
            np.abs(penalty) <= 1e-12,
            wt.no_overlap_bonus,
            np.where((penalty >= thr) & (penalty < 0.0),
                     wt.no_overlap_middle_bonus * (1.0 - penalty / thr) ** 2, 0.0),
        )
        overlap_term = penalty + bonus

        d = _pair_distance(x[:, self.iu0], y[:, self.iu0], self.pw_i, self.ph_i,
                           x[:, self.iu1], y[:, self.iu1], self.pw_j, self.ph_j)
        nd = np.clip(d / self.dmax, 0.0, 1.0) ** wt.distance_power
        a = self.adj_pairs
        contrib = np.where(a >= 0, a * nd, (-a) * (1.0 - nd))
        contrib = np.where(ov > 0.0, 0.0, contrib)
        adj_term = wt.adj_weight * contrib.sum(axis=1)

        ovm = self.overlap_matrix(x, y)
        clean = ovm.sum(axis=2) <= 0.0                      # (B, n)
        tol = 1e-9
        touch_l = np.abs(x - self.I.w / 2.0) < tol
        touch_r = np.abs((x + self.I.w / 2.0) - self.I.gw[:, None]) < tol
        touch_b = np.abs(y - self.I.h / 2.0) < tol
        touch_t = np.abs((y + self.I.h / 2.0) - self.I.gh[:, None]) < tol
        bonus_room = ((touch_l | touch_r) * self.I.edge_side
                      + touch_b * self.I.edge_bottom
                      + touch_t * self.I.edge_top)
        edge_term = (bonus_room * clean).sum(axis=1)

        total = adj_term + edge_term + overlap_term

        if soft_overlap > 0.0:
            # Training-only smoothing.  The exact objective zeroes the adjacency
            # contribution of any overlapping pair and zeroes a room's wall bonus
            # the moment it touches anything -- two cliffs that give a policy
            # gradient nothing to follow.  The soft version fades those terms out
            # continuously instead.  Never used at evaluation time.
            frac = np.clip(ov / np.maximum(self.pw_i * self.ph_i, 1e-9), 0.0, 1.0)
            soft_contrib = np.where(a >= 0, a * nd, (-a) * (1.0 - nd)) * (1.0 - frac)
            soft_adj = wt.adj_weight * soft_contrib.sum(axis=1)
            room_ov = ovm.sum(axis=2)
            soft_clean = 1.0 / (1.0 + room_ov)
            soft_edge = (bonus_room * soft_clean).sum(axis=1)
            soft_total = soft_adj + soft_edge + overlap_term
            total = (1.0 - soft_overlap) * total + soft_overlap * soft_total

        if components:
            return total, {"adj": adj_term, "edge": edge_term,
                           "overlap": overlap_term, "overlap_area": overlap_area}
        return total

    def evaluate_masked(self, x, y, mask, components=False):
        """Objective of a *partial* layout: only rooms with mask=True exist.

        Needed by the constructive formulation, where the reward for placing a
        room is the objective it adds.  With mask all-True this is exactly
        `evaluate`, which `verify.py` checks.
        """
        wt = self.wt
        mp = mask[:, self.iu0] & mask[:, self.iu1]
        ov = self.overlap_pairs(x, y) * mp
        overlap_area = ov.sum(axis=1)

        penalty = wt.overlap_weight * overlap_area
        thr = wt.overlap_bonus_threshold
        bonus = np.where(
            np.abs(penalty) <= 1e-12, wt.no_overlap_bonus,
            np.where((penalty >= thr) & (penalty < 0.0),
                     wt.no_overlap_middle_bonus * (1.0 - penalty / thr) ** 2, 0.0))
        overlap_term = penalty + bonus

        d = _pair_distance(x[:, self.iu0], y[:, self.iu0], self.pw_i, self.ph_i,
                           x[:, self.iu1], y[:, self.iu1], self.pw_j, self.ph_j)
        nd = np.clip(d / self.dmax, 0.0, 1.0) ** wt.distance_power
        a = self.adj_pairs
        contrib = np.where(a >= 0, a * nd, (-a) * (1.0 - nd))
        contrib = np.where((ov > 0.0) | (~mp), 0.0, contrib)
        adj_term = wt.adj_weight * contrib.sum(axis=1)

        ovm = self.overlap_matrix(x, y) * (mask[:, None, :] & mask[:, :, None])
        clean = (ovm.sum(axis=2) <= 0.0) & mask
        tol = 1e-9
        touch_l = np.abs(x - self.I.w / 2.0) < tol
        touch_r = np.abs((x + self.I.w / 2.0) - self.I.gw[:, None]) < tol
        touch_b = np.abs(y - self.I.h / 2.0) < tol
        touch_t = np.abs((y + self.I.h / 2.0) - self.I.gh[:, None]) < tol
        bonus_room = ((touch_l | touch_r) * self.I.edge_side
                      + touch_b * self.I.edge_bottom + touch_t * self.I.edge_top)
        edge_term = (bonus_room * clean).sum(axis=1)

        total = adj_term + edge_term + overlap_term
        if components:
            return total, {"adj": adj_term, "edge": edge_term,
                           "overlap": overlap_term, "overlap_area": overlap_area}
        return total

    # -- observation -------------------------------------------------------
    def observation(self, x, y):
        """(B, n, 8 + 5n) per-room features.  Same layout as v1 plus affinity."""
        B, n = x.shape
        gx = self.I.gw[:, None]
        gy = self.I.gh[:, None]

        geom = np.stack([x / gx, y / gy, self.I.w / gx, self.I.h / gy], axis=-1)
        bnd = np.stack([(x - self.I.w / 2.0) / gx,
                        (gx - (x + self.I.w / 2.0)) / gx,
                        (y - self.I.h / 2.0) / gy,
                        (gy - (y + self.I.h / 2.0)) / gy], axis=-1)

        dx = x[:, None, :] - x[:, :, None]
        dy = y[:, None, :] - y[:, :, None]
        dist = np.sqrt(dx * dx + dy * dy)
        denom = np.sqrt(self.I.gw ** 2 + self.I.gh ** 2)[:, None, None]
        ovm = self.overlap_matrix(x, y)
        area = (self.I.w * self.I.h).mean(axis=1)[:, None, None]
        norm = np.maximum(dist, 1e-8)
        amax = np.maximum(np.abs(self.I.adj).max(axis=(1, 2)), 1.0)[:, None, None]
        rel = np.stack([dist / denom, ovm / area, dx / norm, dy / norm,
                        self.I.adj / amax], axis=-1).reshape(B, n, 5 * n)
        return np.concatenate([geom, bnd, rel], axis=-1).astype(np.float32)


# ---------------------------------------------------------------------------
# procedural instance generator (Track D)
# ---------------------------------------------------------------------------

# Wall-preference archetypes seen in both reference scenarios, as
# (c_side, c_bottom, c_top) coefficients multiplying EDGE_WEIGHT * (h or w).
EDGE_ARCHETYPES = np.array([
    [0.0,  0.0,  0.0],    # indifferent (phone booths)
    [1.0,  1.0,  1.0],    # wants any wall (utility / storage / server)
    [0.8,  0.8,  0.8],    # wants any wall, weaker (clean / soiled / med)
    [0.0,  0.0,  1.0],    # wants the front  (reception / waiting / triage)
    [0.0,  0.0,  0.8],    # wants the front, weaker (executive)
    [0.3,  0.3,  0.3],    # mild preference (open office / meeting / lounge)
    [0.25, 0.0,  0.0],    # mild side preference (consult / exam)
])


def generate_instances(n_inst, n_rooms, rng, n_types=(6, 12), fill=(0.45, 0.68),
                       aspect=(1.0, 2.0), size_range=(2, 6), affinity=8.0,
                       weights: Weights = None, max_grid=None, jitter_weights=False):
    """Sample `n_inst` office-layout briefs with `n_rooms` rooms each.

    Randomized: the type inventory, how many rooms of each type, per-type room
    dimensions, the type-level affinity table, the boundary aspect ratio, and the
    per-type wall preferences.  The reference scenarios comb_high (36x24, 32
    rooms, 53.6% fill) and hospital (28x16, 32 rooms, 68.1% fill) both sit
    comfortably inside this family, so they can be used as held-out test
    instances that were never seen during training.
    """
    wt = weights or Weights()
    if jitter_weights:
        # the two reference scenarios differ in these constants (comb_high uses
        # -8/400/-200, hospital -10/450/-250), so a policy meant to transfer to
        # both must have seen the range rather than one point of it
        wt = Weights(
            adj_weight=wt.adj_weight, edge_weight=wt.edge_weight,
            overlap_weight=float(rng.uniform(-12.0, -6.0)),
            no_overlap_bonus=float(rng.uniform(320.0, 520.0)),
            no_overlap_middle_bonus=wt.no_overlap_middle_bonus,
            overlap_bonus_threshold=float(rng.uniform(-280.0, -160.0)),
            distance_power=wt.distance_power, step_size=wt.step_size)
    W = np.zeros((n_inst, n_rooms))
    H = np.zeros((n_inst, n_rooms))
    ADJ = np.zeros((n_inst, n_rooms, n_rooms))
    GW = np.zeros(n_inst)
    GH = np.zeros(n_inst)
    ES = np.zeros((n_inst, n_rooms))
    EB = np.zeros((n_inst, n_rooms))
    ET = np.zeros((n_inst, n_rooms))
    TID = np.zeros((n_inst, n_rooms), dtype=np.int64)

    for b in range(n_inst):
        T = rng.integers(n_types[0], n_types[1] + 1)

        # per-type geometry
        tw = rng.integers(size_range[0], size_range[1] + 1, size=T).astype(float)
        th = np.clip(np.rint(tw * rng.uniform(1.0 / aspect[1], aspect[1], size=T)),
                     size_range[0], size_range[1])
        # swap so neither dimension is systematically larger
        flip = rng.random(T) < 0.5
        tw2 = np.where(flip, th, tw)
        th2 = np.where(flip, tw, th)

        # type-level affinity, symmetric, integer-valued like the references
        A = rng.integers(-affinity, affinity + 1, size=(T, T)).astype(float)
        A = np.triu(A) + np.triu(A, 1).T
        # references are sparse-ish: ~25% of pairs are neutral
        A *= (rng.random((T, T)) > 0.25)
        A = np.triu(A) + np.triu(A, 1).T
        np.fill_diagonal(A, np.diag(A))

        arch = EDGE_ARCHETYPES[rng.integers(0, len(EDGE_ARCHETYPES), size=T)]

        # assign rooms to types: every type used at least once, rest multinomial
        assign = np.concatenate([np.arange(T),
                                 rng.integers(0, T, size=max(0, n_rooms - T))])[:n_rooms]
        rng.shuffle(assign)

        W[b] = tw2[assign]
        H[b] = th2[assign]
        TID[b] = assign
        ADJ[b] = A[assign][:, assign]
        np.fill_diagonal(ADJ[b], 0.0)
        ES[b] = arch[assign, 0] * wt.edge_weight * H[b]
        EB[b] = arch[assign, 1] * wt.edge_weight * W[b]
        ET[b] = arch[assign, 2] * wt.edge_weight * W[b]

        # boundary sized to hit a target fill ratio at a random aspect ratio
        area = (W[b] * H[b]).sum()
        target = area / rng.uniform(*fill)
        ar = rng.uniform(1.0, aspect[1])
        gw = math.ceil(math.sqrt(target * ar))
        gh = math.ceil(target / gw)
        # every room must fit inside the boundary
        if max_grid is not None:
            gw = min(gw, max_grid)
            gh = min(gh, max_grid)
        gw = max(gw, int(W[b].max()))
        gh = max(gh, int(H[b].max()))
        GW[b], GH[b] = float(gw), float(gh)

    return InstanceBatch(w=W, h=H, adj=ADJ, gw=GW, gh=GH, edge_side=ES,
                         edge_bottom=EB, edge_top=ET, type_id=TID, weights=wt)


def random_layout(obj: BatchObjective, rng, grid_aligned=False):
    """Uniform random room centres inside the boundary."""
    x = rng.uniform(obj.xmin, obj.xmax)
    y = rng.uniform(obj.ymin, obj.ymax)
    if grid_aligned:
        x = np.clip(np.rint(x - obj.xmin) + obj.xmin, obj.xmin, obj.xmax)
        y = np.clip(np.rint(y - obj.ymin) + obj.ymin, obj.ymin, obj.ymax)
    return x, y
