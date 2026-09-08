"""
A held-out benchmark suite of layout instances.

The case studies in this paper are two hand-authored briefs.  Repeating a solver
64 times on each measures solver variability, not instance variability, so no
statement about how a method behaves *across problems* can rest on them.  This
module builds the missing axis: a frozen suite of generated instances spanning
facility count and packing density, on which every method is run once per
instance so that differences can be paired and tested.

Feasibility by construction.  Sampling room sizes independently produces
instances that are unpackable at high density, and a suite containing impossible
instances measures nothing.  Every instance here is therefore built from a
guillotine partition of the plate: the plate is recursively sliced into `n`
disjoint rectangles, and each is then shrunk inside its own slot to hit the
target fill ratio.  The shrunken rectangles still fit in disjoint slots, so a
feasible packing provably exists.  The witness coordinates are recorded for
sanity checking and withheld from every solver, and shrinking means the witness
is generally *not* optimal, so it does not leak the answer.

Types and preferences are drawn independently of geometry: a room's type carries
its affinity row and its wall-preference archetype, while its size comes from the
partition.  This is deliberately more general than `core.generate_instances`,
where type determines size as it does in the two legacy hand-authored scenarios.

Seed discipline.  `split` selects disjoint seed ranges for train, validation and
test, so a policy trained on one can never have optimized an instance from
another.  Instance identity is (split, n, fill, index) and is reproducible from
the seed alone.
"""

import math

import numpy as np

import core
from core import EDGE_ARCHETYPES, InstanceBatch, Weights

SPLIT_OFFSET = {"train": 0, "val": 4_000_000, "test": 8_000_000}


# ---------------------------------------------------------------------------
def guillotine(n, GW, GH, rng, min_size=2):
    """Slice the GW x GH plate into n disjoint integer rectangles that tile it."""
    rects = [(0, 0, GW, GH)]
    guard = 0
    while len(rects) < n and guard < 20 * n:
        guard += 1
        area = np.array([r[2] * r[3] for r in rects], dtype=float)
        splittable = np.array([(r[2] >= 2 * min_size) or (r[3] >= 2 * min_size)
                               for r in rects])
        if not splittable.any():
            break
        p = area * splittable
        k = int(rng.choice(len(rects), p=p / p.sum()))
        x, y, w, h = rects.pop(k)
        # cut the longer side when it is splittable, else the other one
        vertical = w >= h
        if vertical and w < 2 * min_size:
            vertical = False
        if (not vertical) and h < 2 * min_size:
            vertical = True
        if vertical:
            cut = int(rng.integers(min_size, w - min_size + 1))
            rects += [(x, y, cut, h), (x + cut, y, w - cut, h)]
        else:
            cut = int(rng.integers(min_size, h - min_size + 1))
            rects += [(x, y, w, cut), (x, y + cut, w, h - cut)]
    return rects


def _has_full_cut(rects, GW, GH):
    """Whether a tiled rectangle set has an immediate guillotine cut."""
    for cut in range(1, GW):
        if not any(x < cut < x + w for x, y, w, h in rects):
            return True
    for cut in range(1, GH):
        if not any(y < cut < y + h for x, y, w, h in rects):
            return True
    return False


def nonslicing(n, GW, GH, rng, min_size=2):
    """Build a tiled family around a five-rectangle nonslicing pinwheel.

    The five seed rectangles form the standard pinwheel pattern: four bent
    around a central rectangle, with no horizontal or vertical cut spanning the
    plate.  Further rooms are obtained by local splits that are accepted only
    while the complete tiling still has no first guillotine cut.  This produces
    a geometry family whose source packing is structurally different from the
    recursive slicing tree used by :func:`guillotine`.
    """
    if n < 5 or GW < 3 * min_size or GH < 3 * min_size:
        return []
    a = int(rng.integers(min_size, GW - 2 * min_size + 1))
    b = int(rng.integers(a + min_size, GW - min_size + 1))
    c = int(rng.integers(min_size, GH - 2 * min_size + 1))
    d = int(rng.integers(c + min_size, GH - min_size + 1))
    rects = [
        (0, d, b, GH - d),
        (b, c, GW - b, GH - c),
        (a, 0, GW - a, c),
        (0, 0, a, d),
        (a, c, b - a, d - c),
    ]
    if _has_full_cut(rects, GW, GH):
        raise RuntimeError("pinwheel seed unexpectedly has a guillotine cut")

    guard = 0
    while len(rects) < n and guard < 100 * n:
        guard += 1
        candidates = [k for k, (_, _, w, h) in enumerate(rects)
                      if w >= 2 * min_size or h >= 2 * min_size]
        if not candidates:
            break
        areas = np.array([rects[k][2] * rects[k][3] for k in candidates],
                         dtype=float)
        k = int(rng.choice(candidates, p=areas / areas.sum()))
        x, y, w, h = rects[k]
        directions = []
        if w >= 2 * min_size:
            directions.append("vertical")
        if h >= 2 * min_size:
            directions.append("horizontal")
        rng.shuffle(directions)
        accepted = False
        for direction in directions:
            if direction == "vertical":
                cut = int(rng.integers(min_size, w - min_size + 1))
                children = [(x, y, cut, h), (x + cut, y, w - cut, h)]
            else:
                cut = int(rng.integers(min_size, h - min_size + 1))
                children = [(x, y, w, cut), (x, y + cut, w, h - cut)]
            trial = rects[:k] + children + rects[k + 1:]
            if not _has_full_cut(trial, GW, GH):
                rects = trial
                accepted = True
                break
        if not accepted:
            continue
    return rects


def _shrink_to_fill(rects, GW, GH, fill, rng, min_size=2):
    """Shrink each rectangle inside its own slot until the fill ratio is hit.

    Bisection on a common scale factor overshoots downward, because flooring each
    dimension of a small rectangle discards a large fraction of it.  A greedy
    pass then grows individual rooms back, one cell at a time and never past
    their own slot, until the target is met from below.  Feasibility is preserved
    at every step: a room that fits in its slot cannot overlap another slot.
    """
    slots = np.array([[r[2], r[3]] for r in rects], dtype=np.int64)
    plate = GW * GH
    lo, hi = 0.05, 1.0
    w = h = None
    for _ in range(40):
        s = 0.5 * (lo + hi)
        cw = np.clip(np.floor(slots[:, 0] * s), min_size, slots[:, 0]).astype(np.int64)
        ch = np.clip(np.floor(slots[:, 1] * s), min_size, slots[:, 1]).astype(np.int64)
        f = (cw * ch).sum() / plate
        if f > fill:
            hi = s
        else:
            lo, w, h = s, cw, ch
    if w is None:                            # even the smallest sizes overshoot
        w = np.minimum(min_size, slots[:, 0])
        h = np.minimum(min_size, slots[:, 1])

    n = len(slots)
    for _ in range(40 * n):
        deficit = fill * plate - float((w * h).sum())
        if deficit <= 0:
            break
        # a growth step costs the room's other dimension in cells
        cand = [(i, 0, h[i]) for i in range(n) if w[i] < slots[i, 0]] + \
               [(i, 1, w[i]) for i in range(n) if h[i] < slots[i, 1]]
        cand = [c for c in cand if c[2] <= deficit]
        if not cand:
            break
        i, dim, _ = cand[int(rng.integers(len(cand)))]
        if dim == 0:
            w[i] += 1
        else:
            h[i] += 1
    return w, h, float((w * h).sum()) / plate


def make_instances(n_inst, n_rooms, fill, rng, max_grid=44, min_size=2,
                   n_types=(6, 12), affinity=8.0, aspect=(1.0, 1.8),
                   weights=None, jitter_weights=True, avg_area=(9.0, 30.0),
                   geometry="guillotine"):
    """Feasible instances with ``n_rooms`` facilities at about ``fill``.

    ``geometry='guillotine'`` uses the original recursive slicing family.
    ``geometry='nonslicing'`` uses an independently seeded pinwheel family.

    `avg_area` bounds the mean room area and therefore the plate size.  The exact
    CP-SAT model of `exact_cpsat.py` tabulates every position pair, so its cost
    grows as the fourth power of the plate dimension; narrowing this range is how
    the small cells are made small enough for optimality to be *proven* rather
    than merely bounded.
    """
    wt = weights or Weights()
    W = np.zeros((n_inst, n_rooms))
    H = np.zeros((n_inst, n_rooms))
    ADJ = np.zeros((n_inst, n_rooms, n_rooms))
    GW = np.zeros(n_inst)
    GH = np.zeros(n_inst)
    ES = np.zeros((n_inst, n_rooms))
    EB = np.zeros((n_inst, n_rooms))
    ET = np.zeros((n_inst, n_rooms))
    TID = np.zeros((n_inst, n_rooms), dtype=np.int64)
    witness = np.zeros((n_inst, n_rooms, 2))
    achieved = np.zeros(n_inst)

    for b in range(n_inst):
        # plate sized so the mean room is neither a sliver nor the whole plate,
        # and never larger than the canvas the policy was trained on
        cap = max_grid * max_grid * fill / n_rooms
        a_avg = min(rng.uniform(*avg_area), 0.95 * cap)
        target = n_rooms * a_avg / fill
        ar = rng.uniform(*aspect)
        gw = min(max_grid, max(2 * min_size, int(round(math.sqrt(target * ar)))))
        gh = min(max_grid, max(2 * min_size, int(round(target / max(gw, 1)))))

        partition = nonslicing if geometry == "nonslicing" else guillotine
        if geometry not in ("guillotine", "nonslicing"):
            raise ValueError("unknown geometry family: {}".format(geometry))
        rects = partition(n_rooms, gw, gh, rng, min_size)
        while len(rects) < n_rooms:          # plate too small to slice that far
            gw = min(max_grid, gw + 1)
            gh = min(max_grid, gh + 1)
            rects = partition(n_rooms, gw, gh, rng, min_size)
            if gw == max_grid and gh == max_grid and len(rects) < n_rooms:
                raise RuntimeError("could not generate requested geometry family")
        rects = rects[:n_rooms]

        w, h, f = _shrink_to_fill(rects, gw, gh, fill, rng, min_size)
        W[b], H[b] = w, h
        GW[b], GH[b] = float(gw), float(gh)
        witness[b, :, 0] = [r[0] for r in rects]
        witness[b, :, 1] = [r[1] for r in rects]
        achieved[b] = f

        # function types, independent of geometry
        T = int(rng.integers(n_types[0], min(n_types[1], n_rooms) + 1))
        assign = np.concatenate([np.arange(T),
                                 rng.integers(0, T, size=max(0, n_rooms - T))])[:n_rooms]
        rng.shuffle(assign)
        A = rng.integers(-affinity, affinity + 1, size=(T, T)).astype(float)
        A = np.triu(A) + np.triu(A, 1).T
        A *= (rng.random((T, T)) > 0.25)
        A = np.triu(A) + np.triu(A, 1).T
        arch = EDGE_ARCHETYPES[rng.integers(0, len(EDGE_ARCHETYPES), size=T)]

        TID[b] = assign
        ADJ[b] = A[assign][:, assign]
        np.fill_diagonal(ADJ[b], 0.0)
        ES[b] = arch[assign, 0] * wt.edge_weight * H[b]
        EB[b] = arch[assign, 1] * wt.edge_weight * W[b]
        ET[b] = arch[assign, 2] * wt.edge_weight * W[b]

    if jitter_weights:
        # the two legacy scenarios differ in these constants, so a suite meant
        # to test generalization must span the range rather than one point
        wt = Weights(
            adj_weight=wt.adj_weight, edge_weight=wt.edge_weight,
            overlap_weight=float(rng.uniform(-12.0, -6.0)),
            no_overlap_bonus=float(rng.uniform(320.0, 520.0)),
            no_overlap_middle_bonus=wt.no_overlap_middle_bonus,
            overlap_bonus_threshold=float(rng.uniform(-280.0, -160.0)),
            distance_power=wt.distance_power, step_size=wt.step_size)

    inst = InstanceBatch(w=W, h=H, adj=ADJ, gw=GW, gh=GH, edge_side=ES,
                         edge_bottom=EB, edge_top=ET, type_id=TID, weights=wt)
    return inst, witness, achieved


# ---------------------------------------------------------------------------
def suite_cell(split, n_rooms, fill, n_inst, base_seed=20260812, tag="", **kw):
    """One (n, fill) cell of the suite, reproducible from its coordinates.

    `tag` shifts the seed so a differently-parameterized family (for instance the
    small cells used for exact solving) cannot collide with the main suite.
    """
    seed = (base_seed + SPLIT_OFFSET[split]
            + n_rooms * 10_000 + int(round(fill * 100))
            + (sum(ord(c) for c in tag) * 1_000_003 if tag else 0))
    rng = np.random.default_rng(seed)
    inst, witness, achieved = make_instances(n_inst, n_rooms, fill, rng, **kw)
    return {"split": split, "n_rooms": n_rooms, "fill": fill, "seed": seed,
            "inst": inst, "witness": witness, "achieved_fill": achieved}


def verify_witness(cell, tol=1e-9):
    """The recorded packing must be overlap-free and inside the plate."""
    inst = cell["inst"]
    obj = core.BatchObjective(inst)
    x = cell["witness"][:, :, 0] + inst.w / 2.0
    y = cell["witness"][:, :, 1] + inst.h / 2.0
    tot, c = obj.evaluate(x, y, components=True)
    inside = bool(np.all(cell["witness"][:, :, 0] + inst.w <= inst.gw[:, None] + tol)
                  and np.all(cell["witness"][:, :, 1] + inst.h <= inst.gh[:, None] + tol))
    return {"max_overlap": float(c["overlap_area"].max()),
            "all_inside": inside,
            "witness_J_mean": float(tot.mean())}
