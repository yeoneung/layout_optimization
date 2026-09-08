"""
Strengthened classical baselines, all on the v2 per-instance objective.

The point of this file is fairness in the direction that hurts: before claiming
anything for RL, the classical side must be given every advantage it can have.
Beyond the paper's SA this adds

  * `sa_grid`   - SA restricted to the integer lattice.  The paper's SA moves in
                  continuous coordinates by whole cells, so each room keeps the
                  fractional offset it was handed at initialization and two rooms
                  with different offsets can never be placed flush.  That is a
                  handicap, not a property of annealing.  Removing it makes SA
                  strictly stronger and is the honest comparator for any method
                  that places rooms on the lattice.
  * `tabu`      - steepest ascent with a tabu list, the standard FLP local search.
  * `ga`        - a population method, the other standard FLP metaheuristic.
  * `sa_restarts` - multi-start SA, so the budget can be spent on diversification
                  instead of one long chain.

Every solver reports the number of objective evaluations it consumed and its
wall-clock time, and takes an explicit `budget` in evaluations, so methods can be
compared on the x-axis that actually matters rather than on "steps".
"""

import time

import numpy as np

import core

DIRS = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])


def expand(inst, k):
    """BatchObjective over `k` copies of every instance (for populations)."""
    return core.BatchObjective(inst.repeat(k))


def _result(x, y, best, evals, t0, trace=None):
    return {"x": x, "y": y, "best": best, "evals": int(evals),
            "wall_time": time.time() - t0, "trace": trace or []}


def snap_to_grid(obj, x, y):
    """Round each room to the nearest lattice position (keeps it in bounds)."""
    return (np.clip(np.rint(x - obj.xmin) + obj.xmin, obj.xmin, obj.xmax),
            np.clip(np.rint(y - obj.ymin) + obj.ymin, obj.ymin, obj.ymax))


# ---------------------------------------------------------------------------
# constructive, no search
# ---------------------------------------------------------------------------

def shelf_pack(obj, rng, order_by="random", n_tries=1):
    """First-fit shelf packing.  `n_tries>1` keeps the best of several orders,
    which is the fair version when it is compared against a sampled policy."""
    I = obj.I
    B, n = I.B, I.n
    bestx = np.zeros((B, n))
    besty = np.zeros((B, n))
    bestv = np.full(B, -np.inf)
    for _ in range(n_tries):
        X = np.zeros((B, n))
        Y = np.zeros((B, n))
        for b in range(B):
            if order_by == "area":
                order = np.argsort(-(I.w[b] * I.h[b]))
            elif order_by == "height":
                order = np.argsort(-I.h[b])
            else:
                order = rng.permutation(n)
            cx = cy = shelf_h = 0.0
            for i in order:
                w, h = I.w[b, i], I.h[b, i]
                if cx + w > I.gw[b] + 1e-9:
                    cx, cy, shelf_h = 0.0, cy + shelf_h, 0.0
                if cy + h > I.gh[b] + 1e-9:
                    X[b, i] = rng.uniform(obj.xmin[b, i], obj.xmax[b, i])
                    Y[b, i] = rng.uniform(obj.ymin[b, i], obj.ymax[b, i])
                    continue
                X[b, i] = cx + w / 2.0
                Y[b, i] = cy + h / 2.0
                cx += w
                shelf_h = max(shelf_h, h)
        v = obj.evaluate(X, Y)
        imp = v > bestv
        bestv = np.where(imp, v, bestv)
        bestx[imp] = X[imp]
        besty[imp] = Y[imp]
    return bestx, besty


# ---------------------------------------------------------------------------
# local search
# ---------------------------------------------------------------------------

def greedy_polish(obj, x, y, max_passes=40, step=1.0, budget=None):
    """Steepest ascent over single-room one-cell moves."""
    B, n = x.shape
    x, y = x.copy(), y.copy()
    cur = obj.evaluate(x, y)
    evals = B
    bidx = np.arange(B)
    for _ in range(max_passes):
        best_score = cur.copy()
        best_room = np.full(B, -1)
        best_dir = np.zeros(B, dtype=np.int64)
        for i in range(n):
            for d in range(4):
                cx, cy = x.copy(), y.copy()
                cx[:, i] = np.clip(cx[:, i] + DIRS[d, 0] * step, obj.xmin[:, i], obj.xmax[:, i])
                cy[:, i] = np.clip(cy[:, i] + DIRS[d, 1] * step, obj.ymin[:, i], obj.ymax[:, i])
                s = obj.evaluate(cx, cy)
                evals += B
                better = s > best_score + 1e-12
                best_score = np.where(better, s, best_score)
                best_room = np.where(better, i, best_room)
                best_dir = np.where(better, d, best_dir)
        act = best_room >= 0
        if not np.any(act):
            break
        idx = bidx[act]
        r = best_room[act]
        dd = best_dir[act]
        x[idx, r] = np.clip(x[idx, r] + DIRS[dd, 0] * step, obj.xmin[idx, r], obj.xmax[idx, r])
        y[idx, r] = np.clip(y[idx, r] + DIRS[dd, 1] * step, obj.ymin[idx, r], obj.ymax[idx, r])
        cur = np.where(act, best_score, cur)
        if budget is not None and evals >= budget:
            break
    return x, y, cur, evals


def tabu(obj, x, y, budget, step=1.0, tenure=12, rng=None, trace_every=0):
    """Steepest ascent that must move every iteration, with a tabu list on
    (room, direction) so it cannot immediately undo its last moves."""
    t0 = time.time()
    B, n = x.shape
    x, y = x.copy(), y.copy()
    cur = obj.evaluate(x, y)
    best = cur.copy()
    bx, by = x.copy(), y.copy()
    evals = B
    tab = np.zeros((B, n, 4), dtype=np.int32)
    bidx = np.arange(B)
    trace = []
    it = 0
    while evals + B * n * 4 <= budget:
        it += 1
        scores = np.full((B, n, 4), -np.inf)
        for i in range(n):
            for d in range(4):
                cx, cy = x.copy(), y.copy()
                cx[:, i] = np.clip(cx[:, i] + DIRS[d, 0] * step, obj.xmin[:, i], obj.xmax[:, i])
                cy[:, i] = np.clip(cy[:, i] + DIRS[d, 1] * step, obj.ymin[:, i], obj.ymax[:, i])
                scores[:, i, d] = obj.evaluate(cx, cy)
                evals += B
        # aspiration: a tabu move is allowed if it beats the incumbent
        blocked = (tab > it) & (scores <= best[:, None, None])
        masked = np.where(blocked, -np.inf, scores)
        flat = masked.reshape(B, -1).argmax(axis=1)
        r, d = flat // 4, flat % 4
        x[bidx, r] = np.clip(x[bidx, r] + DIRS[d, 0] * step, obj.xmin[bidx, r], obj.xmax[bidx, r])
        y[bidx, r] = np.clip(y[bidx, r] + DIRS[d, 1] * step, obj.ymin[bidx, r], obj.ymax[bidx, r])
        cur = scores[bidx, r, d]
        tab[bidx, r, d ^ 1] = it + tenure          # forbid undoing this move (0<->1, 2<->3)
        imp = cur > best
        best = np.where(imp, cur, best)
        bx[imp], by[imp] = x[imp], y[imp]
        if trace_every and it % trace_every == 0:
            trace.append({"evals_per_inst": evals / B, "best_mean": float(best.mean())})
    return _result(bx, by, best, evals, t0, trace)


# ---------------------------------------------------------------------------
# simulated annealing
# ---------------------------------------------------------------------------

def simulated_annealing(obj, x0, y0, n_steps, t0=30.0, t1=0.08, greedy_every=5000,
                        seed=0, grid=False, final_polish=True, trace_every=0,
                        jump_prob=0.20, step=1.0):
    """The paper's SA, vectorized.  `grid=True` keeps every room on the integer
    lattice (init is snapped and jumps land on lattice points)."""
    wall0 = time.time()
    rng = np.random.default_rng(seed)
    B, n = x0.shape
    x, y = x0.copy(), y0.copy()
    if grid:
        x, y = snap_to_grid(obj, x, y)
    cur = obj.evaluate(x, y)
    evals = B
    best = cur.copy()
    bx, by = x.copy(), y.copy()
    b = np.arange(B)
    trace = []

    for s in range(1, n_steps + 1):
        frac = (s - 1) / max(1, n_steps - 1)
        temp = t0 * ((t1 / t0) ** frac)

        room = rng.integers(0, n, size=B)
        cx, cy = x.copy(), y.copy()
        big = rng.random(B) >= (1.0 - jump_prob)
        d = rng.integers(0, 4, size=B)
        lo_x, hi_x = obj.xmin[b, room], obj.xmax[b, room]
        lo_y, hi_y = obj.ymin[b, room], obj.ymax[b, room]
        px = x[b, room] + DIRS[d, 0] * step
        py = y[b, room] + DIRS[d, 1] * step
        if grid:
            jx = lo_x + rng.integers(0, np.maximum(np.rint(hi_x - lo_x), 0) + 1)
            jy = lo_y + rng.integers(0, np.maximum(np.rint(hi_y - lo_y), 0) + 1)
        else:
            jx = rng.uniform(0, 1, B) * (hi_x - lo_x) + lo_x
            jy = rng.uniform(0, 1, B) * (hi_y - lo_y) + lo_y
        cx[b, room] = np.clip(np.where(big, jx, px), lo_x, hi_x)
        cy[b, room] = np.clip(np.where(big, jy, py), lo_y, hi_y)

        cand = obj.evaluate(cx, cy)
        evals += B
        delta = cand - cur
        accept = (delta >= 0) | (rng.random(B) < np.exp(np.clip(delta / max(temp, 1e-12), -50, 0)))
        x = np.where(accept[:, None], cx, x)
        y = np.where(accept[:, None], cy, y)
        cur = np.where(accept, cand, cur)

        imp = cur > best
        best = np.where(imp, cur, best)
        bx[imp], by[imp] = x[imp], y[imp]

        if trace_every > 0 and (s % trace_every == 0 or s == 1):
            trace.append({"evals_per_inst": evals / B, "best_mean": float(best.mean()),
                          "wall": time.time() - wall0})

        if greedy_every > 0 and s % greedy_every == 0:
            px_, py_, ps, ev = greedy_polish(obj, bx.copy(), by.copy(), 20, step)
            evals += ev
            imp = ps > best
            best = np.where(imp, ps, best)
            bx[imp], by[imp] = px_[imp], py_[imp]

    if final_polish:
        bx, by, best, ev = greedy_polish(obj, bx, by, 40, step)
        evals += ev
    return _result(bx, by, best, evals, wall0, trace)


def sa_restarts(obj_single, inst, x0, y0, n_restarts, n_steps, seed=0, grid=False,
                **kw):
    """`n_restarts` independent SA chains per instance; best chain wins.
    Same total budget can therefore be spent on diversification instead of depth."""
    t0 = time.time()
    objP = expand(inst, n_restarts)
    rng = np.random.default_rng(seed)
    X = np.repeat(x0, n_restarts, axis=0)
    Y = np.repeat(y0, n_restarts, axis=0)
    # restarts after the first get fresh random starts
    rx, ry = core.random_layout(objP, rng, grid_aligned=grid)
    keep = (np.arange(X.shape[0]) % n_restarts) == 0
    X = np.where(keep[:, None], X, rx)
    Y = np.where(keep[:, None], Y, ry)
    res = simulated_annealing(objP, X, Y, n_steps, seed=seed, grid=grid, **kw)
    B = inst.B
    bs = res["best"].reshape(B, n_restarts)
    pick = bs.argmax(axis=1)
    flat = np.arange(B) * n_restarts + pick
    return _result(res["x"][flat], res["y"][flat], bs.max(axis=1), res["evals"], t0)


# ---------------------------------------------------------------------------
# genetic algorithm
# ---------------------------------------------------------------------------

def genetic(obj_single, inst, x0, y0, pop, generations, seed=0, grid=False,
            elite_frac=0.2, mut_rate=0.15, jump_rate=0.05, step=1.0,
            local_search_every=0, trace_every=0):
    """Population GA with uniform per-room crossover (rooms are the genes),
    Gaussian/lattice mutation and occasional teleport, plus optional memetic
    local search -- i.e. the standard strong FLP genetic algorithm."""
    t0 = time.time()
    B, n = x0.shape
    objP = expand(inst, pop)
    rng = np.random.default_rng(seed)
    X, Y = core.random_layout(objP, rng, grid_aligned=grid)
    keep = (np.arange(B * pop) % pop) == 0        # seed one individual with x0
    X = np.where(keep[:, None], np.repeat(x0, pop, axis=0), X)
    Y = np.where(keep[:, None], np.repeat(y0, pop, axis=0), Y)
    if grid:
        X, Y = snap_to_grid(objP, X, Y)

    fit = objP.evaluate(X, Y)
    evals = B * pop
    n_elite = max(1, int(pop * elite_frac))
    objE = expand(inst, n_elite) if local_search_every else None
    trace = []
    best_x = np.zeros((B, n))
    best_y = np.zeros((B, n))
    best_v = np.full(B, -np.inf)

    def record():
        f = fit.reshape(B, pop)
        pick = f.argmax(axis=1)
        flat = np.arange(B) * pop + pick
        imp = f.max(axis=1) > best_v
        best_v[imp] = f.max(axis=1)[imp]
        best_x[imp] = X[flat][imp]
        best_y[imp] = Y[flat][imp]

    record()
    for g in range(generations):
        f = fit.reshape(B, pop)
        order = np.argsort(-f, axis=1)                       # (B, pop)
        elite = order[:, :n_elite]
        # tournament selection of two parents per offspring
        cand = rng.integers(0, pop, size=(B, pop, 2, 2))
        p1 = cand[..., 0, 0]
        p2 = cand[..., 0, 1]
        q1 = cand[..., 1, 0]
        q2 = cand[..., 1, 1]
        pa = np.where(np.take_along_axis(f, p1, 1) >= np.take_along_axis(f, p2, 1), p1, p2)
        pb = np.where(np.take_along_axis(f, q1, 1) >= np.take_along_axis(f, q2, 1), q1, q2)

        gi = np.arange(B)[:, None] * pop
        XA, YA = X[(gi + pa).ravel()], Y[(gi + pa).ravel()]
        XB, YB = X[(gi + pb).ravel()], Y[(gi + pb).ravel()]
        take_a = rng.random((B * pop, n)) < 0.5              # uniform crossover over rooms
        CX = np.where(take_a, XA, XB)
        CY = np.where(take_a, YA, YB)

        mut = rng.random((B * pop, n)) < mut_rate
        if grid:
            CX = CX + mut * rng.integers(-2, 3, size=CX.shape)
            CY = CY + mut * rng.integers(-2, 3, size=CY.shape)
        else:
            CX = CX + mut * rng.normal(0, 1.5, CX.shape)
            CY = CY + mut * rng.normal(0, 1.5, CY.shape)
        jmp = rng.random((B * pop, n)) < jump_rate
        jx, jy = core.random_layout(objP, rng, grid_aligned=grid)
        CX = np.where(jmp, jx, CX)
        CY = np.where(jmp, jy, CY)
        CX, CY = objP.clip(CX, CY)
        if grid:
            CX, CY = snap_to_grid(objP, CX, CY)

        # elitism: overwrite the first n_elite offspring of each instance
        el = (gi + elite).ravel()
        pos = (np.arange(B)[:, None] * pop + np.arange(n_elite)[None, :]).ravel()
        CX[pos] = X[el]
        CY[pos] = Y[el]

        X, Y = CX, CY
        fit = objP.evaluate(X, Y)
        evals += B * pop
        if local_search_every and (g + 1) % local_search_every == 0:
            # memetic step: hill-climb the elites only, as is standard -- polishing
            # the whole population costs pop/n_elite times as much for no gain
            top = np.argsort(-fit.reshape(B, pop), axis=1)[:, :n_elite]
            sel = (np.arange(B)[:, None] * pop + top).ravel()
            ex, ey, ef, ev = greedy_polish(objE, X[sel], Y[sel], 4, step)
            X[sel], Y[sel] = ex, ey
            fit[sel] = ef
            evals += ev
        record()
        if trace_every and (g + 1) % trace_every == 0:
            trace.append({"evals_per_inst": evals / B, "best_mean": float(best_v.mean()),
                          "wall": time.time() - t0})
    return _result(best_x, best_y, best_v, evals, t0, trace)
