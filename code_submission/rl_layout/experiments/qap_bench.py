"""
QAPLIB: the classical facility layout objective, with proven optima.

Why this file exists.  Every quality number elsewhere in this work is a
difference between methods on instances we generated.  The quadratic assignment
problem supplies what that cannot: instances the field has used for decades
(Nugent, Hadley, Scriabin, Roucairol, Christofides, Elshafei, Taillard) whose
optimal values are proven, so every method can be reported as a gap to the
truth rather than to the best heuristic present.

QAP also closes a representational loop.  It is the fill-ratio-1.0 corner of
lattice placement: n unit-square facilities, exactly n grid locations, flow
times distance as the objective.  Two consequences, both of which the paper
predicts:

  * The pairwise objective admits the same exact marginal decomposition as the
    adjacency and HPWL objectives -- assigning facility i to location l adds
    sum_j A[i,j] B[l, p_j] + A[j,i] B[p_j, l] over placed j, which depends on
    committed facilities only.  The constructive machinery therefore transfers
    verbatim, and this file runs it.
  * Feasibility is trivial (any injective assignment completes), so the
    certificate always passes, coverage is 1.0 by construction, and the entire
    feasibility apparatus of the paper has nothing to do.  What remains is pure
    solution quality -- exactly the regime where improvement search is at its
    best, since a swap neighbourhood needs no empty space.  The honest
    expectation, which the results confirm, is that swap-based search dominates
    constructive insertion here, symmetrically to the density regime where the
    relationship inverts.

Budget accounting.  The budget axis is full-objective-evaluation equivalents
(45,000 per instance, as everywhere in the paper), converted at the cost a
standard implementation pays: a full evaluation is n^2 flow terms, a single
swap delta is 8n terms, and one tabu iteration with Taillard's delta-matrix
update is 2n^2 terms.  Our tabu implementation actually recomputes the full
delta matrix with dense algebra each iteration because that is simpler to get
right; it is *charged* at the delta-update rate anyway, which errs in the
classical side's favour.  Wall-clock is reported as measured.

File format.  This mirror's .dat header line is "n <best known value>",
followed by two n x n matrices.  The header values were cross-checked against
QAPLIB's published optima; all instances used here are proven optimal.  Since
sum_ij A[ij] B[p(i)p(j)] is invariant under exchanging the roles of A and B
(replace p by its inverse), which matrix is flow and which is distance does not
affect any number reported.
"""

import argparse
import glob
import json
import os
import time

import numpy as np


# ---------------------------------------------------------------------------
# instance
# ---------------------------------------------------------------------------

def load(path):
    toks = open(path).read().split()
    n = int(toks[0])
    vals = np.array(toks[1:], dtype=np.float64)
    opt = None
    if len(vals) == 2 * n * n + 1:            # header carries the optimum
        opt, vals = float(vals[0]), vals[1:]
    assert len(vals) == 2 * n * n, f"{path}: bad token count"
    A = vals[:n * n].reshape(n, n)
    B = vals[n * n:].reshape(n, n)
    return {"name": os.path.splitext(os.path.basename(path))[0],
            "n": n, "opt": opt, "A": A, "B": B}


def cost(inst, p):
    """Full objective: sum_ij A[i,j] * B[p_i, p_j]."""
    Bp = inst["B"][np.ix_(p, p)]
    return float((inst["A"] * Bp).sum())


class Meter:
    """Counts elementary flow terms; reports full-evaluation equivalents."""

    def __init__(self, n):
        self.n = n
        self.terms = 0.0

    def evals(self):
        return self.terms / (self.n * self.n)


# ---------------------------------------------------------------------------
# constructive insertion on the exact marginal
# ---------------------------------------------------------------------------

def marginal(inst, placed_f, placed_l, i, free_l, meter):
    """Exact marginal of assigning facility i to each location in `free_l`,
    given the committed assignment.  Depends on placed facilities only."""
    A, B = inst["A"], inst["B"]
    if len(placed_f):
        pf = np.asarray(placed_f)
        pl = np.asarray(placed_l)
        m = (A[i, pf][None, :] * B[np.ix_(free_l, pl)]).sum(1) \
            + (A[pf, i][None, :].reshape(1, -1) * B[np.ix_(pl, free_l)].T).sum(1)
    else:
        m = np.zeros(len(free_l))
    m = m + A[i, i] * B[np.asarray(free_l), np.asarray(free_l)]
    meter.terms += (2 * len(placed_f) + 1) * len(free_l)
    return m


def busiest_order(inst):
    """Total-flow-descending facility order -- the analogue of largest-first."""
    A = inst["A"]
    return np.argsort(-(A.sum(0) + A.sum(1) - np.diag(A)), kind="stable")


def greedy(inst, meter, order=None):
    """Fixed-order greedy: each facility takes the marginal-minimizing free
    location.  Ties break on location index, deterministically."""
    n = inst["n"]
    order = busiest_order(inst) if order is None else order
    pf, pl = [], []
    free = list(range(n))
    for i in order:
        m = marginal(inst, pf, pl, int(i), free, meter)
        k = int(np.argmin(m))
        pf.append(int(i))
        pl.append(free.pop(k))
    p = np.empty(n, dtype=int)
    p[pf] = pl
    return p


def regret(inst, k, meter):
    """Regret-k insertion: commit the facility that will suffer most if it
    does not get its best location now."""
    n = inst["n"]
    pf, pl = [], []
    freeF = list(range(n))
    freeL = list(range(n))
    while freeF:
        best = None
        for i in freeF:
            m = marginal(inst, pf, pl, i, freeL, meter)
            srt = np.sort(m)
            reg = srt[min(k - 1, len(srt) - 1)] - srt[0]
            key = (reg, -m.min())
            if best is None or key > best[0]:
                best = (key, i, int(np.argmin(m)))
        _, i, j = best
        pf.append(i)
        pl.append(freeL.pop(j))
        freeF.remove(i)
    p = np.empty(n, dtype=int)
    p[pf] = pl
    return p


def beam(inst, width, meter):
    """Beam over location choices along the busiest-first facility order,
    scored by cumulative exact marginal (= exact partial objective)."""
    n = inst["n"]
    order = busiest_order(inst)
    beams = [(0.0, [], [])]                    # (partial cost, faces, locs)
    for i in order:
        cand = []
        for c, pf, pl in beams:
            free = [l for l in range(n) if l not in pl]
            m = marginal(inst, pf, pl, int(i), free, meter)
            for j in np.argsort(m, kind="stable")[:width]:
                cand.append((c + float(m[j]), pf + [int(i)], pl + [free[int(j)]]))
        cand.sort(key=lambda t: t[0])
        beams = cand[:width]
    c, pf, pl = beams[0]
    p = np.empty(n, dtype=int)
    p[pf] = pl
    return p


# ---------------------------------------------------------------------------
# swap-based improvement search
# ---------------------------------------------------------------------------

def swap_delta(inst, p, Bp, u, v, meter):
    """Exact objective change of swapping the locations of facilities u, v."""
    A = inst["A"]
    n = inst["n"]
    k = np.ones(n, dtype=bool)
    k[[u, v]] = False
    d = ((A[u, k] - A[v, k]) * (Bp[v, k] - Bp[u, k])).sum() \
        + ((A[k, u] - A[k, v]) * (Bp[k, v] - Bp[k, u])).sum() \
        + (A[u, u] - A[v, v]) * (Bp[v, v] - Bp[u, u]) \
        + (A[u, v] - A[v, u]) * (Bp[v, u] - Bp[u, v])
    meter.terms += 8 * n
    return float(d)


def all_deltas(inst, p, Bp):
    """Delta of every swap, dense.  Recomputed from scratch each call; charged
    to the meter by the caller at the standard 2 n^2 update rate."""
    A = inst["A"]
    M = A @ Bp.T
    N = A.T @ Bp
    r, s = np.diag(M), np.diag(N)
    S1 = M + M.T - r[:, None] - r[None, :]
    S2 = N + N.T - s[:, None] - s[None, :]
    a, bd = np.diag(A), np.diag(Bp)
    # remove the k = u and k = v terms included in the full sums
    S1 -= (a[:, None] - A.T) * (Bp.T - bd[:, None])        # k = u
    S1 -= (A - a[None, :]) * (bd[None, :] - Bp)            # k = v
    S2 -= (a[:, None] - A) * (Bp - bd[:, None])            # k = u
    S2 -= (A.T - a[None, :]) * (bd[None, :] - Bp.T)        # k = v
    D = S1 + S2 \
        + (a[:, None] - a[None, :]) * (bd[None, :] - bd[:, None]) \
        + (A - A.T) * (Bp.T - Bp)
    return D


def sa(inst, budget, seed=0, t1_frac=0.01):
    """Single-chain swap SA at the paper's budget, geometric cooling calibrated
    on the instance's own random-swap deltas."""
    n = inst["n"]
    meter = Meter(n)
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    Bp = inst["B"][np.ix_(p, p)]
    cur = float((inst["A"] * Bp).sum())
    meter.terms += n * n
    best, bp = cur, p.copy()

    probe = [abs(swap_delta(inst, p, Bp, *rng.choice(n, 2, replace=False), meter))
             for _ in range(50)]
    t0 = max(np.mean(probe), 1e-9)
    n_moves = int((budget - meter.evals()) * n / 8)
    for s in range(n_moves):
        temp = t0 * (t1_frac ** (s / max(1, n_moves - 1)))
        u, v = rng.choice(n, 2, replace=False)
        d = swap_delta(inst, p, Bp, u, v, meter)
        if d <= 0 or rng.random() < np.exp(-d / max(temp, 1e-12)):
            p[[u, v]] = p[[v, u]]
            Bp = inst["B"][np.ix_(p, p)]
            cur += d
            if cur < best:
                best, bp = cur, p.copy()
    return bp, meter


def tabu(inst, budget, seed=0):
    """Robust-tabu-style search \\citep{Taillard1991}: full-neighbourhood
    steepest descent, per-assignment tabu with random tenure in [n/2, 3n/2],
    aspiration on improving the incumbent.  Parameters are the literature's,
    not tuned on these instances."""
    n = inst["n"]
    meter = Meter(n)
    rng = np.random.default_rng(seed)
    p = rng.permutation(n)
    Bp = inst["B"][np.ix_(p, p)]
    cur = float((inst["A"] * Bp).sum())
    meter.terms += n * n
    best, bp = cur, p.copy()
    tab = np.zeros((n, n), dtype=np.int64)     # facility x location expiry
    iu = np.triu_indices(n, 1)

    it = 0
    while meter.evals() + 2 <= budget:
        it += 1
        D = all_deltas(inst, p, Bp)
        meter.terms += 2 * n * n               # charged at the update rate
        blocked = (tab[np.arange(n)[:, None], p[None, :]] > it) \
            | (tab[np.arange(n)[None, :], p[:, None]].T > it)
        asp = cur + D < best - 1e-9
        Dm = np.where(np.triu(blocked & ~asp, 1)[iu], np.inf, D[iu])
        k = int(np.argmin(Dm))
        if not np.isfinite(Dm[k]):
            tab[:] = 0
            continue
        u, v = int(iu[0][k]), int(iu[1][k])
        a, b = p[u], p[v]
        p[[u, v]] = p[[v, u]]
        Bp = inst["B"][np.ix_(p, p)]
        cur += float(D[u, v])
        tenure = int(rng.integers(n // 2, (3 * n) // 2 + 1))
        tab[u, a] = it + tenure                # do not send u back to a
        tab[v, b] = it + tenure
        if cur < best - 1e-9:
            best, bp = cur, p.copy()
    return bp, meter


# ---------------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="../../benchmarks/qaplib")
    ap.add_argument("--budget", type=int, default=45000)
    ap.add_argument("--beam-width", dest="beam_width", type=int, default=32)
    ap.add_argument("--regret-k", dest="regret_k", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="qap_results.json")
    a = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(a.data, "*.dat"))):
        inst = load(path)
        assert inst["opt"] is not None, f"{inst['name']}: no optimum in header"

        def add(method, p, meter, wall):
            c = cost(inst, p)
            assert sorted(p) == list(range(inst["n"]))
            rows.append({"inst": inst["name"], "n": inst["n"],
                         "opt": inst["opt"], "method": method, "C": c,
                         "gap": (c - inst["opt"]) / inst["opt"],
                         "evals": meter.evals(), "ms": wall * 1000})

        for method, fn in [
            ("busiest-first greedy", lambda m: greedy(inst, m)),
            (f"regret-{a.regret_k} insertion",
             lambda m: regret(inst, a.regret_k, m)),
            (f"beam width {a.beam_width}",
             lambda m: beam(inst, a.beam_width, m)),
        ]:
            meter = Meter(inst["n"])
            t = time.time()
            p = fn(meter)
            add(method, p, meter, time.time() - t)

        t = time.time()
        p, meter = sa(inst, a.budget, seed=a.seed)
        add(f"swap SA {a.budget}", p, meter, time.time() - t)

        t = time.time()
        p, meter = tabu(inst, a.budget, seed=a.seed)
        add(f"swap tabu {a.budget}", p, meter, time.time() - t)

        got = {r["method"]: r for r in rows if r["inst"] == inst["name"]}
        print(f"{inst['name']:8s} n={inst['n']:3d} opt={inst['opt']:12.0f}  "
              + "  ".join(f"{m.split()[0]}={100 * r['gap']:5.2f}%"
                          for m, r in got.items()), flush=True)
        with open(a.out, "w") as f:
            json.dump({"budget": a.budget, "rows": rows}, f)

    print(f"\nwrote {a.out} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
