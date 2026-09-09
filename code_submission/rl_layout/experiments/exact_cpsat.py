"""
Exact optimum for small instances, so the paper can report an optimality gap.

Every comparison elsewhere in this work is between methods.  That says which
method is better; it does not say whether any of them is close to the best
possible layout.  This module closes that by solving the placement problem
exactly with CP-SAT on the small cells of the benchmark, turning every reported
number into a gap against a proven optimum rather than against the best
heuristic we happened to run.

Model.  Facility i has integer lower-left coordinates (x_i, y_i) inside the
plate.  Non-overlap is CP-SAT's native two-dimensional no-overlap over interval
variables, which is far stronger than a per-cell packing encoding.  The pairwise
adjacency term is not linear in the coordinates -- it is a clipped power of a
centre distance with directional radii removed -- but it is a function of the
coordinate *differences* alone: the centre offset of a pair is
(dx, dy) = (x_j - x_i + (w_j - w_i)/2, y_j - y_i + (h_j - h_i)/2).  So for each
pair with non-zero affinity we introduce integer difference variables
u = x_j - x_i and v = y_j - y_i and tabulate the exact contribution over
(u, v) only.  An earlier version of this model tabulated the full
(x_i, y_i, x_j, y_j) product, which at n = 8 produced ~1.6e5 rows and a model
CP-SAT could not close in 300 s; the difference form has ~(2W)(2H) rows per
pair, an order of magnitude fewer, and propagates through the differences
directly.  Wall bonuses are reified booleans on the coordinates.  Nothing is
approximated except the conversion of the objective to integers at a fixed
scale, which is reported.

The geometry helpers below are transcribed from `core.py` rather than imported,
because this file runs in a separate interpreter that has ortools installed.
`audit_exact_results.py` checks the returned layouts against the stated
real-valued objective and transfers the integer-model bound with an explicit
rounding allowance.

Scale.  Table size grows as |P_i| x |P_j| per affine pair, so this is meant for
the n = 8 cells -- exactly where calibrating the heuristics against the truth is
possible.
"""

import argparse
import json
import math
import time

try:
    from ortools.sat.python import cp_model
except ImportError:                       # only the exact environment has it
    cp_model = None

SCALE = 10_000                            # objective is rational, CP-SAT is integral


def quantization_error_bound(spec, scale=SCALE):
    """Uniform error bound for the scaled integer objective.

    Exactly one rounded coefficient is active for every nonzero pair term and
    every nonzero wall-preference term.  Nearest-integer rounding contributes
    at most ``1/(2*scale)`` per active term.  The no-overlap bonus is added
    outside the integer sum and is not rounded.
    """
    n = spec["n"]
    pairs = sum(spec["adj"][i][j] != 0.0
                for i in range(n) for j in range(i + 1, n))
    walls = sum(spec[key][i] != 0.0
                for key in ("edge_side", "edge_bottom", "edge_top")
                for i in range(n))
    return (pairs + walls) / (2.0 * scale), pairs + walls


def evaluate_real_objective(spec, x, y):
    """Evaluate the stated real-valued objective for one feasible layout."""
    n = spec["n"]
    value = float(spec["no_overlap_bonus"])
    for i in range(n):
        left = x[i] - spec["w"][i] / 2.0
        bottom = y[i] - spec["h"][i] / 2.0
        if left == 0 or left == spec["gw"] - spec["w"][i]:
            value += spec["edge_side"][i]
        if bottom == 0:
            value += spec["edge_bottom"][i]
        if bottom == spec["gh"] - spec["h"][i]:
            value += spec["edge_top"][i]
    for i in range(n):
        for j in range(i + 1, n):
            affinity = spec["adj"][i][j]
            if affinity == 0.0:
                continue
            distance = _pair_distance(x[i], y[i], spec["w"][i], spec["h"][i],
                                      x[j], y[j], spec["w"][j], spec["h"][j])
            dmax = max(spec["dmax"][f"{i},{j}"], 1e-12)
            normalized = min(max(distance / dmax, 0.0), 1.0) ** \
                spec["distance_power"]
            contribution = (affinity * normalized if affinity >= 0
                            else (-affinity) * (1.0 - normalized))
            value += spec["adj_weight"] * contribution
    return value


def _junk_distance(tan, w, h):
    """Centre-to-boundary distance of a w x h rectangle along slope `tan`."""
    hw, hh = w / 2.0, h / 2.0
    if tan > h / w:
        kw = 0.0 if math.isinf(tan) else hh / tan
        kh = hh
    else:
        kw = hw
        kh = 0.0 if math.isinf(tan) else hw * tan
    return math.hypot(kw, kh)


def _pair_distance(x1, y1, w1, h1, x2, y2, w2, h2):
    """Centre distance with both rectangles' directional radii removed."""
    dx, dy = x2 - x1, y2 - y1
    adx, ady = abs(dx), abs(dy)
    tan = math.inf if adx <= 1e-12 else ady / adx
    return (math.hypot(dx, dy)
            - _junk_distance(tan, w1, h1) - _junk_distance(tan, w2, h2))


def build_and_solve(spec, time_limit=300.0, workers=8, log=False):
    n = spec["n"]
    GW, GH = spec["gw"], spec["gh"]
    w, h = spec["w"], spec["h"]
    adj = spec["adj"]
    dmax = spec["dmax"]                   # dict keyed "i,j" for i < j
    aw = spec["adj_weight"]
    power = spec["distance_power"]

    m = cp_model.CpModel()
    X = [m.NewIntVar(0, GW - w[i], f"x{i}") for i in range(n)]
    Y = [m.NewIntVar(0, GH - h[i], f"y{i}") for i in range(n)]
    xiv = [m.NewIntervalVar(X[i], w[i], m.NewIntVar(w[i], GW, f"xe{i}"), f"ix{i}")
           for i in range(n)]
    yiv = [m.NewIntervalVar(Y[i], h[i], m.NewIntVar(h[i], GH, f"ye{i}"), f"iy{i}")
           for i in range(n)]
    m.AddNoOverlap2D(xiv, yiv)

    terms = []

    # ---- wall bonuses: reified on the coordinates -------------------------
    for i in range(n):
        if spec["edge_side"][i]:
            bl, br, bs = (m.NewBoolVar(f"l{i}"), m.NewBoolVar(f"r{i}"),
                          m.NewBoolVar(f"s{i}"))
            m.Add(X[i] == 0).OnlyEnforceIf(bl)
            m.Add(X[i] != 0).OnlyEnforceIf(bl.Not())
            m.Add(X[i] == GW - w[i]).OnlyEnforceIf(br)
            m.Add(X[i] != GW - w[i]).OnlyEnforceIf(br.Not())
            m.AddMaxEquality(bs, [bl, br])
            terms.append((int(round(spec["edge_side"][i] * SCALE)), bs))
        if spec["edge_bottom"][i]:
            bb = m.NewBoolVar(f"b{i}")
            m.Add(Y[i] == 0).OnlyEnforceIf(bb)
            m.Add(Y[i] != 0).OnlyEnforceIf(bb.Not())
            terms.append((int(round(spec["edge_bottom"][i] * SCALE)), bb))
        if spec["edge_top"][i]:
            bt = m.NewBoolVar(f"t{i}")
            m.Add(Y[i] == GH - h[i]).OnlyEnforceIf(bt)
            m.Add(Y[i] != GH - h[i]).OnlyEnforceIf(bt.Not())
            terms.append((int(round(spec["edge_top"][i] * SCALE)), bt))

    # ---- pairwise adjacency: exact cost table over coordinate differences --
    cost_vars = []
    rows_total = 0
    for i in range(n):
        for j in range(i + 1, n):
            a = adj[i][j]
            if a == 0.0:
                continue
            dm = max(dmax[f"{i},{j}"], 1e-12)
            u_lo, u_hi = -(GW - w[i]), GW - w[j]
            v_lo, v_hi = -(GH - h[i]), GH - h[j]
            table = []
            lo, hi = math.inf, -math.inf
            for du in range(u_lo, u_hi + 1):
                dx = du + (w[j] - w[i]) / 2.0
                for dv in range(v_lo, v_hi + 1):
                    dy = dv + (h[j] - h[i]) / 2.0
                    d = _pair_distance(0.0, 0.0, w[i], h[i],
                                       dx, dy, w[j], h[j])
                    nd = min(max(d / dm, 0.0), 1.0) ** power
                    c = a * nd if a >= 0 else (-a) * (1.0 - nd)
                    ic = int(round(c * aw * SCALE))
                    table.append((du, dv, ic))
                    lo, hi = min(lo, ic), max(hi, ic)
            u = m.NewIntVar(u_lo, u_hi, f"u{i}_{j}")
            v = m.NewIntVar(v_lo, v_hi, f"v{i}_{j}")
            m.Add(u == X[j] - X[i])
            m.Add(v == Y[j] - Y[i])
            cv = m.NewIntVar(int(lo), int(hi), f"c{i}_{j}")
            m.AddAllowedAssignments([u, v, cv], table)
            cost_vars.append(cv)
            rows_total += len(table)

    m.Maximize(sum(c * v for c, v in terms) + sum(cost_vars))

    s = cp_model.CpSolver()
    s.parameters.max_time_in_seconds = time_limit
    s.parameters.num_search_workers = workers
    s.parameters.log_search_progress = log
    t = time.time()
    st = s.Solve(m)
    wall = time.time() - t

    rounding_delta, rounded_terms = quantization_error_bound(spec)
    out = {"status": s.StatusName(st), "wall_s": wall, "table_rows": rows_total,
           "scale": SCALE, "rounded_terms": rounded_terms,
           "quantization_error_bound": rounding_delta}
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        # the no-overlap bonus is constant across the feasible set
        out["x"] = [s.Value(X[i]) + w[i] / 2.0 for i in range(n)]
        out["y"] = [s.Value(Y[i]) + h[i] / 2.0 for i in range(n)]
        out["quantized_objective"] = (s.ObjectiveValue() / SCALE
                                      + spec["no_overlap_bonus"])
        out["quantized_bound"] = (s.BestObjectiveBound() / SCALE
                                  + spec["no_overlap_bonus"])
        out["objective"] = evaluate_real_objective(spec, out["x"], out["y"])
        out["bound"] = out["quantized_bound"] + rounding_delta
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--specs", default="exact_specs.json")
    p.add_argument("--out", default="exact_results.json")
    p.add_argument("--time-limit", dest="time_limit", type=float, default=300.0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--limit", type=int, default=None)
    a = p.parse_args()

    with open(a.specs) as f:
        specs = json.load(f)["specs"]
    if a.limit:
        specs = specs[:a.limit]

    res = []
    for k, sp in enumerate(specs):
        r = build_and_solve(sp, a.time_limit, a.workers)
        r.update({key: sp[key] for key in ("split", "n_rooms", "fill", "instance")})
        res.append(r)
        print(f"[{k + 1}/{len(specs)}] n={sp['n_rooms']} fill={sp['fill']} "
              f"i={sp['instance']:2d}  {r['status']:12s} "
              f"obj={r.get('objective', float('nan')):9.2f} "
              f"bound={r.get('bound', float('nan')):9.2f} "
              f"rows={r['table_rows']:9d}  {r['wall_s']:6.1f}s", flush=True)
        with open(a.out, "w") as f:
            json.dump({"results": res}, f, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
