"""
The witness-certified decoder with a training-free selector.

Same mechanism as `certified_decode.run_certified_decode`, same witness
generator, same rng seed and consumption order, same candidate budget
(kappa=32) and same carried-witness fallback; the only replaced part is the
selector.  Rooms are committed largest-first and positions are ranked by the
exact marginal objective (`GreedyConstructor._gain`), i.e. the strongest
training-free constructive rule of the paper, run inside the identical
certified mechanism.  Whatever quality difference remains against
`policy_witness_certified` is what learning contributes *inside* the
guarantee; coverage cannot differ, because the initial witness precedes
selection.

Step-level certificate calls use a single deterministic attempt
(`step_tries=1`), so the whole run is deterministic given the witness seed;
the script asserts that its certified flags equal the recorded coverage flags
of the policy study, cell for cell.
"""

import argparse
import json
import time

import numpy as np

import bench
import core
from certified_decode import FirstFitWitness
from greedy_construct import GreedyConstructor


def decode_cell(cell, effort, n_cand=32, step_tries=1, canvas=44,
                init="firstfit"):
    inst = cell["inst"]
    B, n = inst.B, inst.n
    iw = np.rint(inst.w).astype(np.int64)
    ih = np.rint(inst.h).astype(np.int64)
    wits = [FirstFitWitness(iw[b], ih[b], canvas, canvas,
                            int(round(inst.gw[b])), int(round(inst.gh[b])))
            for b in range(B)]

    # initial witnesses first, in batch order, exactly as run_certified_decode.
    # init="contact" swaps only the empty-plate generator for the best-contact
    # rule; the witness is verified by the same checker, and every later
    # certificate call is unchanged, so the guarantee is unaffected.
    rng = np.random.default_rng(0)
    certified = np.zeros(B, dtype=bool)
    witness = [None] * B
    for b in range(B):
        if init == "contact":
            from witness_frontier import best_contact
            order = sorted(range(n), key=lambda i: -(iw[b][i] * ih[b][i]))
            ok, w = best_contact(wits[b], order)
            if ok and not wits[b].verify(
                    np.zeros((canvas, canvas), dtype=np.int32),
                    list(range(n)), w):
                raise RuntimeError("contact witness failed verification")
        else:
            ok, w = wits[b].certify(np.zeros((canvas, canvas), dtype=np.int32),
                                    list(range(n)), tries=effort, rng=rng)
        certified[b] = ok
        witness[b] = w

    t0 = time.time()
    x = np.zeros((B, n))
    y = np.zeros((B, n))
    witness_used = 0
    rejects = 0
    for b in range(B):
        if not certified[b]:
            continue
        sub = inst.take([b])
        gc = GreedyConstructor(sub, canvas=canvas)
        order = np.argsort(-(sub.w[0] * sub.h[0]))
        occ = np.zeros((1, canvas, canvas), dtype=np.int32)
        placed = np.zeros((1, n), dtype=bool)
        wit = witness[b]
        for t in range(n):
            i = int(order[t])
            legal, _, _ = gc._legal(occ, i)
            g = gc._gain(i, x[b:b + 1], y[b:b + 1], placed)
            score = np.where(legal[0], g[0], -np.inf)
            flat_order = np.argsort(score, axis=None)[::-1]
            rem = [j for j in range(n) if not placed[0, j] and j != i]
            wit_pos = next(((r, c) for (j, r, c) in wit if j == i))

            chosen = None
            for k in range(min(n_cand, flat_order.size)):
                f = int(flat_order[k])
                r, c = f // canvas, f % canvas
                if not np.isfinite(score[r, c]):
                    break
                trial = occ[0].copy()
                trial[r:r + ih[b, i], c:c + iw[b, i]] += 1
                ok, w = wits[b].certify(trial, rem, tries=step_tries, rng=rng)
                if ok:
                    chosen = (r, c, trial, w)
                    break
                rejects += 1

            if chosen is None:
                # the theorem's insurance: the remainder of the carried witness
                r, c = wit_pos
                trial = occ[0].copy()
                trial[r:r + ih[b, i], c:c + iw[b, i]] += 1
                w = [(j, wr, wc) for (j, wr, wc) in wit if j != i]
                if not wits[b].verify(trial, rem, w):
                    raise RuntimeError(
                        f"carried witness invariant failed, instance {b} room {i}")
                chosen = (r, c, trial, w)
                witness_used += 1

            r, c, occ[0], wit = chosen
            placed[0, i] = True
            x[b, i] = c + inst.w[b, i] / 2.0
            y[b, i] = r + inst.h[b, i] / 2.0

    return {"x": x, "y": y, "certified": certified,
            "witness_used": witness_used, "rejects": rejects,
            "wall_time": time.time() - t0}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", default="certified_study_v3.json")
    p.add_argument("--efforts", nargs="+", type=int, default=[1, 128])
    p.add_argument("--n-cand", type=int, default=32)
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--out", default="certified_greedy_v3.json")
    a = p.parse_args()

    with open(a.study, encoding="utf-8") as f:
        study = json.load(f)
    pol = [r for r in study["records"]
           if r["method"] == "policy_witness_certified"
           and r["witness_effort"] in a.efforts]
    cells = sorted({(r["suite"], r["suite_tag"], r["split"], r["n_rooms"],
                     r["fill"], r["witness_effort"], r["instances"])
                    for r in pol})
    print(f"{len(cells)} (cell, effort) combinations")

    out_rows = []
    for suite, tag, split, n_rooms, fill, effort, n_inst in cells:
        cell = bench.suite_cell(split, n_rooms, fill, n_inst, tag=tag)
        obj = core.BatchObjective(cell["inst"])
        res = decode_cell(cell, effort, n_cand=a.n_cand, canvas=a.canvas)
        tot, comp = obj.evaluate(res["x"], res["y"], components=True)
        ret = res["certified"]
        if np.any(ret & ~(comp["overlap_area"] <= 1e-9)):
            raise RuntimeError("certified greedy returned an infeasible layout")

        # the same mechanism must produce the same coverage as the policy runs
        mates = [r for r in pol if (r["suite"], r["suite_tag"], r["n_rooms"],
                                    r["fill"], r["witness_effort"])
                 == (suite, tag, n_rooms, fill, effort)]
        deltas = {}
        for m in mates:
            rec_ret = np.asarray(m["returned"], dtype=bool)
            if not np.array_equal(ret, rec_ret):
                raise RuntimeError(f"coverage mismatch vs {m['policy']} in "
                                   f"{suite} n={n_rooms} f={fill} e={effort}")
            jp = np.array([v for v, q in zip(m["J"], rec_ret) if q])
            deltas[m["policy"]] = (tot[ret] - jp)

        row = {
            "suite": suite, "n": n_rooms, "fill": fill, "effort": effort,
            "returned": int(ret.sum()), "instances": int(n_inst),
            "J": [float(v) if q else None for v, q in zip(tot, ret)],
            "mean_J": float(tot[ret].mean()) if ret.any() else None,
            "witness_used": res["witness_used"], "rejects": res["rejects"],
            "ms_per_instance": 1000.0 * res["wall_time"] / n_inst,
            "greedy_minus_policy": {
                k: {"mean": float(d.mean()),
                    "win_tie_loss": [int((d > 1e-9).sum()),
                                     int((np.abs(d) <= 1e-9).sum()),
                                     int((d < -1e-9).sum())]}
                for k, d in deltas.items()},
        }
        out_rows.append(row)
        dtxt = "  ".join(f"{k}:{v['mean']:+7.1f}" for k, v in
                         row["greedy_minus_policy"].items())
        print(f"{suite:5s} n={n_rooms:2d} f={fill:.2f} e={effort:3d} "
              f"ret={row['returned']:3d}/{n_inst:3d} J={row['mean_J'] or 0:7.1f} "
              f"fb={res['witness_used']:3d} rej={res['rejects']:5d}  "
              f"greedy-policy: {dtxt}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"metadata": {"selector": "largest-first, exact marginal",
                                "n_cand": a.n_cand, "step_tries": 1,
                                "witness_seed": 0},
                   "rows": out_rows}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
