"""
What the certified decoder adds over the witness it starts from.

The certified decoder verifies a complete feasible witness w0 before its first
commitment, so feasibility of its returns is guaranteed by construction; the
empirical question is how much objective the certified deviations from w0 buy.
This script regenerates w0 for every certified-return population exactly as
`run_certified_decode` does (same generator class, same rng seed and
consumption order), verifies that its success pattern matches the recorded
coverage flags bit for bit, evaluates J(w0), and pairs it with the recorded
per-instance J of the certified returns.

CPU only; no policy or torch involved, because w0 never depends on the selector.
"""

import argparse
import json
from collections import defaultdict

import numpy as np

import bench
import core
from certified_decode import FirstFitWitness


def witness_layouts(cell, effort, canvas):
    """Reproduce the initial witnesses of `run_certified_decode(seed=0)`."""
    inst = cell["inst"]
    B, n = inst.w.shape
    iw = np.rint(inst.w).astype(np.int64)
    ih = np.rint(inst.h).astype(np.int64)
    ok = np.zeros(B, dtype=bool)
    x = np.zeros((B, n)); y = np.zeros((B, n))
    rng = np.random.default_rng(0)          # decode always passes seed=0
    for b in range(B):
        gen = FirstFitWitness(iw[b], ih[b], canvas, canvas,
                              int(round(inst.gw[b])), int(round(inst.gh[b])))
        good, wit = gen.certify(np.zeros((canvas, canvas), dtype=np.int32),
                                list(range(n)), tries=effort, rng=rng)
        ok[b] = good
        if good:
            for i, r, c in wit:
                x[b, i] = c + inst.w[b, i] / 2.0
                y[b, i] = r + inst.h[b, i] / 2.0
    return ok, x, y


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--study", default="certified_study_v3.json")
    p.add_argument("--canvas", type=int, default=44)
    p.add_argument("--out", default="witness_gap_v3.json")
    a = p.parse_args()

    with open(a.study, encoding="utf-8") as f:
        study = json.load(f)
    recs = [r for r in study["records"]
            if r["method"] == "policy_witness_certified"]
    print(f"{len(recs)} certified records")

    # witness generation is selector-free: one (cell, effort) computation
    # serves every policy seed that shares it
    wit_cache = {}
    cell_cache = {}
    rows, total_pairs, total_better, total_worse = [], 0, 0, 0
    deltas_all = []
    for r in recs:
        ckey = (r["suite"], r["suite_tag"], r["n_rooms"], r["fill"])
        if ckey not in cell_cache:
            cell = bench.suite_cell(r["split"], r["n_rooms"], r["fill"],
                                    r["instances"], tag=r["suite_tag"])
            cell_cache[ckey] = (cell, core.BatchObjective(cell["inst"]))
        cell, obj = cell_cache[ckey]

        wkey = ckey + (r["witness_effort"],)
        if wkey not in wit_cache:
            ok, x, y = witness_layouts(cell, r["witness_effort"], a.canvas)
            tot, comp = obj.evaluate(x, y, components=True)
            feas = comp["overlap_area"] <= 1e-9
            if np.any(ok & ~feas):
                raise RuntimeError(f"regenerated witness infeasible in {wkey}")
            wit_cache[wkey] = (ok, tot)
        ok, jw = wit_cache[wkey]

        returned = np.asarray(r["returned"], dtype=bool)
        if not np.array_equal(ok, returned):
            raise RuntimeError(
                f"witness success pattern does not match recorded coverage "
                f"for {wkey} policy={r['policy']}: "
                f"{ok.sum()} regenerated vs {returned.sum()} recorded")

        jf = np.array([v for v, rt in zip(r["J"], returned) if rt])
        j0 = jw[returned]
        d = jf - j0
        n_pairs = len(d)
        total_pairs += n_pairs
        total_better += int((d > 1e-9).sum())
        total_worse += int((d < -1e-9).sum())
        deltas_all.append(d)
        se = d.std(ddof=1) / np.sqrt(n_pairs) if n_pairs > 1 else 0.0
        rows.append({
            "suite": r["suite"], "n": r["n_rooms"], "fill": r["fill"],
            "effort": r["witness_effort"], "policy": r["policy"],
            "pairs": n_pairs,
            "mean_J_witness": float(j0.mean()),
            "mean_J_certified": float(jf.mean()),
            "mean_delta": float(d.mean()),
            "delta_ci95": [float(d.mean() - 1.96 * se),
                           float(d.mean() + 1.96 * se)],
            "win_tie_loss": [int((d > 1e-9).sum()),
                             int((np.abs(d) <= 1e-9).sum()),
                             int((d < -1e-9).sum())],
            "rel_improvement": float(d.mean() / abs(j0.mean())),
            "fallback_share": (r["summary"]["witness_used_total"]
                               / (n_pairs * r["n_rooms"]) if n_pairs else None),
            "ms_per_instance": r["summary"]["ms_per_instance"],
        })

    hdr = (f"{'suite':6s} {'n':>3s} {'fill':>5s} {'eff':>4s} {'pol':6s} "
           f"{'N':>4s} {'J(w0)':>8s} {'J(cert)':>8s} {'delta':>8s} "
           f"{'rel%':>6s} {'W/T/L':>12s} {'fb%':>5s}")
    print(hdr)
    for w in sorted(rows, key=lambda w: (w["suite"], w["n"], w["fill"],
                                         w["effort"], w["policy"])):
        wtl = "/".join(str(v) for v in w["win_tie_loss"])
        print(f"{w['suite']:6s} {w['n']:3d} {w['fill']:5.2f} {w['effort']:4d} "
              f"{w['policy']:6s} {w['pairs']:4d} {w['mean_J_witness']:8.1f} "
              f"{w['mean_J_certified']:8.1f} {w['mean_delta']:8.1f} "
              f"{100 * w['rel_improvement']:5.1f}% {wtl:>12s} "
              f"{100 * (w['fallback_share'] or 0):4.1f}%")

    d = np.concatenate(deltas_all)
    print(f"\nALL certified returns: {total_pairs} pairs, "
          f"improved {total_better}, worsened {total_worse}, "
          f"mean delta {d.mean():.1f}, min {d.min():.1f}")

    with open(a.out, "w", encoding="utf-8") as f:
        json.dump({"rows": rows,
                   "overall": {"pairs": total_pairs, "improved": total_better,
                               "worsened": total_worse,
                               "mean_delta": float(d.mean()),
                               "min_delta": float(d.min())}}, f, indent=1)
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
