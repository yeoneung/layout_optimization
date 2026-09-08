"""
Three-layer reporting: reliability, conditional quality, and operational cost.

Comparing mean objective across methods with different feasibility rates is
biased in both directions.  Averaging over all instances charges infeasible
layouts an arbitrary penalty; averaging over each method's own feasible subset
lets an unreliable method look good by having survived only the easy instances.
Neither is a defensible summary on its own, so this module reports the three
quantities that are.

  reliability          per-method success rate with a Clopper--Pearson interval,
                       so that "feasible on all 20" is reported as the interval
                       it actually supports rather than as certainty;

  conditional quality  paired objective differences restricted to the instances
                       both methods solved feasibly, which is the only
                       comparison free of selection effects;

  operational cost     time and evaluation counts, reported beside the above
                       rather than folded into it.

The Clopper--Pearson bounds are computed from the beta quantile so that no
normal approximation is used at rates of 0 or 1, where it is worst.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats


def clopper_pearson(k, n, alpha=0.05):
    """Exact binomial interval; handles k = 0 and k = n without approximation."""
    lo = 0.0 if k == 0 else stats.beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else stats.beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def load(path, n_rooms=None, fills=None):
    with open(path) as f:
        recs = json.load(f)["records"]
    if n_rooms is not None:
        recs = [r for r in recs if r["n_rooms"] == n_rooms]
    if fills is not None:
        recs = [r for r in recs if any(abs(r["fill"] - f) < 1e-9 for f in fills)]
    by = defaultdict(dict)
    for r in recs:
        by[(r["n_rooms"], r["fill"], r["instance"])][r["method"]] = r
    return by


def reliability(by, methods):
    out = {}
    for m in methods:
        ks = [v[m] for v in by.values() if m in v]
        k = sum(1 for r in ks if r["feasible"])
        n = len(ks)
        lo, hi = clopper_pearson(k, n)
        out[m] = (k, n, k / max(n, 1), lo, hi)
    return out


def paired_conditional(by, a, b):
    """Objective difference a - b on instances both solved feasibly."""
    da, n_both = [], 0
    for v in by.values():
        if a in v and b in v and v[a]["feasible"] and v[b]["feasible"]:
            da.append(v[a]["J"] - v[b]["J"])
            n_both += 1
    if n_both < 3:
        return n_both, float("nan"), (float("nan"),) * 2, float("nan")
    d = np.array(da)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(10000, len(d)))
    means = d[idx].mean(axis=1)
    ci = (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
    p = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return n_both, float(d.mean()), ci, float(p)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="bench_density.json")
    p.add_argument("--rooms", type=int, default=32)
    p.add_argument("--fills", nargs="+", type=float, default=None)
    p.add_argument("--reference", required=True)
    p.add_argument("--methods", nargs="*", default=None)
    a = p.parse_args()

    by = load(a.bench, a.rooms, a.fills)
    all_m = sorted({m for v in by.values() for m in v})
    methods = [m for m in (a.methods or all_m) if m in all_m]
    if a.reference not in methods:
        print("reference not present. available:")
        for m in all_m:
            print("   ", m)
        return

    rel = reliability(by, methods)
    print(f"{len(by)} instances\n")
    print("RELIABILITY  (success rate, 95% Clopper-Pearson)")
    print(f"{'method':32s} {'k/n':>9s} {'rate':>6s} {'95% CI':>16s}")
    for m in sorted(methods, key=lambda m: -rel[m][2]):
        k, n, r, lo, hi = rel[m]
        print(f"{m[:32]:32s} {f'{k}/{n}':>9s} {r:6.2f}  [{lo:5.2f},{hi:5.2f}]")

    print(f"\nCONDITIONAL QUALITY  (objective difference against "
          f"'{a.reference}', instances both solved)")
    print(f"{'method':32s} {'n_both':>7s} {'mean diff':>10s} {'95% CI':>19s} {'p':>9s}")
    for m in methods:
        if m == a.reference:
            continue
        nb, md, ci, pv = paired_conditional(by, m, a.reference)
        cis = f"[{ci[0]:8.1f},{ci[1]:8.1f}]" if nb >= 3 else " " * 19
        print(f"{m[:32]:32s} {nb:7d} {md:10.1f} {cis} {pv:9.2e}")


if __name__ == "__main__":
    main()
