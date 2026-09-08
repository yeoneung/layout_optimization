"""
Paired statistics over the benchmark suite.

Every method solves every instance, so method differences are paired and the
right analysis is on within-instance differences rather than on group means.
Three things are reported for each comparison against a chosen reference method:

  * the mean paired gap in percent, with a 95% bootstrap confidence interval
    over instances (not over layouts, and not over cells);
  * a Wilcoxon signed-rank test on the paired differences, Holm-corrected across
    the family of methods compared against the same reference;
  * the number of instances on which each method wins, so a small mean gap driven
    by a few instances cannot pass as a uniform advantage.

Gaps are normalized per instance by the best value any method achieved on that
instance, because raw objective values differ by an order of magnitude between
an 8-room and a 64-room cell and averaging them directly would weight the large
cells arbitrarily.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy import stats


def load(path):
    with open(path) as f:
        d = json.load(f)
    by = defaultdict(dict)                    # (n, fill, i) -> method -> record
    for r in d["records"]:
        by[(r["n_rooms"], r["fill"], r["instance"])][r["method"]] = r
    return by


def gap_matrix(by, methods):
    """(n_instances, n_methods) normalized gap to the per-instance best, in %."""
    keys = [k for k, v in sorted(by.items()) if all(m in v for m in methods)]
    G = np.zeros((len(keys), len(methods)))
    for r, k in enumerate(keys):
        vals = np.array([by[k][m]["J"] for m in methods])
        best = vals.max()
        denom = max(abs(best), 1e-9)
        G[r] = 100.0 * (best - vals) / denom
    return keys, G


def boot_ci(d, n_boot=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    out = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        out[i] = min(1.0, running)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bench", default="bench_test.json")
    p.add_argument("--reference", default="constructive policy")
    p.add_argument("--methods", nargs="*", default=None)
    p.add_argument("--by-cell", dest="by_cell", action="store_true")
    a = p.parse_args()

    by = load(a.bench)
    all_methods = sorted({m for v in by.values() for m in v})
    methods = a.methods or all_methods
    methods = [m for m in methods if m in all_methods]
    if a.reference not in methods:
        print(f"reference '{a.reference}' not present; methods are:")
        for m in all_methods:
            print("   ", m)
        return

    keys, G = gap_matrix(by, methods)
    ref = methods.index(a.reference)
    print(f"{len(keys)} instances, {len(methods)} methods, "
          f"reference = {a.reference}\n")
    print(f"{'method':30s} {'gap%':>7s} {'95% CI':>17s} {'vs ref%':>9s} "
          f"{'p(Holm)':>9s} {'wins':>6s}")

    others = [i for i in range(len(methods)) if i != ref]
    raw_p = []
    for i in others:
        d = G[:, i] - G[:, ref]
        raw_p.append(stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0)
    adj = holm(np.array(raw_p))

    rows = []
    for j, i in enumerate(others):
        d = G[:, i] - G[:, ref]
        lo, hi = boot_ci(G[:, i])
        wins = int(np.sum(G[:, i] < G[:, ref] - 1e-12))
        rows.append((G[:, i].mean(), methods[i], lo, hi, d.mean(), adj[j], wins))
    lo, hi = boot_ci(G[:, ref])
    rows.append((G[:, ref].mean(), methods[ref] + "  (reference)", lo, hi,
                 0.0, float("nan"), 0))

    for mean, name, lo, hi, dv, pv, wins in sorted(rows):
        ps = "  --" if np.isnan(pv) else f"{pv:9.2e}"
        print(f"{name:30s} {mean:7.2f} [{lo:7.2f},{hi:7.2f}] {dv:9.2f} {ps} "
              f"{wins:6d}/{len(keys)}")

    if a.by_cell:
        print("\nmean gap%% by cell")
        cells = sorted({(k[0], k[1]) for k in keys})
        print(f"{'cell':>14s} " + " ".join(f"{m[:13]:>14s}" for m in methods))
        for c in cells:
            sel = [r for r, k in enumerate(keys) if (k[0], k[1]) == c]
            vals = G[sel].mean(axis=0)
            print(f"n={c[0]:3d} f={c[1]:.2f} " + " ".join(f"{v:14.2f}" for v in vals))


if __name__ == "__main__":
    main()
