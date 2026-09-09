"""Plot the primary anytime comparison and initialization coverage."""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def mean_ci(values, rng, draws=5000):
    values = np.asarray(values, dtype=float)
    samples = rng.choice(values, (draws, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), np.quantile(samples, [0.025, 0.975])


def wilson(successes, total, z=1.96):
    p = successes / total
    d = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / d
    half = z * np.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / d
    return center - half, center + half


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="completion_comparison_dense50.json")
    parser.add_argument("--out", default="../../figs/fig_completion_comparison")
    args = parser.parse_args()
    with open(args.input, encoding="utf-8") as handle:
        data = json.load(handle)

    initial = {}
    values = defaultdict(list)
    for run in data["runs"]:
        identifier = (run["geometry"], run["n"], run["fill"], run["instance"])
        initial[identifier] = run["initial_J"]
        for point in run["rows"]:
            values[(identifier, run["method"], point["budget_s"])].append(point["J"])

    averaged = {key: float(np.mean(row)) for key, row in values.items()}
    budgets = sorted({key[2] for key in averaged})
    methods = ["B0", "B1", "B2", "B2S", "B3", "M0", "R0", "M1"]
    colors = {"B0": "#777777", "B1": "#d95f02", "B2": "#7570b3",
              "B2S": "#e6ab02", "B3": "#1b9e77", "M0": "#66a61e",
              "R0": "#a6761d", "M1": "#1f4e9e"}
    markers = {"B0": "o", "B1": "s", "B2": "^", "B2S": "<",
               "B3": "D", "M0": "v", "R0": ">", "M1": "P"}
    rng = np.random.default_rng(20260907)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.65),
                             gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    for method in methods:
        means, lower, upper = [], [], []
        for budget in budgets:
            row = []
            for identifier in sorted(initial):
                key = (identifier, method, budget)
                if key in averaged:
                    row.append(100.0 * (averaged[key] - initial[identifier])
                               / initial[identifier])
            mean, interval = mean_ci(row, rng)
            means.append(mean)
            lower.append(mean - interval[0])
            upper.append(interval[1] - mean)
        ax.errorbar(budgets, means, yerr=[lower, upper], label=method,
                    color=colors[method], marker=markers[method], linewidth=1.7,
                    markersize=5, capsize=2.5)
    ax.set_xlabel("post-initialization budget (s)")
    ax.set_ylabel("mean improvement over $w_0$ (%)")
    ax.set_xticks(budgets)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, ncol=4, fontsize=7.5, loc="upper left")
    ax.set_title("(a) Anytime quality on 196 instances", loc="left", fontsize=10)

    ax = axes[1]
    grouped = defaultdict(list)
    for row in data["initializations"]:
        grouped[(row["geometry"], row["n"])].append(row)
    for (geometry, n), rows in sorted(grouped.items()):
        successes = sum(row["initialized"] for row in rows)
        coverage = successes / len(rows)
        lo, hi = wilson(successes, len(rows))
        mean_ms = 1000.0 * np.mean([row["initialization_s"] for row in rows])
        color = "#1f4e9e" if geometry == "guillotine" else "#d95f02"
        marker = "o" if n == 32 else "s"
        ax.errorbar([mean_ms], [coverage], yerr=[[coverage - lo], [hi - coverage]],
                    color=color, marker=marker, markersize=6, capsize=3)
        label = ("G" if geometry == "guillotine" else "N") + str(n)
        ax.annotate(label, (mean_ms, coverage), xytext=(4, -10),
                    textcoords="offset points", fontsize=8)
    ax.set_xlabel("mean initialization time (ms)")
    ax.set_ylabel("initialization coverage")
    ax.set_ylim(0.78, 1.025)
    ax.grid(axis="y", alpha=0.25)
    ax.set_title("(b) Best-contact initialization", loc="left", fontsize=10)
    ax.text(0.02, 0.02, "G: guillotine   N: nonslicing",
            transform=ax.transAxes, fontsize=8)

    fig.tight_layout()
    directory = os.path.dirname(args.out)
    if directory:
        os.makedirs(directory, exist_ok=True)
    fig.savefig(args.out + ".pdf", bbox_inches="tight")
    fig.savefig(args.out + ".png", dpi=220, bbox_inches="tight")
    print("wrote {}.[pdf|png]".format(args.out))


if __name__ == "__main__":
    main()
