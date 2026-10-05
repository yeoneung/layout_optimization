"""Paper figures for the EAAI version (matplotlib, print/light mode).

python figures_eaai.py <phase_dir> --out <figure dir> [--methods BANDIT:8,M0,M1,ALNS:small_beam]

F1  anytime_all.pdf     mean improvement vs budget, 4 sizes x 2 fills, <= 4 series
F2  regime_map.pdf      best fixed cap (4 / 8 / 16 / regeneration) by (n, fill, budget)
F3  arm_shares.pdf      time share of each bandit arm by elapsed time, per n

Colors follow the validated reference palette of the data-viz method: fixed
categorical order (blue, orange, aqua, yellow) for the compared methods and an
ordinal blue ramp for the repair caps. Seeds are averaged within instance and
the two geometry families weighted equally, as in every table.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
# Journals reject Type 3 fonts; embed TrueType outlines instead.
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]      # categorical slots 1-4 (light)
CAP_RAMP = {"4": "#86b6ef", "8": "#3987e5", "16": "#1c5cab", "M0": "#0d366b"}   # ordinal blue ramp
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
# ITOR text block is 456 pt; draw at that width so nothing is scaled down.
TEXTWIDTH_IN = 456.0 / 72.27
LABEL_PT = 8
TITLE_PT = 9

GEOMETRIES = ("guillotine", "nonslicing")
LABELS = {"M0": "full regeneration (M0)", "M1": "partial repair, cap 4 (M1)",
          "ALNS:small_beam": "ALNS control", "BANDIT:8": "cap bandit",
          "BANDIT2:8": "two-arm cap bandit", "INTERLEAVE": "interleave", "CAP:8": "cap 8", "CAP:16": "cap 16",
          "SW:fixed3": "switch at 3 s", "SWL:swl_v1": "learned switch", "BANDITW:priors_v1": "warm-started bandit",
          "BANDITP:8": "PORT5", "BANDITP3:8": "PORT3-B (proposed)"}


def load_runs(directory, keep_schedule=False, metric="improvement"):
    initial_cost = {}
    if metric == "cost":
        for path in (Path(directory) / "initializations").glob("*.json"):
            init = json.loads(path.read_text(encoding="utf-8"))
            if init.get("initial_cost"):
                initial_cost[(init["geometry"], init["n"], init["fill"], init["instance"])] = init["initial_cost"]
    runs = []
    for path in sorted((Path(directory) / "runs").glob("*.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        key = (raw["geometry"], raw["n"], raw["fill"], raw["instance"])
        if metric == "cost":
            if key not in initial_cost:
                continue
            imp = {r["budget_s"]: 100.0 * r["improvement"] / initial_cost[key] for r in raw["rows"]}
        else:
            imp = {r["budget_s"]: r["improvement_pct"] for r in raw["rows"]}
        run = {"method": raw["method"], "geometry": raw["geometry"], "n": raw["n"], "fill": raw["fill"],
               "instance": raw["instance"], "seed": raw["evaluation_seed"], "imp": imp}
        if keep_schedule:
            run["schedule"] = raw["stats"].get("schedule")
        runs.append(run)
    return runs


def cell_mean(runs, method, n, fill, budget):
    per_instance = defaultdict(list)
    for r in runs:
        if r["method"] == method and r["n"] == n and r["fill"] == fill:
            per_instance[(r["geometry"], r["instance"])].append(r["imp"][budget])
    by_geometry = defaultdict(list)
    for (g, _), values in per_instance.items():
        by_geometry[g].append(np.mean(values))
    present = [g for g in GEOMETRIES if by_geometry[g]]
    return float(np.mean([np.mean(by_geometry[g]) for g in present])) if present else np.nan


def style_axes(ax):
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=LABEL_PT, length=2)
    ax.grid(True, axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


YLABEL = {"improvement": "improvement (%)", "cost": "cost reduction (%)"}
_metric = {"name": "improvement"}


def figure_anytime(runs, methods, budgets, out):
    sizes = sorted({r["n"] for r in runs})
    fills = sorted({r["fill"] for r in runs})
    fig, axes = plt.subplots(len(fills), len(sizes), figsize=(TEXTWIDTH_IN, 1.75 * len(fills) + 0.7),
                             sharex=True)
    axes = np.atleast_2d(axes)
    xs = [b for b in budgets if b > 0]
    for i, fill in enumerate(fills):
        for j, n in enumerate(sizes):
            ax = axes[i, j]
            style_axes(ax)
            for k, method in enumerate(methods):
                ys = [cell_mean(runs, method, n, fill, b) for b in xs]
                if np.all(np.isnan(ys)):
                    continue
                ax.plot(xs, ys, color=SERIES_COLORS[k], linewidth=1.6, marker="o", markersize=2.8,
                        markeredgecolor="white", markeredgewidth=0.6, label=LABELS.get(method, method), zorder=3)
            ax.set_xscale("log")
            ax.set_xticks([0.1, 1, 10, 60])
            ax.set_xticklabels(["0.1", "1", "10", "60"])
            ax.set_title("n = {}, fill {:.2f}".format(n, fill), fontsize=TITLE_PT, color=INK, pad=3)
            if j == 0:
                ax.set_ylabel(YLABEL[_metric["name"]], fontsize=LABEL_PT, color=INK2)
            if i == len(fills) - 1:
                ax.set_xlabel("seconds after initialization", fontsize=LABEL_PT, color=INK2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(methods), fontsize=LABEL_PT, frameon=False,
               bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out / "anytime_all.pdf")
    fig.savefig(out / "anytime_all.png", dpi=200)
    plt.close(fig)


def figure_regime(runs, budgets, out):
    """Best fixed cap per (n, fill, budget) among M1 (cap 4), CAP:8, CAP:16, M0."""
    arms = [("M1", "4"), ("CAP:8", "8"), ("CAP:16", "16"), ("M0", "M0")]
    available = [a for a in arms if any(r["method"] == a[0] for r in runs)]
    if len(available) < 2:
        return
    sizes = sorted({r["n"] for r in runs})
    fills = sorted({r["fill"] for r in runs})
    xs = [b for b in budgets if b > 0]
    rows = [(n, fill) for fill in fills for n in sizes]
    fig, ax = plt.subplots(figsize=(TEXTWIDTH_IN, 0.34 * len(rows) + 1.0))
    for i, (n, fill) in enumerate(rows):
        for j, b in enumerate(xs):
            values = {label: cell_mean(runs, m, n, fill, b) for m, label in available}
            best = max(values, key=lambda k: values[k])
            second = sorted(values.values())[-2]
            margin = values[best] - second
            ax.add_patch(plt.Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, color=CAP_RAMP[best], linewidth=0))
            text = "∞" if best == "M0" else best
            ax.text(j + 0.5, i + 0.5, text, ha="center", va="center", fontsize=LABEL_PT,
                    color="white" if best in ("16", "M0") else INK)
            if margin < 0.5:
                ax.add_patch(plt.Rectangle((j + 0.05, i + 0.05), 0.9, 0.9, fill=False, edgecolor=INK2,
                                           linewidth=0.6, linestyle=(0, (1.5, 1.5))))
    ax.set_xlim(0, len(xs))
    ax.set_ylim(len(rows), 0)
    ax.set_xticks([j + 0.5 for j in range(len(xs))])
    ax.set_xticklabels(["{:g}".format(b) for b in xs], fontsize=LABEL_PT, color=INK2)
    ax.set_yticks([i + 0.5 for i in range(len(rows))])
    ax.set_yticklabels(["n = {}, fill {:.2f}".format(n, fill) for n, fill in rows], fontsize=LABEL_PT, color=INK2)
    ax.set_xlabel("seconds after initialization", fontsize=LABEL_PT, color=INK2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=CAP_RAMP[k]) for k in ("4", "8", "16", "M0")]
    fig.legend(handles, ["cap 4", "cap 8", "cap 16", "regeneration"], loc="lower center", ncol=4,
               fontsize=LABEL_PT, frameon=False, bbox_to_anchor=(0.5, -0.02))
    ax.set_title("best fixed repair scope (dotted: margin below 0.5 pp)", fontsize=TITLE_PT, color=INK)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out / "regime_map.pdf")
    fig.savefig(out / "regime_map.png", dpi=200)
    plt.close(fig)


def figure_arm_shares(runs, method, out, grid=np.array([1, 2, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60])):
    """Share of search time spent on each arm within [0, t], averaged over runs, per n."""
    sel = [r for r in runs if r["method"] == method and r.get("schedule")]
    if not sel:
        return
    sizes = sorted({r["n"] for r in sel})
    played = {(s["mode"], s["cap"]) for r in sel for s in r["schedule"]}
    order = [a for a in [("M1", 4), ("M1", 8), ("M1", 16), ("M0", None), ("ALNS", None)] if a in played]
    names = [{("M1", 4): "4", ("M1", 8): "8", ("M1", 16): "16", ("M0", None): "M0", ("ALNS", None): "ALNS"}[a] for a in order]
    colors = {"4": CAP_RAMP["4"], "8": CAP_RAMP["8"], "16": CAP_RAMP["16"], "M0": CAP_RAMP["M0"], "ALNS": SERIES_COLORS[1]}
    legend = {"4": "cap 4", "8": "cap 8", "16": "cap 16", "M0": "regeneration", "ALNS": "destroy-and-repair round"}
    fig, axes = plt.subplots(1, len(sizes), figsize=(TEXTWIDTH_IN, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, n in zip(axes, sizes):
        style_axes(ax)
        shares = np.zeros((len(order), len(grid)))
        count = 0
        for r in sel:
            if r["n"] != n:
                continue
            count += 1
            for gi, t in enumerate(grid):
                spent = np.zeros(len(order))
                for s in r["schedule"]:
                    start, end = s["start_s"], s["start_s"] + s["elapsed_s"]
                    overlap = max(0.0, min(end, t) - start)
                    if overlap > 0:
                        spent[order.index((s["mode"], s["cap"]))] += overlap
                total = spent.sum()
                if total > 0:
                    shares[:, gi] += spent / total
        shares /= max(count, 1)
        ax.stackplot(grid, shares, colors=[colors[k] for k in names], labels=[legend[k] for k in names],
                     edgecolor="white", linewidth=0.8)
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30, 60])
        ax.set_xticklabels(["1", "3", "10", "30", "60"])
        ax.set_ylim(0, 1)
        ax.set_title("n = {} ({} runs)".format(n, count), fontsize=TITLE_PT, color=INK, pad=3)
        ax.set_xlabel("seconds", fontsize=LABEL_PT, color=INK2)
    axes[0].set_ylabel("share of search time", fontsize=LABEL_PT, color=INK2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(names), fontsize=LABEL_PT, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(out / "arm_shares.pdf")
    fig.savefig(out / "arm_shares.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--methods", default="BANDIT:8,M0,M1,ALNS:small_beam")
    parser.add_argument("--bandit", default="BANDIT:8")
    parser.add_argument("--metric", choices=["improvement", "cost"], default="improvement")
    parser.add_argument("--extra-dirs", dest="extra_dirs", default="",
                        help="comma-separated phase directories with the same instances whose runs are pooled in")
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((args.directory / "protocol.json").read_text(encoding="utf-8"))["settings"]
    budgets = protocol["budgets_s"]
    runs = load_runs(args.directory, keep_schedule=True, metric=args.metric)
    for extra in [Path(p) for p in args.extra_dirs.split(",") if p.strip()]:
        runs.extend(load_runs(extra, keep_schedule=True, metric=args.metric))
    _metric["name"] = args.metric
    methods = [m for m in args.methods.split(",") if any(r["method"] == m for r in runs)][:4]
    figure_anytime(runs, methods, budgets, out)
    figure_regime(runs, budgets, out)
    figure_arm_shares(runs, args.bandit, out)
    print("figures written to", out, "series:", methods)


if __name__ == "__main__":
    main()
