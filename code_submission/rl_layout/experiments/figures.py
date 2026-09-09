"""
Paper figures.

Colours come from the data-viz reference palette, used verbatim and in its
documented slot order (that ordering is the colour-vision-deficiency safety
mechanism, not decoration).  Scatter figures compare every pair of series at
once, so they are capped at the first three slots, which are the ones validated
under the all-pairs rule; line and bar figures only need adjacent pairs to
separate and may use more.  Three of the light-mode slots sit below 3:1 against
the surface, so every series is also labelled directly -- colour never carries
identity on its own.
"""

import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
from matplotlib.patches import Rectangle                       # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# reference palette, light mode, documented order
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7", "#e34948"]
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURF = "#fcfcfb"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans"],
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.facecolor": SURF,
    "figure.facecolor": SURF,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "text.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "lines.linewidth": 2.0,
})

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "..", "figs")


def load(path):
    if not os.path.exists(path):
        print(f"  (skip, missing: {path})")
        return None
    with open(path) as f:
        return json.load(f)


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.pdf")


def pick(rows, pred):
    return sorted([r for r in rows if pred(r)], key=lambda r: r["evals_per_inst"])


# ---------------------------------------------------------------------------
def fig_anytime(*sources, name="fig_anytime"):
    """Objective against the budget each method spent -- the comparison the
    literature usually replaces with 'steps', which is not comparable."""
    rows = [r for d in sources if d for r in d["rows"]]
    fig, ax = plt.subplots(figsize=(6.6, 4.2))

    def is_sa_sweep(r, kind):
        m = r["method"]
        return (m.startswith("SA ") and kind in m and "from" not in m
                and "restart" not in m)

    def is_classical_constructive(r):
        m = r["method"]
        return m.startswith("shelf packing") or m.startswith("greedy position") \
            or m.startswith("greedy room") or m.startswith("greedy + jitter") \
            or m.startswith("greedy constructive")

    series = [
        ("SA, continuous coordinates", lambda r: is_sa_sweep(r, "continuous"),
         C[1], "-o", 5),
        ("SA, lattice", lambda r: is_sa_sweep(r, "lattice"), C[3], "-s", 5),
        ("classical constructive,\nsame masked action set",
         is_classical_constructive, C[2], "-^", 6),
        ("RL, constructive", lambda r: r["method"].startswith("RL-construct")
         and "polish" not in r["method"], C[0], "-o", 8),
    ]
    for label, pred, col, style, ms in series:
        s = pick(rows, pred)
        if not s:
            continue
        x = [r["evals_per_inst"] for r in s]
        y = [r["mean"] for r in s]
        ax.plot(x, y, style, color=col, markersize=ms, label=label.replace("\n", " "),
                zorder=4 if col == C[0] else 3, markeredgecolor=SURF,
                markeredgewidth=1.3)
        best = max(s, key=lambda r: r["mean"])
        ax.annotate(label, (best["evals_per_inst"], best["mean"]),
                    textcoords="offset points", xytext=(8, 5), color=col,
                    fontsize=8, fontweight="bold")

    s = pick(rows, lambda r: r["method"].startswith("RL-improve"))
    if s:
        ax.plot([r["evals_per_inst"] for r in s], [r["mean"] for r in s], "-d",
                color=C[5], markersize=5, zorder=3, markeredgecolor=SURF,
                markeredgewidth=1.2, label="RL, improvement")

    ax.set_xscale("log")
    ax.set_xlabel("objective evaluations per layout  (log scale)")
    ax.set_ylabel("objective  $J$")
    ax.set_title("What each method buys with its search budget", loc="left",
                 fontsize=10, fontweight="bold", pad=10)
    ax.legend(loc="lower right", fontsize=7.5)
    ax.grid(axis="both", alpha=0.7)
    save(fig, name)


def fig_ablation(entries, name="fig_ablation"):
    """One change per row: the ablation *is* the argument, so it is a single
    series and takes a single hue.  `entries` is [(label, J, note), ...]."""
    labels = [e[0] for e in entries]
    vals = [e[1] for e in entries]
    notes = [e[2] if len(e) > 2 else "" for e in entries]
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.2, 0.46 * len(labels) + 1.6))
    ypos = np.arange(len(labels))[::-1]
    ax.barh(ypos, vals, height=0.6, color=C[0], zorder=3)
    for y, v, nt in zip(ypos, vals, notes):
        ax.text(v + max(vals) * 0.012, y, f"{v:.0f}", va="center", fontsize=8.5,
                color=INK, fontweight="bold")
        if nt:
            ax.text(max(vals) * 0.015, y, nt, va="center", fontsize=7.2, color=SURF)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8.2, color=INK2)
    ax.set_xlabel("objective  $J$")
    ax.set_xlim(0, max(vals) * 1.12)
    ax.grid(axis="y", visible=False)
    if len(labels) > 5:
        # the ablation proper is the top block; the rest are reference points
        ax.axhline(len(labels) - 5.5, color=AXIS, linewidth=1, linestyle="--", zorder=2)
        ax.text(max(vals) * 0.985, len(labels) - 5.4, "improvement MDP: one change per row",
                ha="right", va="bottom", fontsize=7.5, color=MUTED)
        ax.text(max(vals) * 0.985, len(labels) - 5.6, "reference points",
                ha="right", va="top", fontsize=7.5, color=MUTED)
    ax.set_title("Fixing the improvement MDP helps, but not enough to change the ranking",
                 loc="left", fontsize=10, fontweight="bold", pad=10)
    save(fig, name)


def fig_qd(qd, name="fig_qd"):
    """Quality against variety.  A scatter compares all pairs at once, so this
    is capped at the three slots validated under the all-pairs rule."""
    if qd is None:
        return
    rows = qd["rows"]
    # Three colour-coded series only: a scatter puts every pair on screen at once
    # and the first three palette slots are the ones validated under that rule.
    # Shelf packing is a single reference point, drawn in muted ink so it does
    # not consume a series slot.
    # The third series is the training-free constructor given the *same*
    # diversity mechanism as the policy.  It replaced the continuous-coordinate
    # annealer here, which moved to muted reference ink: with three colour slots
    # available under the all-pairs rule, they belong to the three methods the
    # section's conclusion is actually partitioned between.
    groups = [
        ("RL, constructive\nrandom commit order",
         lambda r: r.get("order") == "random", C[0], "o", (6, 8), "left"),
        ("SA, lattice (45 000 evals)",
         lambda r: r["method"].startswith("SA ") and "lattice" in r["method"],
         C[1], "s", (-14, -30), "right"),
        ("greedy constructive,\nno learning",
         lambda r: r["method"].startswith("greedy"),
         C[2], "^", (-4, 12), "right"),
    ]

    def front(s):
        """Upper-right staircase: the points of `s` no other point of `s`
        dominates in both diversity and quality.  Each series here mixes more
        than one way of buying variety (commit order, jitter, budget), so
        connecting the raw sweep would draw a line that doubles back and read as
        noise rather than as a front."""
        out, best = [], -np.inf
        for r in sorted(s, key=lambda r: -r["diversity"]):
            if r["quality_mean"] > best:
                out.append(r)
                best = r["quality_mean"]
        return out[::-1]

    fig, ax = plt.subplots(figsize=(6.8, 4.6))
    for label, pred, col, mk, off, ha in groups:
        s = [r for r in rows if pred(r)]
        if not s:
            continue
        f = front(s)
        dominated = [r for r in s if r not in f]
        if dominated:                       # shown, but not part of the front
            ax.plot([r["diversity"] for r in dominated],
                    [r["quality_mean"] for r in dominated], mk, color=col,
                    markersize=5, linestyle="none", alpha=0.35, zorder=2)
        ax.plot([r["diversity"] for r in f], [r["quality_mean"] for r in f],
                mk, color=col, markersize=7, linestyle="--" if len(f) > 1 else "none",
                linewidth=1.4, markeredgecolor=SURF, markeredgewidth=1.2,
                label=label.replace("\n", " "), zorder=3)
        anchor = max(f, key=lambda r: r["quality_mean"])
        ax.annotate(label, (anchor["diversity"], anchor["quality_mean"]),
                    textcoords="offset points", xytext=off, color=col,
                    fontsize=8, fontweight="bold", ha=ha)
    # reference points drawn in muted ink: they are context, not series
    for pred, label, dy in (
        (lambda r: r.get("order") == "policy", "same policy, but the order is\n"
         "its own decision and the entropy\nbonus was annealed away", -30),
        (lambda r: r["method"].startswith("SA ") and "lattice" not in r["method"],
         "SA, continuous coords", -12),
        (lambda r: r["method"].startswith("shelf"), "shelf packing", -12)):
        s = [r for r in rows if pred(r)]
        if not s:
            continue
        ax.plot([r["diversity"] for r in s], [r["quality_mean"] for r in s], "x",
                color=MUTED, markersize=7, linestyle="none", zorder=3)
        b = max(s, key=lambda r: r["quality_mean"])
        ax.annotate(label, (b["diversity"], b["quality_mean"]),
                    textcoords="offset points", xytext=(10, dy), color=INK2,
                    fontsize=7.5)
    ax.set_xlabel("diversity  (mean pairwise fraction of cells with a different room type)")
    ax.set_ylabel("objective  $J$")
    ax.set_title("Quality and diversity for the three methods",
                 loc="left", fontsize=10, fontweight="bold", pad=10)
    ax.legend(loc="lower left", fontsize=7.5)
    save(fig, name)


def fig_density(dens, name="fig_density"):
    """The regime result: what learning is worth as the plate fills up.

    Two panels, because the two things that happen at high density are different
    in kind.  Left: the fraction of instances on which a method produces any
    overlap-free layout at all -- a feasibility cliff, not a quality trend.
    Right: solution quality among the methods still standing, as a paired gap to
    the per-instance best.  Three colour-coded series only, the slots validated
    under the all-pairs rule; annealing is drawn in muted ink as context.
    """
    if dens is None:
        return
    rows = dens["records"]
    fills = sorted({r["fill"] for r in rows})

    # The policy predicate must require the x16 sampling variant: the
    # deterministic single decode is a different operating point and averaging
    # the two understates the sampled policy it is being compared against.
    series = [
        ("learned policy\n(chooses its own order)",
         lambda m: m.startswith("policy own-order") and m.endswith("x16"),
         C[0], "o", "--", 4),
        ("largest first order\n+ position beam (no learning)",
         lambda m: m == "beam, area order, width 16", C[2], "^", "-", 3),
        ("free order beam\n(no learning)",
         lambda m: m == "beam width 16", C[1], "s", "-", 3),
    ]
    ref = (lambda m: m == "SA 45000 lattice", "annealing")

    # per-instance best over every method, for the normalized gap
    best = {}
    for r in rows:
        k = (r["fill"], r["instance"])
        best[k] = max(best.get(k, -np.inf), r["J"])

    def curves(pred):
        feas, gap = [], []
        for f in fills:
            sel = [r for r in rows if r["fill"] == f and pred(r["method"])]
            if not sel:
                feas.append(np.nan)
                gap.append(np.nan)
                continue
            feas.append(np.mean([r["feasible"] for r in sel]))
            g = [100.0 * (best[(f, r["instance"])] - r["J"])
                 / max(abs(best[(f, r["instance"])]), 1e-9) for r in sel]
            gap.append(np.mean(g))
        return np.array(feas), np.array(gap)

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.9))
    for ax, idx, ylab, title in (
            (axes[0], 0, "fraction of instances solved feasibly",
             "Feasibility collapses, and not at the same point"),
            (axes[1], 1, "mean paired gap to best for each instance  (%)",
             "Quality among what survives")):
        for label, pred, col, mk, ls, zo in series:
            v = curves(pred)[idx]
            # the policy is dashed and on top: its feasibility curve coincides
            # with the largest-first one up to fill 0.90 and would be hidden
            ax.plot(fills, v, marker=mk, linestyle=ls, color=col, markersize=7,
                    markeredgecolor=SURF, markeredgewidth=1.2,
                    label=label.replace("\n", " "), zorder=zo)
        v = curves(ref[0])[idx]
        ax.plot(fills, v, "x--", color=MUTED, markersize=7, linewidth=1.2,
                zorder=2, label=ref[1])
        ax.set_xlabel("fill ratio  (occupied fraction of the plate)")
        ax.set_ylabel(ylab)
        ax.set_title(title, loc="left", fontsize=9.5, fontweight="bold", pad=8)
        ax.set_xticks(fills)
    axes[0].set_ylim(-0.04, 1.06)
    axes[1].legend(loc="upper left", fontsize=7.5)

    # direct labels at the right edge of the left panel, where the curves have
    # separated; anchoring at each series' own endpoint spaces them vertically
    for label, pred, col, _, _, _ in series:
        v = curves(pred)[0]
        axes[0].annotate(label, (fills[-1], v[-1]), textcoords="offset points",
                         xytext=(-10, 12), ha="right", color=col, fontsize=7.5,
                         fontweight="bold")
    save(fig, name)


def fig_transfer(tr, name="fig_transfer"):
    """Three series, grouped bars -- first three slots, all-pairs validated."""
    if tr is None:
        return
    rows = tr["rows"]
    sets = []
    for r in rows:
        if r["test_set"] not in sets:
            sets.append(r["test_set"])
    # Shelf packing was dropped as a bar: it is dominated everywhere and the slot
    # is better spent on the training-free constructor, which is the row the
    # amortization claim has to survive.
    want = [("greedy constructive (no training)",
             lambda r: r["method"] == "greedy constructive (room+position)", C[2]),
            ("SA 45000, from scratch",
             lambda r: r["method"].startswith("SA 45000"), C[1]),
            ("RL zero shot (x16)",
             lambda r: r["method"] == "RL-construct zero-shot x16", C[0])]
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    w = 0.26
    xs = np.arange(len(sets))
    for k, (label, pred, col) in enumerate(want):
        vals, evs = [], []
        for s in sets:
            m = [r for r in rows if r["test_set"] == s and pred(r)]
            vals.append(m[0]["mean"] if m else np.nan)
            evs.append(m[0]["evals_per_inst"] if m else np.nan)
        pos = xs + (k - 1) * w
        ax.bar(pos, vals, width=w * 0.88, color=col, label=label, zorder=3)
        for p, v, e in zip(pos, vals, evs):
            if np.isfinite(v):
                ax.text(p, v * 1.008, f"{v:.0f}", ha="center", va="bottom",
                        fontsize=8, color=INK, fontweight="bold")
                ax.text(p, v * 0.965, f"{e:,.0f}\nevals", ha="center", va="top",
                        fontsize=6.6, color="white")
    # the manuscript calls the two legacy hand-authored scenarios OFFICE and CLINIC
    pretty = {"comb_high": "office", "hospital": "clinic"}
    display_sets = [s.replace("in-distribution", "in distribution") for s in sets]
    ax.set_xticks(xs)
    ax.set_xticklabels([pretty.get(s.split(" (")[0], s.split(" (")[0])
                        + "\n(" + s.split(" (")[1] for s in display_sets], fontsize=8.5)
    ax.set_ylabel("objective  $J$")
    ax.grid(axis="x", visible=False)
    ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10),
              columnspacing=1.4)
    ax.set_title("Held out briefs. The policy did not optimize these instances",
                 loc="left", fontsize=10.5, fontweight="bold", pad=34)
    save(fig, name)


def fig_amortize(train_s, rl_ms, rl_J, sas, name="fig_amortize"):
    """Total wall-clock to answer m requests for the same brief.

    Training is charged in full and up front.  `sas` is a list of
    (label, seconds per layout, J) for the annealing configurations; the point of
    plotting more than one is that the annealer only closes the quality gap by
    buying budget, and buying budget is what moves its line up.
    """
    m = np.logspace(0, 4, 300)
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    rl = train_s + m * rl_ms / 1000.0
    for k, (label, s_per, J) in enumerate(sas):
        col = C[1 + k]
        ax.plot(m, m * s_per, color=col, zorder=3,
                label=f"{label}  ($J$={J:.0f})")
        ax.annotate(f"{label}\n$J$={J:.0f}", (m[-1], m[-1] * s_per),
                    textcoords="offset points", xytext=(-6, 2), ha="right",
                    color=col, fontsize=8, fontweight="bold")
        cross = train_s / max(s_per - rl_ms / 1000.0, 1e-9)
        if 1 < cross < 1e4:
            ax.plot([cross], [train_s + cross * rl_ms / 1000.0], "o", color=col,
                    markersize=6, markeredgecolor=SURF, markeredgewidth=1.2, zorder=5)
            ax.annotate(f"pays for itself\nafter {cross:.0f} layouts",
                        (cross, train_s + cross * rl_ms / 1000.0),
                        textcoords="offset points", xytext=(6, -22), color=INK2,
                        fontsize=7.5)
    ax.plot(m, rl, color=C[0], zorder=4,
            label=f"RL, constructive: {train_s:.0f} s training + {rl_ms:.0f} ms each"
                  f"  ($J$={rl_J:.0f})")
    ax.annotate(f"RL, constructive\n$J$={rl_J:.0f}", (m[-1], rl[-1]),
                textcoords="offset points", xytext=(-6, 4), ha="right", color=C[0],
                fontsize=8, fontweight="bold")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("layouts requested for the same brief  (log scale)")
    ax.set_ylabel("total wall-clock, seconds  (training included)")
    ax.set_title("Training is paid once; searching is paid per layout",
                 loc="left", fontsize=10, fontweight="bold", pad=10)
    ax.legend(loc="upper left", fontsize=7.5)
    save(fig, name)


# ---------------------------------------------------------------------------
TYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7",
               "#e34948", "#86b6ef", "#008300", "#c98500", "#9085e9", "#d55181",
               "#1c5cab", "#199e70"]


def draw_layout(ax, inst, x, y, b, title, names=None):
    gw, gh = inst.gw[b], inst.gh[b]
    ax.add_patch(Rectangle((0, 0), gw, gh, fill=False, edgecolor=AXIS, lw=1.4))
    for i in range(inst.n):
        w, h = inst.w[b, i], inst.h[b, i]
        col = TYPE_COLORS[int(inst.type_id[b, i]) % len(TYPE_COLORS)]
        ax.add_patch(Rectangle((x[b, i] - w / 2, y[b, i] - h / 2), w, h,
                               facecolor=col, edgecolor=SURF, lw=1.2, alpha=0.92))
        if names is not None and w * h >= 8:
            ax.text(x[b, i], y[b, i], names[i], ha="center", va="center",
                    fontsize=6.4, color="white")
    ax.set_xlim(-0.6, gw + 0.6)
    ax.set_ylim(-0.6, gh + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8.5, color=INK2, pad=4)


ABLATION_ROWS = [
    # (label shown, substring identifying the row)
    ("legacy formulation: $\\Delta J$ reward,\ntranslate only, $\\gamma$=0.995",
     "A0_legacy_delta_g995 H4096 stoch"),
    ("$\\gamma$ = 0 (legacy's own best discount)", "A0_legacy_delta_g0 H4096 stoch"),
    ("best-so-far reward, $\\gamma$ = 1", "A1_best_translate H4096 stoch"),
    ("+ jump and swap moves", "A2_best_allops H4096 stoch"),
    ("+ Metropolis accept/reject", "C1_neural_sa H4096 stoch"),
    ("shelf packing (no search)", "shelf packing (no search)"),
    ("strongest annealing (SA 45 000, lattice)", "SA 45000 lattice + polish"),
    ("masked greedy constructive (no learning)", "greedy room + position"),
    ("constructive policy (learned)", "RL-construct B1_constr_fixed greedy"),
]


def collect(sources, needle):
    for d in sources:
        if not d:
            continue
        for r in d["rows"]:
            if needle in r["method"]:
                return r
    return None


def main():
    print("figures ->", os.path.abspath(OUT))
    main_json = load("compare_comb_high.json")
    base_json = load("baselines_comb_high.json")
    cbase_json = load("constr_baselines_comb_high.json")
    improve_json = load("compare_improve.json")
    srcs = [main_json, base_json, cbase_json, improve_json]

    fig_anytime(main_json, cbase_json, base_json)

    qd = load("qd_comb_high.json")
    if qd:
        merged = {"rows": list(qd["rows"])}
        for extra in ("qd_policies.json", "qd_policies_hiT.json",
                      "qd_greedy_sweep.json"):
            d = load(extra)
            if d:
                merged["rows"] += d["rows"]
        fig_qd(merged)
    fig_transfer(load("transfer.json"))
    fig_density(load("bench_density.json"))

    entries = []
    for label, needle in ABLATION_ROWS:
        r = collect(srcs, needle)
        if r:
            entries.append((label, r["mean"],
                            f"{r['feasible_rate']*100:.0f}% feasible"))
    if entries:
        fig_ablation(entries)

    scal = load("sa_scaling.json")
    hist_path = "runs/B1_constr_fixed_history.json"
    rl = collect(srcs, "RL-construct B1_constr_fixed greedy")
    if scal and rl and os.path.exists(hist_path):
        with open(hist_path) as f:
            hist = json.load(f)
        train_s = hist[-1]["elapsed"]
        sa45 = collect(srcs, "SA 45000 lattice")
        big = max(scal["rows"], key=lambda r: r["mean"])
        sas = []
        if sa45:
            sas.append(("SA 45 000, lattice", sa45["ms_per_layout"] / 1000.0,
                        sa45["mean"]))
        sas.append((f"SA {big['method'].split()[1]}, lattice",
                    big["ms_per_layout"] / 1000.0, big["mean"]))
        fig_amortize(train_s, rl["ms_per_layout"], rl["mean"], sas)


if __name__ == "__main__":
    main()
