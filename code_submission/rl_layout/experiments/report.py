"""
Turn the result JSONs into the tables the paper needs, in LaTeX (booktabs) and
markdown, so the manuscript never carries a hand-copied number.
"""

import json
import os
import sys


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def fmt(v, nd=1):
    return "--" if v is None else f"{v:,.{nd}f}"


def latex_table(caption, label, header, rows, align=None):
    align = align or ("l" + "r" * (len(header) - 1))
    out = ["\\begin{table}[t]", "\\centering",
           f"\\caption{{{caption}}}", f"\\label{{{label}}}",
           "\\resizebox{\\linewidth}{!}{%",
           f"\\begin{{tabular}}{{@{{}}{align}@{{}}}}", "\\toprule",
           " & ".join(header) + " \\\\", "\\midrule"]
    for r in rows:
        if r is None:
            out.append("\\midrule")
            continue
        out.append(" & ".join(str(c) for c in r) + " \\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table}", ""]
    return "\n".join(out)


def md_table(header, rows):
    out = ["| " + " | ".join(header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        if r is None:
            continue
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out) + "\n"


# ---------------------------------------------------------------------------
MAIN_ORDER = [
    ("random init", "random initialization"),
    ("greedy steepest ascent", "greedy steepest ascent"),
    ("tabu search", "tabu search"),
    ("GA (pop 64 x 700 gen)", "genetic algorithm"),
    ("memetic GA (GA + local search)", "memetic GA"),
    ("shelf packing (no search)", "shelf packing, random order (no search)"),
    ("shelf packing (area order)", "shelf packing, area order (no search)"),
    ("shelf + greedy", "shelf packing + greedy"),
    ("SA 45k + polish (legacy setup)", "SA 45\\,000 + polish \\emph{(legacy setup)}"),
    ("SA 4096 lattice + polish", "SA 4\\,096, lattice"),
    ("SA 15000 lattice + polish", "SA 15\\,000, lattice"),
    ("SA 8x5625 restarts (lattice)", "SA 8 restarts $\\times$ 5\\,625, lattice"),
    ("SA 45000 lattice + polish", "SA 45\\,000, lattice"),
    (None, None),
    ("greedy position, area order", "masked greedy constructive, area order"),
    ("greedy room + position", "masked greedy constructive, greedy order"),
    ("greedy + jitter, best of 16", "masked greedy constructive, best of 16"),
    ("greedy constructive best-of-17 + polish",
     "masked greedy constructive, best of 17 + polish"),
    ("SA 4096 lattice from greedy constructive",
     "SA 4\\,096 warm-started from the greedy constructive"),
]


def ms_per_layout(r, n_inst):
    """run_baselines reports a total wall time; the comparison scripts report a
    per-layout one.  Normalize so the column means the same thing everywhere."""
    if r.get("ms_per_layout") is not None:
        return r["ms_per_layout"]
    if r.get("wall_s"):
        return r["wall_s"] / n_inst * 1000.0
    return None


def main_table(*sources):
    rows = []
    seen = {}
    n_inst = 64
    for d in sources:
        if d:
            n_inst = d.get("instances", n_inst)
            for r in d["rows"]:
                seen.setdefault(r["method"], r)
    used = set()

    def row_of(label, r):
        return [label, fmt(r["mean"]), f"{r['feasible_rate']*100:.0f}",
                fmt(r["adj"], 0), fmt(r["edge"], 0), fmt(r["evals_per_inst"], 0),
                fmt(ms_per_layout(r, n_inst), 0)]

    for key, label in MAIN_ORDER:
        if key is None:
            rows.append(None)
            continue
        r = seen.get(key)
        if r:
            used.add(key)
            rows.append(row_of(label, r))
    rows.append(None)
    for m, r in sorted(seen.items(), key=lambda kv: kv[1]["evals_per_inst"]):
        if m in used or not m.startswith("RL-improve"):
            continue
        used.add(m)
        rows.append(row_of(m.replace("RL-improve ", "RL, improvement: "), r))
    rows.append(None)
    for m, r in sorted(seen.items(), key=lambda kv: kv[1]["evals_per_inst"]):
        if m in used:
            continue
        if m.startswith("RL-construct") or (m.startswith("SA ") and "from" in m):
            used.add(m)
            rows.append(row_of(m, r))
    header = ["method", "$J$", "feas.\\ \\%", "adj.", "wall",
              "evals/layout", "ms/layout"]
    return header, rows


def main():
    outdir = "report"
    os.makedirs(outdir, exist_ok=True)
    base = load("baselines_comb_high.json")
    cbase = load("constr_baselines_comb_high.json")
    main_j = load("compare_comb_high.json")
    qd = load("qd_comb_high.json")
    tr = load("transfer.json")

    tex, md = [], []

    if base or main_j or cbase:
        h, rows = main_table(base, cbase, main_j)
        tex.append(latex_table(
            "Every method optimizes the same objective on the same 32-room office "
            "scenario. Feasibility is the fraction of layouts with exactly zero "
            "overlap; evaluations per layout is the search budget consumed.",
            "tab:main", h, rows))
        md.append("## Main comparison (comb_high)\n\n" + md_table(h, rows))

    if qd:
        h = ["method", "$J$", "diversity", "feas.\\ \\%", "ms/sample"]
        rows = [[r["method"], fmt(r["quality_mean"]), f"{r['diversity']:.3f}",
                 f"{r['feasible_rate']*100:.0f}", fmt(r["ms_per_sample"], 0)]
                for r in sorted(qd["rows"], key=lambda r: -r["quality_mean"])]
        tex.append(latex_table(
            "Quality and variety, both swept: the policy over its sampling "
            "temperature, the annealer over its budget.",
            "tab:qd", h, rows))
        md.append("## Quality-diversity\n\n" + md_table(h, rows))

    if tr:
        h = ["test set", "method", "$J$", "feas.\\ \\%", "evals/layout", "ms/layout"]
        rows = [[r["test_set"], r["method"], fmt(r["mean"]),
                 f"{r['feasible_rate']*100:.0f}", fmt(r["evals_per_inst"], 0),
                 fmt(r["ms_per_layout"], 0)] for r in tr["rows"]]
        tex.append(latex_table(
            "Held-out briefs. The policy was trained on procedurally generated "
            "instances and has never optimized any scenario in this table.",
            "tab:transfer", h, rows))
        md.append("## Transfer to unseen briefs\n\n" + md_table(h, rows))

    with open(os.path.join(outdir, "tables.tex"), "w") as f:
        f.write("\n".join(tex))
    with open(os.path.join(outdir, "tables.md"), "w") as f:
        f.write("\n".join(md))
    print(f"wrote {outdir}/tables.tex and {outdir}/tables.md")
    print("\n".join(md))


if __name__ == "__main__":
    main()
