"""LaTeX tables for the EAAI manuscript, generated from the analysis CSVs so that
no number is typed by hand.

python make_tables_eaai.py <report dir with quality.csv/contrasts.csv/oracle.csv> --out <tex dir>

Writes:
  tab_primary.tex    BANDIT:8 minus M0 / M1 / ALNS at 10 and 60 s per cell, with intervals
  tab_quality.tex    mean improvement of every method per cell at 10 and 60 s
  tab_oracle.tex     gap to the oracle best-of(M0, M1) and to the all-method oracle, mean rank
  tab_summary.tex    wins / ties / losses vs M0 and M1, mean rank, oracle gap, per budget
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

SHORT = {"M0": "M0", "M1": "M1", "ALNS:small_beam": "ALNS", "CAP:8": "CAP8", "CAP:16": "CAP16",
         "SW:fixed3": "SW3", "INTERLEAVE": "INTL", "BANDIT:8": "CAPB", "BANDIT2:8": "CAPB2",
         "SWL:swl_v1": "SWL", "BANDITW:priors_v1": "CAPBW", "BANDITP:8": "PORT5",
         "BANDITP3:8": "PORT3-B", "ALNS:alns_c8_b4_t25": "ALNSR",
         "RR3": "PORT3-R", "RND3": "PORT3-U"}
PRIMARY = "BANDITP3:8"


def read(path):
    return list(csv.DictReader(open(path, encoding="utf-8")))


def fmt(x, signed=False, digits=2):
    return ("{:+." + str(digits) + "f}").format(x) if signed else ("{:." + str(digits) + "f}").format(x)


DEFAULT_CONTROLS = ("M0", "M1", "ALNS:small_beam", "ALNS:alns_c8_b4_t25", "BANDIT:8",
                    "RR3", "RND3", "BANDITP3:8")


def table_primary(contrasts, budgets, out, control_order=None):
    """One table per budget: the primary method minus each control with its interval."""
    order = control_order or DEFAULT_CONTROLS
    controls = [c for c in order
                if any(r["control"] == c for r in contrasts) and c != PRIMARY]
    cells = sorted({(int(r["n"]), float(r["fill"])) for r in contrasts})
    for b in budgets:
        lines = [r"\begin{tabular}{@{}rr" + "c" * len(controls) + "@{}}", r"\toprule",
                 r"$n$ & fill & " + " & ".join("minus " + SHORT[c] for c in controls) + r" \\", r"\midrule"]
        for k, (n, fill) in enumerate(cells):
            if k == len(cells) // 2:
                lines.append(r"\midrule")
            entries = []
            for c in controls:
                r = next(x for x in contrasts if int(x["n"]) == n and float(x["fill"]) == fill
                         and x["method"] == PRIMARY and x["control"] == c and float(x["budget_s"]) == b)
                d, lo, hi = float(r["diff_pp"]), float(r["ci_low"]), float(r["ci_high"])
                mark = r"$^{+}$" if lo > 0 else (r"$^{-}$" if hi < 0 else "")
                entries.append("$%s\\;[%s,%s]$%s" % (fmt(d, True), fmt(lo, True), fmt(hi, True), mark))
            lines.append("%d & %.2f & " % (n, fill) + " & ".join(entries) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / ("tab_primary_%gs.tex" % b)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_quality(quality, methods, budgets, out):
    cells = sorted({(int(r["n"]), float(r["fill"])) for r in quality})
    for b in budgets:
        lines = [r"\begin{tabular}{@{}rrr" + "r" * len(methods) + "@{}}", r"\toprule",
                 r"$n$ & fill & inst. & " + " & ".join(SHORT[m] for m in methods) + r" \\", r"\midrule"]
        for k, (n, fill) in enumerate(cells):
            if k == len(cells) // 2:
                lines.append(r"\midrule")
            rows = {r["method"]: r for r in quality if int(r["n"]) == n and float(r["fill"]) == fill and float(r["budget_s"]) == b}
            vals = {m: float(rows[m]["improvement_pct"]) for m in methods}
            best = max(vals.values())
            entries = [(r"\textbf{%s}" % fmt(v)) if abs(v - best) < 1e-9 else fmt(v) for v in (vals[m] for m in methods)]
            lines.append("%d & %.2f & %s & " % (n, fill, rows[methods[0]]["instances"]) + " & ".join(entries) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / ("tab_quality_%gs.tex" % b)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_oracle(oracle, methods, budgets, out):
    cells = sorted({(int(r["n"]), float(r["fill"])) for r in oracle})
    for b in budgets:
        lines = [r"\begin{tabular}{@{}rr" + "r" * len(methods) + "@{}}", r"\toprule",
                 r"$n$ & fill & " + " & ".join(SHORT[m] for m in methods) + r" \\", r"\midrule"]
        for k, (n, fill) in enumerate(cells):
            if k == len(cells) // 2:
                lines.append(r"\midrule")
            rows = {r["method"]: r for r in oracle if int(r["n"]) == n and float(r["fill"]) == fill and float(r["budget_s"]) == b}
            vals = {m: float(rows[m]["gap_to_oracle_pp"]) for m in methods}
            best = min(vals.values())
            entries = [(r"\textbf{%s}" % fmt(v)) if abs(v - best) < 1e-9 else fmt(v) for v in (vals[m] for m in methods)]
            lines.append("%d & %.2f & " % (n, fill) + " & ".join(entries) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        (out / ("tab_oracle_%gs.tex" % b)).write_text("\n".join(lines) + "\n", encoding="utf-8")


def table_summary(quality, contrasts, oracle, methods, budgets, out):
    lines = [r"\begin{tabular}{@{}l" + "rrrr" * len(budgets) + "@{}}", r"\toprule",
             "method & " + " & ".join(r"\multicolumn{4}{c}{%g seconds}" % b for b in budgets) + r" \\"]
    cm = "".join(r"\cmidrule(lr){%d-%d}" % (2 + 4 * i, 5 + 4 * i) for i in range(len(budgets)))
    lines.append(cm)
    lines.append(" & " + " & ".join("vs M0 & vs M1 & rank & gap" for _ in budgets) + r" \\")
    lines.append(r"\midrule")
    for m in methods:
        entries = []
        for b in budgets:
            for control in ("M0", "M1"):
                if control == m:
                    entries.append("--")
                    continue
                w = t = l = 0
                for c in contrasts:
                    if c["method"] == m and c["control"] == control and float(c["budget_s"]) == b:
                        lo, hi = float(c["ci_low"]), float(c["ci_high"])
                        if lo > 0:
                            w += 1
                        elif hi < 0:
                            l += 1
                        else:
                            t += 1
                entries.append("%d/%d/%d" % (w, t, l))
            ranks = [float(q["mean_rank"]) for q in quality if q["method"] == m and float(q["budget_s"]) == b]
            gaps = [float(o["gap_to_oracle_all_methods_pp"]) for o in oracle if o["method"] == m and float(o["budget_s"]) == b]
            entries.append(fmt(np.mean(ranks)))
            entries.append(fmt(np.mean(gaps)))
        lines.append(SHORT[m] + " & " + " & ".join(entries) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (out / "tab_summary.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", default="10,60")
    parser.add_argument("--primary", default=PRIMARY)
    parser.add_argument("--controls", default="",
                        help="comma-separated control order for the primary table")
    args = parser.parse_args()
    globals()["PRIMARY"] = args.primary
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    budgets = [float(b) for b in args.budgets.split(",")]
    quality = read(args.report / "quality.csv")
    contrasts = read(args.report / "contrasts.csv")
    oracle = read(args.report / "oracle.csv")
    methods = []
    for r in quality:
        if r["method"] not in methods:
            methods.append(r["method"])
    order = tuple(c.strip() for c in args.controls.split(",") if c.strip()) or None
    table_primary(contrasts, budgets, out, order)
    table_quality(quality, methods, budgets, out)
    table_oracle(oracle, methods, budgets, out)
    table_summary(quality, contrasts, oracle, methods, budgets, out)
    print("tables written to", out, "methods:", [SHORT[m] for m in methods])


if __name__ == "__main__":
    main()
