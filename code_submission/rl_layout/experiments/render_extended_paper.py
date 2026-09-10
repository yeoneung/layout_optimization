"""Render audited study figures; print the LaTeX table source as JSON.

No experiment files are changed. Run from this directory with --paper pointing
to the manuscript directory. The caller saves the returned ``tex`` field as
extended_results.tex using its normal source-editing workflow.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


METHODS = ("M0", "M1", "ALNS:small_beam")
SIZES = (32, 64, 128, 256)
FILLS = (0.90, 0.95)
GEOMETRIES = ("guillotine", "nonslicing")
BUDGETS = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0)


def table(command, caption, label, columns, header, rows, size="small", spacing=3):
    lines = ["\\newcommand{\\" + command + r"}{%", r"\begin{table}[htbp]",
             r"\centering", r"\caption{" + caption + "}",
             r"\label{" + label + "}", "\\" + size,
             r"\setlength{\tabcolsep}{" + str(spacing) + "pt}",
             r"\begin{tabular}{@{}" + columns + "@{}}", r"\toprule",
             header + r" \\", r"\midrule"]
    lines.extend(rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", "}"])
    return "\n".join(lines)


def mean(values):
    return float(np.mean(list(values)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path,
                        default=Path("../../results/completion_extension_full_cpu"))
    parser.add_argument("--paper", type=Path, required=True)
    args = parser.parse_args()
    root, paper = args.root.resolve(), args.paper.resolve()
    summary = json.loads((root / "test/summary.json").read_text(encoding="utf-8"))
    audit = json.loads((root / "test/audit.json").read_text(encoding="utf-8"))
    complete = json.loads((root / "test/complete.json").read_text(encoding="utf-8"))
    selection = json.loads((root / "selection.json").read_text(encoding="utf-8"))
    assert summary["complete"] and audit["passed"]
    assert complete["runs"] == complete["expected_runs"] == 2124
    assert complete["attempted"] == 240 and complete["initialized"] == 236
    assert audit["verified_saved_layouts"] == 17228
    assert summary["protocol_sha256"] == audit["protocol_sha256"] == complete["protocol_sha256"]
    assert summary["results_manifest_sha256"] == audit["results_manifest_sha256"]
    assert selection["selected_configuration"] == "small_beam"
    assert not selection["test_results_used"]
    quality = {(r["geometry"], r["n"], r["fill"], r["method"], r["budget_s"]): r
               for r in summary["quality"]}
    contrasts = {(r["n"], r["fill"], r["control"], r["budget_s"]): r
                 for r in summary["paired_contrasts"] if r["geometry"] == "pooled"}
    targets = {(r["geometry"], r["n"], r["fill"], r["method"], r["target_pct"]): r
               for r in summary["time_to_target"]}
    assert len(quality) == 16 * 3 * 8
    assert len(contrasts) == 8 * 2 * 8
    assert len(targets) == 16 * 3 * 3

    def pooled(n, fill, method, budget):
        return mean(quality[(g, n, fill, method, budget)]["improvement_pct"]
                    for g in GEOMETRIES)

    for (n, fill, control, budget), contrast in contrasts.items():
        expected = pooled(n, fill, "M1", budget) - pooled(n, fill, control, budget)
        assert np.isclose(contrast["M1_minus_control_pp"], expected, atol=1e-10, rtol=0)
        assert contrast["ci_low"] <= contrast["ci_high"]
    for row in summary["coverage"]:
        for method in METHODS:
            for target in (1.0, 5.0, 10.0):
                record = targets[(row["geometry"], row["n"], row["fill"], method, target)]
                assert record["runs"] == row["initialized"] * 3
                assert 0 <= record["hit_fraction"] <= 1
                assert 0 <= record["restricted_mean_search_s"] <= 60

    text = ["% Generated from the audited 60-second study; do not edit numbers by hand.",
            "% Protocol: " + summary["protocol_sha256"],
            "% Result manifest: " + summary["results_manifest_sha256"]]
    rows = []
    for fill in FILLS:
        if rows:
            rows.append(r"\midrule")
        for n in SIZES:
            values = []
            for budget in (10.0, 60.0):
                for control in ("M0", "ALNS:small_beam"):
                    r = contrasts[(n, fill, control, budget)]
                    values.append(f"${r['M1_minus_control_pp']:+.2f}"
                                  f"\;[{r['ci_low']:+.2f},{r['ci_high']:+.2f}]$")
            rows.append(f"{n} & {fill:.2f} & " + " & ".join(values) + r" \\")
    text.append(table("ExtendedContrastsTable",
        r"M1 minus control in percentage points of initial-objective improvement. "
        r"Entries give the paired mean and descriptive pointwise 95\% interval, "
        r"with equal weight for the two geometries. Positive values favor M1. "
        r"Each fill 0.90 row has 20 initialized instances; fill 0.95 has 36 at "
        r"$n=32$ and 40 at each other size.", "tab:extendedcontrasts", "rrcccc",
        r"$n$ & fill & \multicolumn{2}{c}{10 seconds} & \multicolumn{2}{c}{60 seconds} \\"
        "\n" + r"\cmidrule(lr){3-4}\cmidrule(l){5-6}" + "\n" +
        r" & & M1$-$M0 & M1$-$ALNS & M1$-$M0 & M1$-$ALNS",
        rows, size="scriptsize", spacing=1.5))

    rows = []
    for name, cap, width, candidates in (("small_beam", 4, 4, 4),
            ("medium_beam", 8, 4, 4), ("wide_beam", 8, 8, 8), ("large_beam", 16, 4, 4)):
        score = selection["scores"]["ALNS:" + name]
        rows.append(f"ALNS & {cap} & {width} & {candidates} & {score:.2f}" + r" \\")
    for name, cap in (("B2", 16), ("B2S", 4)):
        rows.append(f"{name} & {cap} & n/a & n/a & {selection['scores'][name]:.2f}" + r" \\")
    text.append(table("ExtendedValidationTable",
        r"Validation at 10 seconds on 16 instances with two seeds. Improvement "
        r"is the mean percentage gain over the common initializer. The selected "
        r"ALNS configuration has removal cap four and beam width four. Candidate "
        r"counts exclude the extra original position when legal; B2 and B2S use "
        r"a different insertion rule.", "tab:extendedvalidation", "lrrrr",
        "method & removal cap & beam width & candidates & improvement", rows))

    rows = []
    for r in summary["coverage"]:
        f = r["achieved_fill"]
        rows.append(f"{r['geometry']} & {r['n']} & {r['fill']:.2f} & "
                    f"{r['initialized']}/{r['attempted']} & "
                    f"{f['min']:.5f} to {f['max']:.5f} & "
                    f"{r['initialization_s']['mean']:.4f}" + r" \\")
    text.append(table("ExtendedCoverageTable",
        r"Initialization coverage and achieved fill in the 60-second study. "
        r"Failures remain in the attempted count. Fill ranges cover all attempted "
        r"instances; mean initialization-attempt time is in seconds over all attempts.",
        "tab:extendedcoverage", "lrrrlr",
        "geometry & $n$ & fill & initialized & achieved fill & init. time", rows,
        size="footnotesize"))

    for fill, command, label in ((0.90, "ExtendedQualityBoundaryTable", "boundary"),
                                 (0.95, "ExtendedQualityDenseTable", "dense")):
        rows = []
        for geometry in GEOMETRIES:
            if rows:
                rows.append(r"\midrule")
            for n in SIZES:
                for budget in (10.0, 60.0):
                    values = [quality[(geometry, n, fill, m, budget)]["improvement_pct"]
                              for m in METHODS]
                    rows.append(f"{geometry} & {n} & {budget:.0f} & " +
                                " & ".join(f"{x:.2f}" for x in values) + r" \\")
        text.append(table(command,
            rf"Geometry-specific mean percentage improvement at fill {fill:.2f}. "
            r"Seeds are averaged within each instance before the cell mean. "
            r"Time excludes initialization; all methods use the common initialized set.",
            "tab:extendedquality" + label, "lrrrrr",
            "geometry & $n$ & seconds & M0 & M1 & ALNS", rows))

    rows = []
    for fill in FILLS:
        if rows:
            rows.append(r"\midrule")
        for n in SIZES:
            for method in METHODS:
                values = []
                for target in (1.0, 5.0, 10.0):
                    hit = mean(targets[(g, n, fill, method, target)]["hit_fraction"]
                               for g in GEOMETRIES)
                    restricted = mean(targets[(g, n, fill, method, target)]
                                      ["restricted_mean_search_s"] for g in GEOMETRIES)
                    values.append(f"{100 * hit:.1f} / {restricted:.2f}")
                rows.append(f"{n} & {fill:.2f} & {method.split(':')[0]} & " +
                            " & ".join(values) + r" \\")
    text.append(table("ExtendedTargetsTable",
        r"Target attainment within 60 seconds, pooling the two geometries with "
        r"equal weight. Each entry is hit percentage / restricted mean search "
        r"time in seconds. Non-hits contribute 60 seconds, so the time is not "
        r"a success-only mean. Three seeds per instance contribute equally.",
        "tab:extendedtargets", "rrlrrr",
        r"$n$ & fill & method & target 1\% & target 5\% & target 10\%", rows,
        size="footnotesize"))

    rows = []
    for method in METHODS:
        r = summary["timing"][method]
        ratio, overrun = r["cpu_wall_ratio"], r["search_overrun_s"]
        rows.append(f"{method.split(':')[0]} & {ratio['mean']:.4f} & "
                    f"{ratio['min']:.4f} & {ratio['median']:.4f} & "
                    f"{overrun['mean'] * 1000:.2f} & {overrun['max'] * 1000:.2f} & "
                    f"{r['late_completed_events']}" + r" \\")
    text.append(table("ExtendedTimingTable",
        r"Timing over 708 runs per method. Ratios compare process CPU time with "
        r"wall time. Return overruns are milliseconds beyond the 60-second "
        r"deadline. Late completed candidates are discarded, not admitted to "
        r"any quality comparison. Final verification failures are zero for all methods.",
        "tab:extendedtiming", "lrrrrrr",
        r"method & mean ratio & min. ratio & median ratio & mean overrun & max. overrun & late",
        rows, size="scriptsize"))

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False,
                         "axes.spines.right": False, "pdf.fonttype": 42})
    style = {"M0": ("#252525", "s", "--"), "M1": ("#0072B2", "o", "-"),
             "ALNS:small_beam": ("#D55E00", "^", ":")}
    for sizes, filename in (((128, 256), "fig_extended_anytime.pdf"),
                            (SIZES, "fig_extended_anytime_full.pdf")):
        fig, axes = plt.subplots(len(sizes), 2, figsize=(6.5, 2.3 * len(sizes) + 0.35),
                                 squeeze=False)
        for i, n in enumerate(sizes):
            for j, fill in enumerate(FILLS):
                ax = axes[i, j]
                for method in METHODS:
                    color, marker, linestyle = style[method]
                    ax.plot(BUDGETS, [pooled(n, fill, method, b) for b in BUDGETS],
                            color=color, marker=marker, linestyle=linestyle,
                            markersize=3.5, linewidth=1.4, label=method.split(":")[0])
                ax.set_xscale("log")
                ax.set_xticks([0.1, 1, 10, 60], ["0.1", "1", "10", "60"])
                ax.set_ylim(bottom=0)
                ax.grid(axis="y", color="0.90", linewidth=0.5)
                ax.set_title(f"n = {n}, fill = {fill:.2f}", fontsize=10)
                if j == 0:
                    ax.set_ylabel("Improvement (%)")
                if i == len(sizes) - 1:
                    ax.set_xlabel("Search time (seconds)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.tight_layout(rect=(0, 0, 1, 0.95 if len(sizes) == 2 else 0.975))
        fig.savefig(paper / "figs" / filename, metadata={"Author": "", "Creator": "Matplotlib"})
        plt.close(fig)
    print(json.dumps({"tex": "\n\n".join(text) + "\n", "checks": {
        "quality_rows": len(quality), "pooled_contrasts": len(contrasts),
        "target_rows": len(targets), "audit_passed": audit["passed"]}}))


if __name__ == "__main__":
    main()
