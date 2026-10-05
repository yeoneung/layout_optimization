"""Analysis and monitoring for EAAI-track phases.

Works on partial result directories (for monitoring) and on complete ones.
Paired statistics use only instances for which every method and seed is
present, so partial reports are internally consistent but marked partial.

Outputs (in the phase directory unless --out is given):
  quality.csv          pooled mean improvement per (n, fill, method, budget)
  contrasts.csv        method minus control (pp), geometry-stratified paired
                       bootstrap interval, win fraction
  oracle.csv           per-instance oracle best-of(M0, M1) and each method's
                       mean gap to it and to the per-instance best-known value
  schedule.csv         switching diagnostics for extended methods
  report.md            compact tables at the two largest budgets
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "code_submission" / "rl_layout" / "experiments"
sys.path.insert(0, str(ARCHIVE))

from analyze_extended_study import paired_interval  # noqa: E402
from extended_protocol import digest  # noqa: E402

GEOMETRIES = ("guillotine", "nonslicing")


def load(directory, metric="improvement"):
    """metric 'improvement': percent of the initial objective (published metric);
    metric 'cost': percent reduction of the initial flow cost (flow-objective phases)."""
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))["settings"]
    initial_cost = {}
    if metric == "cost":
        for path in (directory / "initializations").glob("*.json"):
            init = json.loads(path.read_text(encoding="utf-8"))
            if init.get("initial_cost"):
                initial_cost[(init["geometry"], init["n"], init["fill"], init["instance"])] = init["initial_cost"]
    runs = []
    for path in sorted((directory / "runs").glob("*.json")):
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue  # file being written
        stats = raw["stats"]
        key = (raw["geometry"], raw["n"], raw["fill"], raw["instance"])
        if metric == "cost":
            if key not in initial_cost:
                continue
            imp = {r["budget_s"]: 100.0 * r["improvement"] / initial_cost[key] for r in raw["rows"]}
        else:
            imp = {r["budget_s"]: r["improvement_pct"] for r in raw["rows"]}
        runs.append({
            "method": raw["method"], "geometry": raw["geometry"], "n": raw["n"], "fill": raw["fill"],
            "instance": raw["instance"], "seed": raw["evaluation_seed"], "initial_J": raw["initial_J"],
            "imp": imp,
            "J": {r["budget_s"]: r["J"] for r in raw["rows"]},
            "cpu_wall_ratio": raw.get("cpu_wall_ratio"),
            "first_M0_start_s": stats.get("first_M0_start_s"),
            "time_by_mode": stats.get("time_by_mode"),
            "reconstructions_by_mode": stats.get("reconstructions_by_mode"),
            "iterations": stats.get("iterations"),
            "repair_attempts": stats.get("repair_attempts"), "repair_successes": stats.get("repair_successes"),
        })
    inits = [json.loads(p.read_text(encoding="utf-8")) for p in sorted((directory / "initializations").glob("*.json"))]
    return protocol, inits, runs


def complete_instances(protocol, runs):
    """instance key -> method -> list of runs (one per seed), only when complete."""
    table = defaultdict(lambda: defaultdict(list))
    for run in runs:
        table[(run["geometry"], run["n"], run["fill"], run["instance"])][run["method"]].append(run)
    complete = {}
    for key, per_method in table.items():
        if all(len(per_method.get(m, [])) == protocol["evaluation_seeds"] for m in protocol["methods"]):
            complete[key] = per_method
    return complete


def seed_mean(runs, budget):
    return float(np.mean([r["imp"][budget] for r in runs]))


def pooled(values_by_geometry):
    present = [g for g in GEOMETRIES if values_by_geometry.get(g)]
    return float(np.mean([np.mean(values_by_geometry[g]) for g in present])) if present else float("nan")


def analyze(protocol, complete, replicates, controls=None):
    methods = protocol["methods"]
    budgets = protocol["budgets_s"]
    cells = sorted({(k[1], k[2]) for k in complete})
    quality, contrasts, oracle_rows, schedule = [], [], [], []
    controls = [m for m in (controls or ("M0", "M1")) if m in methods]
    for n, fill in cells:
        keys = sorted(k for k in complete if (k[1], k[2]) == (n, fill))
        strata = [k[0] for k in keys]
        for budget in budgets:
            values = {m: {k: seed_mean(complete[k][m], budget) for k in keys} for m in methods}
            best_known = {k: max(max(r["imp"][budgets[-1]] for r in complete[k][m]) for m in methods) for k in keys}
            oracle2 = ({k: max(values["M0"][k], values["M1"][k]) for k in keys}
                       if {"M0", "M1"} <= set(methods) else None)
            oracle_all = {k: max(values[m][k] for m in methods) for k in keys}
            ranks = {m: [] for m in methods}
            for k in keys:
                order = sorted(methods, key=lambda m: -values[m][k])
                # average ranks for ties
                position = 0
                while position < len(order):
                    tie = [order[position]]
                    while (position + len(tie) < len(order)
                           and abs(values[order[position + len(tie)]][k] - values[tie[0]][k]) < 1e-9):
                        tie.append(order[position + len(tie)])
                    mean_rank = position + (len(tie) + 1) / 2.0
                    for m in tie:
                        ranks[m].append(mean_rank)
                    position += len(tie)
            for m in methods:
                by_geometry = defaultdict(list)
                for k in keys:
                    by_geometry[k[0]].append(values[m][k])
                quality.append({"n": n, "fill": fill, "method": m, "budget_s": budget, "instances": len(keys),
                                "improvement_pct": pooled(by_geometry),
                                "mean_rank": float(np.mean(ranks[m]))})
                gap_best = defaultdict(list)
                gap_all = defaultdict(list)
                for k in keys:
                    gap_best[k[0]].append(best_known[k] - values[m][k])
                    gap_all[k[0]].append(oracle_all[k] - values[m][k])
                row = {"n": n, "fill": fill, "method": m, "budget_s": budget, "instances": len(keys),
                       "gap_to_best_known_pp": pooled(gap_best),
                       "gap_to_oracle_all_methods_pp": pooled(gap_all)}
                if oracle2 is not None:
                    gap_oracle = defaultdict(list)
                    beats = defaultdict(list)
                    for k in keys:
                        gap_oracle[k[0]].append(oracle2[k] - values[m][k])
                        beats[k[0]].append(float(values[m][k] >= oracle2[k] - 1e-9))
                    row["oracle_M0_M1_pct"] = pooled({g: [oracle2[k] for k in keys if k[0] == g] for g in GEOMETRIES})
                    row["gap_to_oracle_pp"] = pooled(gap_oracle)
                    row["matches_or_beats_oracle_frac"] = pooled(beats)
                oracle_rows.append(row)
                for control in controls:
                    if control == m:
                        continue
                    diff = [values[m][k] - values[control][k] for k in keys]
                    seed = int(digest([n, fill, m, control, budget])[:12], 16)
                    effect, low, high = paired_interval(diff, strata, seed, replicates=replicates)
                    wins = defaultdict(list)
                    for k, d in zip(keys, diff):
                        wins[k[0]].append(float(d > 1e-9))
                    contrasts.append({"n": n, "fill": fill, "method": m, "control": control, "budget_s": budget,
                                      "instances": len(keys), "diff_pp": effect, "ci_low": low, "ci_high": high,
                                      "win_fraction": pooled(wins)})
        for m in methods:
            runs = [r for k in keys for r in complete[k][m]]
            if not any(r["first_M0_start_s"] is not None or r["time_by_mode"] for r in runs):
                continue
            first = [r["first_M0_start_s"] if r["first_M0_start_s"] is not None else budgets[-1] for r in runs]
            m0_time = [(r["time_by_mode"] or {}).get("M0", 0.0) / max(sum((r["time_by_mode"] or {}).values()), 1e-9)
                       for r in runs]
            recon = [r["iterations"] for r in runs if r["iterations"] is not None]
            schedule.append({"n": n, "fill": fill, "method": m, "runs": len(runs),
                             "mean_first_M0_start_s": float(np.mean(first)),
                             "mean_M0_time_fraction": float(np.mean(m0_time)),
                             "mean_reconstructions": float(np.mean(recon)) if recon else None})
    return quality, contrasts, oracle_rows, schedule


def write_csv(path, rows):
    import csv
    if not rows:
        return
    fields = sorted({k for r in rows for k in r}, key=lambda k: list(rows[0]).index(k) if k in rows[0] else 99)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def report(directory, protocol, inits, runs, complete, quality, contrasts, oracle_rows, schedule, partial):
    budgets = protocol["budgets_s"]
    show = [b for b in budgets if b in (10.0, 60.0)] or budgets[-2:]
    methods = protocol["methods"]
    expected = sum(i["initialized"] for i in inits) * len(methods) * protocol["evaluation_seeds"]
    lines = ["# EAAI phase report: {}{}".format(protocol["phase"], " (PARTIAL)" if partial else ""), "",
             "Runs present: {} of {} expected on {} initialized instances ({} attempted so far); "
             "instances with all methods and seeds: {}.".format(
                 len(runs), expected, sum(i["initialized"] for i in inits), len(inits), len(complete)), "",
             "Methods: " + ", ".join(methods), "",
             "Seeds averaged within instance; geometries equally weighted; intervals are descriptive",
             "geometry-stratified paired bootstrap intervals.", ""]
    for budget in show:
        lines += ["## Improvement over the common initial layout at {:g} s (%)".format(budget), "",
                  "| n | fill | inst | " + " | ".join(methods) + " | oracle(M0,M1) |",
                  "|---:|---:|---:|" + "---:|" * (len(methods) + 1)]
        cells = sorted({(q["n"], q["fill"]) for q in quality})
        for n, fill in cells:
            vals = {q["method"]: q["improvement_pct"] for q in quality
                    if (q["n"], q["fill"], q["budget_s"]) == (n, fill, budget)}
            inst = next(q["instances"] for q in quality if (q["n"], q["fill"], q["budget_s"]) == (n, fill, budget))
            best = max(vals.values())
            cellvals = ["**{:.2f}**".format(v) if abs(v - best) < 1e-9 else "{:.2f}".format(v)
                        for v in (vals[m] for m in methods)]
            orc = next((o["oracle_M0_M1_pct"] for o in oracle_rows
                        if (o["n"], o["fill"], o["budget_s"]) == (n, fill, budget) and "oracle_M0_M1_pct" in o), None)
            lines.append("| {} | {:.2f} | {} | ".format(n, fill, inst) + " | ".join(cellvals)
                         + " | {} |".format("{:.2f}".format(orc) if orc is not None else "n/a"))
        lines.append("")
        if any("gap_to_oracle_pp" in o for o in oracle_rows):
            lines += ["### Gap to the per-instance oracle best-of(M0, M1) at {:g} s (pp; smaller is better)".format(budget), "",
                      "| n | fill | " + " | ".join(methods) + " |", "|---:|---:|" + "---:|" * len(methods)]
            for n, fill in cells:
                gaps = {o["method"]: o["gap_to_oracle_pp"] for o in oracle_rows
                        if (o["n"], o["fill"], o["budget_s"]) == (n, fill, budget)}
                best = min(gaps.values())
                lines.append("| {} | {:.2f} | ".format(n, fill) + " | ".join(
                    "**{:.2f}**".format(gaps[m]) if abs(gaps[m] - best) < 1e-9 else "{:.2f}".format(gaps[m])
                    for m in methods) + " |")
            lines.append("")
        # Summary across cells: wins / ties / losses against M0 and M1 by interval sign,
        # mean rank across all instances, and gap to the all-method per-instance oracle.
        summary_controls = [c for c in ("M0", "M1") if c in methods]
        lines += ["### Summary across cells at {:g} s".format(budget), "",
                  "| method | " + " | ".join("vs {} W/T/L".format(c) for c in summary_controls)
                  + " | mean rank | mean gap to all-method oracle (pp) | cells where best |",
                  "|---|" + "---:|" * (len(summary_controls) + 3)]
        for m in methods:
            parts = []
            for control in summary_controls:
                if control == m:
                    parts.append("-")
                    continue
                w = t = l = 0
                for c in contrasts:
                    if (c["method"], c["control"], c["budget_s"]) == (m, control, budget):
                        if c["ci_low"] > 0:
                            w += 1
                        elif c["ci_high"] < 0:
                            l += 1
                        else:
                            t += 1
                parts.append("{}/{}/{}".format(w, t, l))
            qs = [q for q in quality if q["method"] == m and q["budget_s"] == budget]
            rank = float(np.mean([q["mean_rank"] for q in qs])) if qs else float("nan")
            gaps = [o["gap_to_oracle_all_methods_pp"] for o in oracle_rows if o["method"] == m and o["budget_s"] == budget]
            best_cells = 0
            for n, fill in cells:
                vals = {q["method"]: q["improvement_pct"] for q in quality
                        if (q["n"], q["fill"], q["budget_s"]) == (n, fill, budget)}
                if abs(vals[m] - max(vals.values())) < 1e-9:
                    best_cells += 1
            lines.append("| {} | ".format(m) + " | ".join(parts) + " | {:.2f} | {:.2f} | {} |".format(
                rank, float(np.mean(gaps)) if gaps else float("nan"), best_cells))
        lines.append("")
        for control in [c for c in ("M0", "M1") if c in methods]:
            lines += ["### Method minus {} at {:g} s (pp, 95% interval)".format(control, budget), "",
                      "| n | fill | " + " | ".join(m for m in methods if m != control) + " |",
                      "|---:|---:|" + "---:|" * (len(methods) - 1)]
            for n, fill in cells:
                entries = []
                for m in methods:
                    if m == control:
                        continue
                    c = next(x for x in contrasts if (x["n"], x["fill"], x["method"], x["control"], x["budget_s"])
                             == (n, fill, m, control, budget))
                    mark = "+" if c["ci_low"] > 0 else ("-" if c["ci_high"] < 0 else "")
                    entries.append("{:+.2f} [{:+.2f},{:+.2f}]{}".format(c["diff_pp"], c["ci_low"], c["ci_high"],
                                                                        " " + mark if mark else ""))
                lines.append("| {} | {:.2f} | ".format(n, fill) + " | ".join(entries) + " |")
            lines.append("")
    if schedule:
        lines += ["## Switching diagnostics", "", "| n | fill | method | runs | first M0 start (s) | M0 time fraction | reconstructions |",
                  "|---:|---:|---|---:|---:|---:|---:|"]
        for s in schedule:
            lines.append("| {n} | {fill:.2f} | {method} | {runs} | {mean_first_M0_start_s:.2f} | {mean_M0_time_fraction:.2f} | {rc} |".format(
                rc="{:.1f}".format(s["mean_reconstructions"]) if s["mean_reconstructions"] is not None else "n/a", **s))
        lines.append("")
    ratios = [r["cpu_wall_ratio"] for r in runs if r["cpu_wall_ratio"] is not None]
    if ratios:
        lines += ["CPU/wall ratio: mean {:.3f}, min {:.3f}.".format(np.mean(ratios), np.min(ratios)), ""]
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--controls", default="M0,M1", help="comma-separated control methods for paired contrasts")
    parser.add_argument("--metric", choices=["improvement", "cost"], default="improvement",
                        help="cost: percent reduction of the initial flow cost (flow-objective phases)")
    parser.add_argument("--extra-dirs", dest="extra_dirs", default="",
                        help="comma-separated phase directories with the same instances whose methods are pooled in")
    parser.add_argument("--print", action="store_true")
    args = parser.parse_args()
    directory = args.directory.resolve()
    out = (args.out or directory).resolve()
    out.mkdir(parents=True, exist_ok=True)
    protocol, inits, runs = load(directory, metric=args.metric)
    for extra in [Path(p).resolve() for p in args.extra_dirs.split(",") if p.strip()]:
        extra_protocol, _, extra_runs = load(extra, metric=args.metric)
        if extra_protocol["tag"] != protocol["tag"] or extra_protocol["split"] != protocol["split"]:
            raise ValueError("extra directory uses different instances: " + str(extra))
        if extra_protocol["evaluation_seeds"] != protocol["evaluation_seeds"]:
            raise ValueError("extra directory uses a different seed count: " + str(extra))
        # methods already present keep the runs of the primary directory (e.g. in-phase drift checks)
        new_methods = [m for m in extra_protocol["methods"] if m not in protocol["methods"]]
        runs.extend(r for r in extra_runs if r["method"] in new_methods)
        protocol = dict(protocol, methods=protocol["methods"] + new_methods)
    complete = complete_instances(protocol, runs)
    partial = not (directory / "complete.json").exists()
    if not complete:
        print("no instance has all methods and seeds yet ({} runs present)".format(len(runs)))
        return
    quality, contrasts, oracle_rows, schedule = analyze(protocol, complete, args.replicates,
                                                        controls=[c for c in args.controls.split(",") if c])
    write_csv(out / "quality.csv", quality)
    write_csv(out / "contrasts.csv", contrasts)
    write_csv(out / "oracle.csv", oracle_rows)
    write_csv(out / "schedule.csv", schedule)
    lines = report(out, protocol, inits, runs, complete, quality, contrasts, oracle_rows, schedule, partial)
    if args.print:
        print("\n".join(lines))
    else:
        print("wrote", out / "report.md", "({} complete instances, {} runs)".format(len(complete), len(runs)))


if __name__ == "__main__":
    main()
