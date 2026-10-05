"""Phase 0b: when does full regeneration start improving faster than partial repair?

From the published traces, build mean best-so-far curves for M0 and M1 on a log
time grid per (n, fill), then report
  * level crossover: first time the M0 curve exceeds the M1 curve,
  * rate crossover: first time the M0 improvement rate exceeds the M1 rate for
    three consecutive grid points,
  * per-instance level-crossover time distribution (heterogeneity motivates an
    adaptive rather than a fixed switching time).
Uses the compact cache written by phase0_headroom.py.
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "phase0"
CACHE = OUT / "runs_compact.pkl"
GRID = np.concatenate([[0.05, 0.1, 0.2, 0.3, 0.5, 0.7], np.linspace(1, 10, 19), np.linspace(12, 60, 25)])


def best_so_far(trace, initial, grid):
    times = np.array([e[0] for e in trace])
    values = np.array([e[1] for e in trace])
    running = np.maximum.accumulate(values)
    idx = np.searchsorted(times, grid, side="right") - 1
    idx = np.clip(idx, 0, len(times) - 1)
    return 100.0 * (running[idx] - initial) / initial


def main():
    frame = pd.read_pickle(CACHE)
    frame = frame[frame.method.isin(["M0", "M1"])]
    curves = defaultdict(list)          # (geometry, n, fill, instance, method) -> list of curves (seeds)
    for row in frame.itertuples():
        curves[(row.geometry, row.n, row.fill, row.instance, row.method)].append(
            best_so_far(row.trace, row.initial_J, GRID))
    per_instance = {k: np.mean(v, axis=0) for k, v in curves.items()}
    rows, dist_rows = [], []
    for (n, fill) in sorted({(k[1], k[2]) for k in per_instance}):
        cell_curve = {}
        for method in ("M0", "M1"):
            by_geometry = defaultdict(list)
            for k, c in per_instance.items():
                if (k[1], k[2], k[4]) == (n, fill, method):
                    by_geometry[k[0]].append(c)
            cell_curve[method] = np.mean([np.mean(v, axis=0) for v in by_geometry.values()], axis=0)
        diff = cell_curve["M0"] - cell_curve["M1"]
        level = next((float(GRID[i]) for i in range(len(GRID)) if diff[i] > 0 and np.all(diff[i:min(i + 3, len(GRID))] > 0)), None)
        rate0 = np.diff(cell_curve["M0"]) / np.diff(GRID)
        rate1 = np.diff(cell_curve["M1"]) / np.diff(GRID)
        faster = rate0 > rate1
        rate = next((float(GRID[i + 1]) for i in range(len(faster) - 2) if faster[i] and faster[i + 1] and faster[i + 2]), None)
        # per-instance level crossover
        inst_cross = []
        for k in per_instance:
            if (k[1], k[2], k[4]) == (n, fill, "M0"):
                m1 = per_instance[(k[0], k[1], k[2], k[3], "M1")]
                d = per_instance[k] - m1
                t = next((float(GRID[i]) for i in range(len(GRID)) if d[i] > 0 and np.all(d[i:] > 0)), None)
                inst_cross.append(t if t is not None else np.inf)
        inst_cross = np.array(inst_cross)
        finite = inst_cross[np.isfinite(inst_cross)]
        rows.append({"n": n, "fill": fill, "instances": len(inst_cross),
                     "level_crossover_s": level, "rate_crossover_s": rate,
                     "frac_M0_never_overtakes": float(np.mean(~np.isfinite(inst_cross))),
                     "inst_crossover_median_s": float(np.median(finite)) if len(finite) else None,
                     "inst_crossover_q25_s": float(np.percentile(finite, 25)) if len(finite) else None,
                     "inst_crossover_q75_s": float(np.percentile(finite, 75)) if len(finite) else None,
                     "M0_at_60": float(cell_curve["M0"][-1]), "M1_at_60": float(cell_curve["M1"][-1])})
        for t, a, b in zip(GRID, cell_curve["M0"], cell_curve["M1"]):
            dist_rows.append({"n": n, "fill": fill, "t_s": float(t), "M0": float(a), "M1": float(b)})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "rate_crossover.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(OUT / "mean_curves.csv", index=False)
    lines = ["# Phase 0b: M0 versus M1 improvement-rate crossover (published traces)", "",
             "level crossover = first grid time at which the mean M0 curve stays above the mean M1 curve;",
             "rate crossover = first time M0 improves faster than M1 for three consecutive grid points;",
             "per-instance crossover = time after which M0 stays above M1 for that instance (inf if never).", "",
             "| n | fill | level cross (s) | rate cross (s) | M0 never overtakes | inst. cross median [q25, q75] (s) | M0@60 | M1@60 |",
             "|---:|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows:
        med = ("{:.1f} [{:.1f}, {:.1f}]".format(r["inst_crossover_median_s"], r["inst_crossover_q25_s"], r["inst_crossover_q75_s"])
               if r["inst_crossover_median_s"] is not None else "n/a")
        lines.append("| {} | {:.2f} | {} | {} | {:.2f} | {} | {:.2f} | {:.2f} |".format(
            r["n"], r["fill"], "{:.1f}".format(r["level_crossover_s"]) if r["level_crossover_s"] else "never",
            "{:.1f}".format(r["rate_crossover_s"]) if r["rate_crossover_s"] else "never",
            r["frac_M0_never_overtakes"], med, r["M0_at_60"], r["M1_at_60"]))
    (OUT / "phase0b_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
