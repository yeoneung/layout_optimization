"""Select ALNS configurations from a completed validation phase (X3).

Two selections are made from the same runs: the configuration with the
largest mean improvement at 10 seconds (ALNS@10) and at 60 seconds (ALNS@60),
seeds averaged within instance before averaging instances, ties resolved by
name.  Test results never enter the selection.  Writes selection.json next to
the phase directory and prints the ranking.

python select_alns.py <validation phase dir>
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    directory = args.directory.resolve()
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))["settings"]
    budgets = protocol["budgets_s"]
    by_instance = defaultdict(lambda: defaultdict(list))
    for path in (directory / "runs").glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if not run["method"].startswith("ALNS:"):
            continue
        key = (run["geometry"], run["n"], run["fill"], run["instance"])
        by_instance[run["method"]][key].append({r["budget_s"]: r["improvement_pct"] for r in run["rows"]})
    scores = {}
    for method, instances in by_instance.items():
        scores[method] = {}
        for b in (10.0, 60.0):
            if b not in budgets:
                continue
            scores[method][str(b)] = float(np.mean([np.mean([r[b] for r in rows]) for rows in instances.values()]))
        scores[method]["instances"] = len(instances)
    selection = {}
    for b in ("10.0", "60.0"):
        candidates = sorted(m for m in scores if b in scores[m])
        if candidates:
            best = max(candidates, key=lambda m: (scores[m][b], -ord(m[5]) if len(m) > 5 else 0))
            best = sorted([m for m in candidates if abs(scores[m][b] - scores[best][b]) < 1e-12])[0]
            selection["ALNS@" + b.split(".")[0]] = best.split(":", 1)[1]
    report = {"selection": selection, "scores": scores, "test_results_used": False,
              "rule": "largest mean within-instance seed-averaged improvement; ties by name"}
    (directory.parent / "selection.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for b in ("10.0", "60.0"):
        print("== ranking at {} s".format(b))
        for m in sorted(scores, key=lambda m: -scores[m].get(b, -1e9))[:8]:
            print("  {:<28s} {:.3f}  ({} instances)".format(m, scores[m].get(b, float("nan")), scores[m]["instances"]))
    print("selected:", selection)


if __name__ == "__main__":
    main()
