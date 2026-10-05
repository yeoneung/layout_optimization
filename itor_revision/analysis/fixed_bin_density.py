"""Recompute the manuscript's observed densities in fixed 20-second bins.

Each recorded round has equal weight. Bins use round start time, not return
 time. Gains already exclude candidates completed after the global deadline.
This is a descriptive aggregation, not a causal estimate of operator quality.
"""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path

ARMS = {("M1", 4): "partial_repair", ("M0", None): "regeneration",
        ("ALNS", None): "destroy_and_repair"}


def summarize(directory, method="BANDITP3:8"):
    values = defaultdict(list)
    runs = rounds = 0
    for path in sorted((directory / "runs").glob("*" + method.replace(":", "_") + "*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record["method"] != method:
            continue
        runs += 1
        for step in record["stats"]["schedule"]:
            start, elapsed = step["start_s"], step["elapsed_s"]
            if not (0 <= start < 60) or elapsed <= 0:
                raise ValueError("Unexpected round interval in " + path.name)
            arm = ARMS[(step["mode"], step["cap"])]
            density = 100 * step["gain"] / (record["initial_J"] * elapsed)
            values[(arm, int(start // 20))].append(density)
            rounds += 1
    if not runs:
        raise ValueError("No matching runs")
    return {"method": method, "runs": runs, "rounds": rounds,
            "bins_s": [[0, 20], [20, 40], [40, 60]],
            "operators": {arm: [{"rounds": len(values[(arm, i)]),
                "mean_density": math.fsum(values[(arm, i)]) / len(values[(arm, i)])
                if values[(arm, i)] else None} for i in range(3)] for arm in ARMS.values()}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(summarize(args.directory), indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
