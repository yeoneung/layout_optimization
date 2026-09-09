"""Validate and summarize the certified-decoder study.

The script deliberately works from the saved per-instance outcomes.  It checks
the soundness and monotonicity invariants, computes exact binomial intervals and
paired exact McNemar tests, and writes a compact machine-readable supplement.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import binomtest, beta, wilcoxon


def cp_interval(successes: int, trials: int, alpha: float = 0.05) -> list[float | None]:
    if trials == 0:
        return [None, None]
    lo = 0.0 if successes == 0 else float(beta.ppf(alpha / 2, successes, trials - successes + 1))
    hi = 1.0 if successes == trials else float(beta.ppf(1 - alpha / 2, successes + 1, trials - successes))
    return [lo, hi]


def record_key(r: dict) -> tuple:
    return (
        r["suite"],
        int(r["n_rooms"]),
        float(r["fill"]),
        r["method"],
        r.get("policy"),
        r.get("witness_effort"),
    )


def mcnemar(unfiltered: dict, guided: dict) -> dict:
    a = np.asarray(unfiltered["geometrically_feasible"], dtype=bool)
    b = np.asarray(guided["geometrically_feasible"], dtype=bool)
    assert a.shape == b.shape
    helped = int(np.sum(~a & b))
    harmed = int(np.sum(a & ~b))
    discordant = helped + harmed
    p = 1.0 if discordant == 0 else float(binomtest(helped, discordant, 0.5).pvalue)
    return {
        "n": int(a.size),
        "unfiltered_feasible": int(a.sum()),
        "guided_feasible": int(b.sum()),
        "helped": helped,
        "harmed": harmed,
        "discordant": discordant,
        "exact_two_sided_p": p,
    }


def paired_quality(certified: dict, guided: dict, rng: np.random.Generator) -> dict:
    returned = np.asarray(certified["returned"], dtype=bool)
    guided_ok = np.asarray(guided["geometrically_feasible"], dtype=bool)
    cert_j = np.asarray([np.nan if x is None else x for x in certified["J"]], dtype=float)
    guided_j = np.asarray([np.nan if x is None else x for x in guided["J"]], dtype=float)
    mask = returned & guided_ok & np.isfinite(cert_j) & np.isfinite(guided_j)
    delta = cert_j[mask] - guided_j[mask]
    if delta.size == 0:
        return {"paired_n": 0}
    boot = np.mean(rng.choice(delta, size=(10_000, delta.size), replace=True), axis=1)
    if np.allclose(delta, 0.0):
        wp = 1.0
    else:
        wp = float(wilcoxon(delta, alternative="two-sided").pvalue)
    return {
        "paired_n": int(delta.size),
        "mean_certified_minus_guided": float(delta.mean()),
        "median_certified_minus_guided": float(np.median(delta)),
        "bootstrap_mean_ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
        "wilcoxon_two_sided_p": wp,
        "interpretation": "negative favors certified because the objective is minimized",
    }


def compact(r: dict) -> dict:
    s = r["summary"]
    returned = int(sum(bool(x) for x in r["returned"]))
    feasible_returns = int(sum(bool(x) and bool(y) for x, y in zip(r["returned"], r["geometrically_feasible"])))
    return {
        "method": r["method"],
        "policy": r.get("policy"),
        "effort": r.get("witness_effort"),
        "returned": returned,
        "instances": int(r["instances"]),
        "coverage": returned / int(r["instances"]),
        "coverage_ci95_exact": cp_interval(returned, int(r["instances"])),
        "feasible_returns": feasible_returns,
        "empirical_feasibility": feasible_returns / int(r["instances"]),
        "return_feasibility": None if returned == 0 else feasible_returns / returned,
        "return_feasibility_ci95_exact": cp_interval(feasible_returns, returned),
        "conditional_mean_J": s.get("mean_J_valid"),
        "ms_per_instance": s.get("ms_per_instance"),
        "rejects_total": s.get("rejects_total"),
        "witness_used_total": s.get("witness_used_total"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("certified_study.json"))
    parser.add_argument("--output", type=Path, default=Path("certified_study_summary.json"))
    args = parser.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    records = data["records"]
    assert len(records) == 160, f"expected 160 method-cell records, found {len(records)}"
    by_key = {record_key(r): r for r in records}
    assert len(by_key) == len(records), "duplicate method-cell record"

    bad_returns = []
    for r in records:
        if r.get("guaranteed"):
            for i, (returned, feasible) in enumerate(zip(r["returned"], r["geometrically_feasible"])):
                if returned and not feasible:
                    bad_returns.append([record_key(r), i])
    assert not bad_returns, f"certified infeasible returns: {bad_returns}"
    witness_checks = data.get("witness_checks", [])
    assert len(witness_checks) == 14
    assert all(float(x["max_overlap"]) <= 1e-9 and bool(x["all_inside"]) for x in witness_checks)
    certified_returns = sum(
        sum(bool(x) for x in r["returned"]) for r in records if r.get("guaranteed")
    )
    dense_certified_returns = sum(
        sum(bool(x) for x in r["returned"])
        for r in records if r.get("guaranteed") and r["suite"] == "dense"
    )
    assert certified_returns == 2202
    assert dense_certified_returns == 882

    coverage_checks = []
    for suite in ("full", "dense"):
        cells = sorted({(int(r["n_rooms"]), float(r["fill"])) for r in records if r["suite"] == suite})
        efforts = (1, 8, 32, 128) if suite == "full" else (1, 128)
        for n, fill in cells:
            masks = []
            for effort in efforts:
                s1 = by_key[(suite, n, fill, "policy_witness_certified", "seed1", effort)]
                s2 = by_key[(suite, n, fill, "policy_witness_certified", "seed2", effort)]
                assert s1["returned"] == s2["returned"], (suite, n, fill, effort, "seed-dependent coverage")
                masks.append(np.asarray(s1["returned"], dtype=bool))
            for previous, current in zip(masks, masks[1:]):
                assert np.all(~previous | current), (suite, n, fill, "non-monotone effort coverage")
            coverage_checks.append({
                "suite": suite,
                "n_rooms": n,
                "fill": fill,
                "coverage_by_effort": {
                    str(e): float(m.mean()) for e, m in zip(efforts, masks)
                },
            })

    dense_cells = []
    mcnemar_results = []
    quality_results = []
    rng = np.random.default_rng(20260816)
    for n, fill in ((32, 0.90), (32, 0.95), (64, 0.90), (64, 0.95)):
        cell_records = []
        for method, policy, effort in (
            ("largest_first_unfiltered", None, None),
            ("largest_first_guided", None, None),
            ("policy_unfiltered", "seed1", None),
            ("policy_unfiltered", "seed2", None),
            ("policy_guided", "seed1", None),
            ("policy_guided", "seed2", None),
            ("policy_witness_certified", "seed1", 1),
            ("policy_witness_certified", "seed2", 1),
            ("policy_witness_certified", "seed1", 128),
            ("policy_witness_certified", "seed2", 128),
        ):
            cell_records.append(compact(by_key[("dense", n, fill, method, policy, effort)]))
        dense_cells.append({"n_rooms": n, "fill": fill, "records": cell_records})

        for policy, base_name, guided_name in (
            (None, "largest_first_unfiltered", "largest_first_guided"),
            ("seed1", "policy_unfiltered", "policy_guided"),
            ("seed2", "policy_unfiltered", "policy_guided"),
        ):
            result = mcnemar(
                by_key[("dense", n, fill, base_name, policy, None)],
                by_key[("dense", n, fill, guided_name, policy, None)],
            )
            result.update({"n_rooms": n, "fill": fill, "selector": "largest_first" if policy is None else policy})
            mcnemar_results.append(result)

        for policy in ("seed1", "seed2"):
            result = paired_quality(
                by_key[("dense", n, fill, "policy_witness_certified", policy, 128)],
                by_key[("dense", n, fill, "policy_guided", policy, None)],
                rng,
            )
            result.update({"n_rooms": n, "fill": fill, "policy": policy, "effort": 128})
            quality_results.append(result)

    # One conservative multiplicity family for all twelve confirmatory binary
    # comparisons (four cells times three selectors).
    order = sorted(range(len(mcnemar_results)), key=lambda i: mcnemar_results[i]["exact_two_sided_p"])
    running = 0.0
    for rank, i in enumerate(order):
        adjusted = min(1.0, (len(order) - rank) * mcnemar_results[i]["exact_two_sided_p"])
        running = max(running, adjusted)
        mcnemar_results[i]["holm_adjusted_p"] = running

    full_cells = []
    for n in (32, 64):
        for fill in (0.75, 0.80, 0.85, 0.90, 0.95):
            rows = [
                compact(by_key[("full", n, fill, "policy_witness_certified", "seed1", effort)])
                for effort in (1, 8, 32, 128)
            ]
            full_cells.append({"n_rooms": n, "fill": fill, "records": rows})

    out = {
        "source": str(args.input),
        "validation": {
            "record_count": len(records),
            "certified_infeasible_returns": 0,
            "certified_returns": certified_returns,
            "dense_certified_returns": dense_certified_returns,
            "coverage_policy_seed_invariant": True,
            "coverage_monotone_in_effort": True,
            "witness_checks": len(witness_checks),
            "all_generated_benchmark_witnesses_valid": True,
        },
        "full_certified": full_cells,
        "dense_confirmation": dense_cells,
        "dense_guidance_mcnemar": mcnemar_results,
        "dense_certified_vs_guided_conditional_quality": quality_results,
    }
    args.output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")

    print(f"validated {len(records)} records; certified infeasible returns: 0")
    for check in coverage_checks:
        print(
            f"{check['suite']:5s} n={check['n_rooms']:2d} f={check['fill']:.2f} "
            f"coverage={check['coverage_by_effort']}"
        )
    print("\nDense paired guidance tests (helped/harmed, exact p):")
    for row in mcnemar_results:
        print(
            f"n={row['n_rooms']:2d} f={row['fill']:.2f} {row['selector']:13s} "
            f"{row['helped']:2d}/{row['harmed']:2d} p={row['exact_two_sided_p']:.4g} "
            f"Holm={row['holm_adjusted_p']:.4g}"
        )
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
