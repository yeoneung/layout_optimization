"""Small deterministic checks of the predeclared aggregation rules."""

from analyze_extended_study import paired_interval, summarize


def main():
    effect, low, high = paired_interval([1, 1, 9], ["g", "g", "n"], seed=2)
    assert (effect, low, high) == (5.0, 5.0, 5.0)
    protocol = {"cells": [{"geometry": "g", "n": 8, "fill": 0.9}],
                "methods": ["control"], "budgets_s": [60],
                "target_improvements_pct": [5], "phase": "unit"}
    initialization = {"geometry": "g", "n": 8, "fill": 0.9, "initialized": True,
                      "achieved_fill": 0.9, "initialization_s": 2, "mean_facility_area": 10}
    runs = []
    for seed, final in enumerate([110, 100]):
        runs.append({"geometry": "g", "n": 8, "fill": 0.9, "method": "control",
                     "instance": 0, "evaluation_seed": seed, "initial_J": 100,
                     "initialization_s": 2, "cpu_wall_ratio": 1,
                     "rows": [{"J": final, "improvement": final - 100, "improvement_pct": final - 100}],
                     "trace": [{"elapsed_s": 0, "J": 100}, {"elapsed_s": 10, "J": final}],
                     "stats": {"overrun_s": 0, "final_verification_failures": 0, "late_completed_events": 0}})
    result = summarize(protocol, [initialization], runs)
    assert result["quality"][0]["improvement_pct"] == 5
    assert result["quality"][0]["instances"] == 1
    target = result["time_to_target"][0]
    assert target["hit_fraction"] == 0.5
    assert target["restricted_mean_search_s"] == 35  # (10 + 60) / 2
    assert target["restricted_mean_plus_initialization_s"] == 37
    print("Geometry weighting, seed averaging and censoring tests passed")


if __name__ == "__main__":
    main()
