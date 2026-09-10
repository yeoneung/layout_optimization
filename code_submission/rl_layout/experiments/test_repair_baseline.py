"""Geometry, scoring, deadlines and output checks for the ALNS control."""

import time

import numpy as np

import bench
from completion_search import LayoutProblem
from layout_verifier import verify_complete_layout
from repair_baseline import (CONFIGURATIONS, beam_repair, destroy,
                             facility_contributions, ranked_positions,
                             regret_repair, run_adaptive_repair)


def main():
    cell = bench.suite_cell("val", 8, 0.75, 1,
                            tag="completion_search_test", avg_area=(6.0, 8.0))
    problem = LayoutProblem(cell["inst"])
    ok, initial, _ = problem.initial_best_contact()
    assert ok
    initial_value = problem.score(initial)
    rng = np.random.default_rng(54321)
    contributions = facility_contributions(problem, initial)
    assert contributions.shape == (8,) and np.isfinite(contributions).all()
    for kind in ["random", "spatial", "low_contribution"]:
        kept, removed, originals = destroy(problem, initial, kind, 4, rng)
        assert 2 <= len(removed) <= 4
        assert set(x[0] for x in kept).isdisjoint(removed)
        assert set(x[0] for x in kept) | set(removed) == set(range(8))
        for repair in [beam_repair, regret_repair]:
            candidate = repair(problem, kept, removed, originals, rng,
                               CONFIGURATIONS["small_beam"], time.perf_counter() + 2)
            if candidate is not None:
                assert verify_complete_layout(problem.iw, problem.ih,
                                              problem.gw, problem.gh, candidate).valid
            assert repair(problem, kept, removed, originals, rng,
                          CONFIGURATIONS["small_beam"], time.perf_counter() - 1) is None
    # The insertion score difference equals the full objective difference when
    # the other facilities are fixed. This checks the beam ranking algebra.
    for facility, row, col in initial:
        kept = [x for x in initial if x[0] != facility]
        options = ranked_positions(problem, kept, facility, 8, (row, col))
        original_gain = next(g for r, c, g in options if (r, c) == (row, col))
        assert abs(contributions[facility] - original_gain) < 1e-8
        for r, c, gain in options:
            value = problem.score(kept + [(facility, r, c)])
            assert abs((value - initial_value) - (gain - original_gain)) < 1e-8
    for name in CONFIGURATIONS:
        result = run_adaptive_repair(problem, initial, [0, 0.01, 0.05],
                                     seed=42, configuration=name)
        values = [x["J"] for x in result["rows"]]
        assert values == sorted(values)
        assert min(values) >= initial_value - 1e-10
        assert all(x["elapsed_s"] <= 0.05 for x in result["trace"])
        assert result["stats"]["final_verification_failures"] == 0
        assert max(result["stats"]["destroy_sizes"]) <= CONFIGURATIONS[name]["destroy_cap"]
        for row in result["rows"]:
            assert verify_complete_layout(problem.iw, problem.ih,
                                          problem.gw, problem.gh, row["layout"]).valid
    print("ALNS geometry, objective, deadline and anytime tests passed")


if __name__ == "__main__":
    main()
