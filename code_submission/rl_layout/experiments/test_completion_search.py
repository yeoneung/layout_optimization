"""Smoke and invariant checks for the shared-incumbent comparison."""

import sys
import time

import bench
from completion_search import (LayoutProblem, METHODS, _empty_stats, _record,
                               run_anytime)
from layout_verifier import verify_complete_layout


def main():
    cell = bench.suite_cell("val", 8, 0.75, 1,
                            tag="completion_search_test", avg_area=(6.0, 8.0))
    problem = LayoutProblem(cell["inst"])
    ok, witness, _ = problem.initial_best_contact()
    assert ok
    assert verify_complete_layout(problem.iw, problem.ih,
                                  problem.gw, problem.gh, witness).valid
    for k, method in enumerate(METHODS):
        out = run_anytime(problem, witness, method, [0.0, 0.03], seed=100 + k,
                          kappa=4, repair_cap=4)
        assert len(out["rows"]) == 2
        assert all(row["J"] + 1e-12 >= out["initial_J"] for row in out["rows"])
        assert out["rows"][1]["J"] + 1e-12 >= out["rows"][0]["J"]
        assert all(event["elapsed_s"] <= 0.03 + 1e-12
                   for event in out["trace"])
        if method == "R0":
            assert out["stats"]["repair_attempts"] == 0
            assert out["stats"]["repair_successes"] == 0
        if method == "B2S" and out["stats"]["destroy_sizes"]:
            assert max(out["stats"]["destroy_sizes"]) <= 4

    # A candidate that starts before the deadline but completes after it must
    # not enter the anytime trace.
    original_score = problem.score
    def slow_score(placements, already_verified=False):
        time.sleep(0.01)
        return original_score(placements, already_verified=already_verified)
    problem.score = slow_score
    start = time.perf_counter()
    events = []
    stats = _empty_stats()
    admitted = _record(problem, witness, start, start + 0.003,
                       events, stats, "deadline-test")
    assert not admitted
    assert not events
    assert stats["late_completed_events"] == 1
    print("SHARED-INCUMBENT SEARCH CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
