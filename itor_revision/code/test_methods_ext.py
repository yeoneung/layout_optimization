"""Checks for the extended methods: parsing, verified rows, schedule semantics."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code_submission" / "rl_layout" / "experiments"))
sys.path.insert(0, str(HERE))

import bench  # noqa: E402
from completion_search import LayoutProblem  # noqa: E402
from layout_verifier import verify_complete_layout  # noqa: E402
import methods_ext  # noqa: E402


def make_problem():
    cell = bench.suite_cell("val", 32, 0.90, 1, tag="eaai_unit_test", max_grid=44)
    problem = LayoutProblem(cell["inst"].take([0]))
    ok, witness, _ = problem.initial_best_contact()
    assert ok, "unit-test instance must initialize"
    return problem, witness


def check_rows(problem, output, budgets):
    assert [r["budget_s"] for r in output["rows"]] == budgets
    previous = output["initial_J"] - 1e-9
    for row in output["rows"]:
        check = verify_complete_layout(problem.iw, problem.ih, problem.gw, problem.gh, row["layout"])
        assert check.valid, check.reason
        assert abs(problem.score(row["layout"], already_verified=True) - row["J"]) < 1e-8
        assert row["J"] >= previous
        previous = row["J"]
    times = [e["elapsed_s"] for e in output["trace"]]
    assert times == sorted(times) and times[-1] <= budgets[-1]
    assert output["stats"]["final_verification_failures"] == 0


def main():
    for token, kind in [("M0", "ARCHIVE"), ("CAP:8", "CAP"), ("SW:fixed0.5", "SW_FIXED"),
                        ("SW:stag3", "SW_STAG"), ("CAPSCHED:2", "CAPSCHED"),
                        ("INTERLEAVE", "INTERLEAVE"), ("BANDIT:8", "BANDIT")]:
        assert methods_ext.parse_method(token)["kind"] == kind, token
    for bad in ("SW:fixed", "CAP:x", "M9", "BANDIT"):
        try:
            methods_ext.parse_method(bad)
        except ValueError:
            pass
        else:
            raise AssertionError("accepted bad token " + bad)

    problem, witness = make_problem()
    budgets = [0.0, 0.3, 1.5]

    # CAP:4 reproduces the deterministic first reconstruction of M1 exactly.
    m1 = methods_ext.run_anytime_ext(problem, witness, "M1", [0.0, 0.05], seed=7)
    cap4 = methods_ext.run_anytime_ext(problem, witness, "CAP:4", [0.0, 0.05], seed=7)
    k = min(len(m1["trace"]), len(cap4["trace"]), 5)
    assert [e["J"] for e in m1["trace"][:k]] == [e["J"] for e in cap4["trace"][:k]]
    assert cap4["method"] == "CAP:4"
    r0 = methods_ext.run_anytime_ext(problem, witness, "CAP:0", [0.0, 0.2], seed=7)
    assert r0["stats"]["repair_attempts"] == 0, "CAP:0 must never repair"

    sw = methods_ext.run_anytime_ext(problem, witness, "SW:fixed0.5", budgets, seed=1)
    check_rows(problem, sw, budgets)
    schedule = sw["stats"]["schedule"]
    assert schedule, "schedule must be recorded"
    for entry in schedule:
        if entry["start_s"] >= 0.5:
            assert entry["mode"] == "M0", entry
        else:
            assert entry["mode"] == "M1" and entry["cap"] == 4, entry
    assert sw["stats"]["first_M0_start_s"] is not None

    inter = methods_ext.run_anytime_ext(problem, witness, "INTERLEAVE", budgets, seed=2)
    check_rows(problem, inter, budgets)
    modes = [s["mode"] for s in inter["stats"]["schedule"]]
    assert modes[:4] == ["M1", "M0", "M1", "M0"][:len(modes[:4])]

    sched = methods_ext.run_anytime_ext(problem, witness, "CAPSCHED:1", budgets, seed=3)
    check_rows(problem, sched, budgets)
    levels = [(s["mode"], s["cap"]) for s in sched["stats"]["schedule"]]
    order = {lvl: i for i, lvl in enumerate(methods_ext.CAPSCHED_LEVELS)}
    ranks = [order[l] for l in levels]
    assert ranks == sorted(ranks), "cap schedule must be nondecreasing"

    stag = methods_ext.run_anytime_ext(problem, witness, "SW:stag2", budgets, seed=4)
    check_rows(problem, stag, budgets)
    modes = [s["mode"] for s in stag["stats"]["schedule"]]
    if "M0" in modes:
        first = modes.index("M0")
        assert all(m == "M0" for m in modes[first:]), "stagnation switch is permanent"
        assert all(m == "M1" for m in modes[:first])

    bandit = methods_ext.run_anytime_ext(problem, witness, "BANDIT:8", budgets, seed=5)
    check_rows(problem, bandit, budgets)
    arms = {(s["mode"], s["cap"]) for s in bandit["stats"]["schedule"]}
    if len(bandit["stats"]["schedule"]) >= 4:
        assert arms == set(methods_ext.BANDIT_ARMS), arms

    bandit2 = methods_ext.run_anytime_ext(problem, witness, "BANDIT2:8", budgets, seed=6)
    check_rows(problem, bandit2, budgets)
    arms2 = {(s["mode"], s["cap"]) for s in bandit2["stats"]["schedule"]}
    assert arms2 <= set(methods_ext.BANDIT2_ARMS), arms2
    assert methods_ext.parse_method("SWL:swl_v1")["kind"] == "SWL"
    for token, arms in (("BANDITP:8", methods_ext.BANDITP_ARMS), ("BANDITP3:8", methods_ext.BANDITP3_ARMS)):
        port = methods_ext.run_anytime_ext(problem, witness, token, budgets, seed=12)
        check_rows(problem, port, budgets)
        played = {(s["mode"], s["cap"]) for s in port["stats"]["schedule"]}
        if len(port["stats"]["schedule"]) >= len(arms):
            assert played == set(arms), (token, played)
        assert "ALNS" in port["stats"]["reconstructions_by_mode"], token
        assert port["stats"]["alns_iterations"] > 0, token
        assert "adaptive-repair" in port["stats"]["verified_events_by_branch"], token
    # The matched selectors share the verified incumbent interface.
    for token in ("RR3", "RND3"):
        port = methods_ext.run_anytime_ext(problem, witness, token, budgets, seed=12)
        check_rows(problem, port, budgets)
        played = [(s["mode"], s["cap"]) for s in port["stats"]["schedule"]]
        assert played and set(played) <= set(methods_ext.BANDITP3_ARMS)
        if token == "RR3":
            assert all(arm == methods_ext.BANDITP3_ARMS[i % 3]
                       for i, arm in enumerate(played)), played
    assert methods_ext.parse_method("BANDITW:priors_v1")["kind"] == "BANDITW"
    from pathlib import Path as _P
    if (_P(HERE).parent / "models" / "priors_v1.pkl").exists():
        warm = methods_ext.run_anytime_ext(problem, witness, "BANDITW:priors_v1", budgets, seed=8)
        check_rows(problem, warm, budgets)
        assert len(warm["stats"]["prior_rewards"]) == len(methods_ext.BANDIT_ARMS)
    if (_P(HERE).parent / "models" / "swl_v1.pkl").exists():
        swl = methods_ext.run_anytime_ext(problem, witness, "SWL:swl_v1", budgets, seed=8)
        check_rows(problem, swl, budgets)
        assert 0 < swl["stats"]["predicted_tau_s"] <= 60

    # Same seed gives the same first verified event (deadline timing aside).
    a = methods_ext.run_anytime_ext(problem, witness, "BANDIT:8", [0.0, 0.05], seed=9)
    b = methods_ext.run_anytime_ext(problem, witness, "BANDIT:8", [0.0, 0.05], seed=9)
    assert a["trace"][1]["J"] == b["trace"][1]["J"]
    print("EXTENDED METHOD CHECKS PASSED")


if __name__ == "__main__":
    main()
