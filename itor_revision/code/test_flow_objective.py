"""Checks for the flow objective: brute-force agreement, exact marginal gains,
ALNS contribution consistency, and short runs of every method family."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code_submission" / "rl_layout" / "experiments"))
sys.path.insert(0, str(HERE))

import bench  # noqa: E402
from layout_verifier import verify_complete_layout  # noqa: E402
from repair_baseline import run_adaptive_repair  # noqa: E402
import flow_objective  # noqa: E402
import methods_ext  # noqa: E402


def brute_cost(problem, layout):
    """Flow cost over the pairs present in the (possibly partial) layout."""
    total = 0.0
    pos = {i: (c + problem.inst.w[0, i] / 2.0, r + problem.inst.h[0, i] / 2.0) for i, r, c in layout}
    keys = sorted(pos)
    for a, i in enumerate(keys):
        for j in keys[a + 1:]:
            total += problem.flow[i, j] * (abs(pos[i][0] - pos[j][0]) + abs(pos[i][1] - pos[j][1]))
    return total


def main():
    cell = bench.suite_cell("val", 32, 0.90, 1, tag="eaai_unit_test", max_grid=44)
    inst = cell["inst"].take([0])
    flow = flow_objective.generate_flow(inst.n, seed=123)
    assert np.allclose(flow, flow.T) and np.all(np.diag(flow) == 0) and (flow >= 0).all()
    assert 0.15 < np.mean(flow[np.triu_indices(inst.n, 1)] > 0) < 0.45
    problem = flow_objective.FlowLayoutProblem(inst, flow)
    ok, witness, _ = problem.initial_best_contact()
    assert ok
    # 1. evaluate agrees with brute force and J = C - cost
    cost = brute_cost(problem, witness)
    assert abs(problem.cost_of(witness) - cost) < 1e-9
    assert abs(problem.score(witness) - (problem.objective.constant[0] - cost)) < 1e-9
    assert problem.score(witness) > 0
    # 2. exact marginal: placing facility k last into the prefix of all others changes J by gain[k at its position]
    k = int(witness[-1][0])
    prefix = [item for item in witness if item[0] != k]
    occ = problem.occupancy(prefix)[None, :, :]
    placed = np.zeros((1, problem.n), dtype=bool)
    x = np.zeros((1, problem.n)); y = np.zeros((1, problem.n))
    for j, r, c in prefix:
        placed[0, j] = True
        x[0, j] = c + problem.inst.w[0, j] / 2.0
        y[0, j] = r + problem.inst.h[0, j] / 2.0
    gain = problem.greedy._gain(k, x, y, placed)[0]
    rk, ck = [(r, c) for i, r, c in witness if i == k][0]
    partial_cost = brute_cost(problem, prefix)  # flows among prefix only
    full_cost = brute_cost(problem, witness)
    assert abs(gain[rk, ck] - (-(full_cost - partial_cost))) < 1e-9, "marginal gain must equal the added cost with opposite sign"
    # 3. ALNS contributions sum to twice the negative cost
    contrib = flow_objective.flow_facility_contributions(problem, witness)
    assert abs(contrib.sum() + 2.0 * cost) < 1e-9
    # 4. every method family runs, improves or keeps the incumbent, and stores verified layouts
    flow_objective.activate()
    try:
        budgets = [0.0, 0.5, 1.5]
        for token in ("M0", "M1", "CAP:8", "INTERLEAVE", "BANDIT:8", "BANDIT2:8"):
            out = methods_ext.run_anytime_ext(problem, witness, token, budgets, seed=3)
            for row in out["rows"]:
                check = verify_complete_layout(problem.iw, problem.ih, problem.gw, problem.gh, row["layout"])
                assert check.valid, check.reason
                assert abs(problem.score(row["layout"], already_verified=True) - row["J"]) < 1e-8
            assert out["rows"][-1]["J"] >= out["initial_J"] - 1e-9
            print("{:<10s} cost {:.1f} -> {:.1f}".format(token, problem.cost_of(witness), problem.cost_of(out["rows"][-1]["layout"])))
        out = run_adaptive_repair(problem, witness, budgets, 5, "small_beam", initial_value=problem.score(witness))
        assert out["rows"][-1]["J"] >= out["initial_J"] - 1e-9
        print("{:<10s} cost {:.1f} -> {:.1f}".format("ALNS", problem.cost_of(witness), problem.cost_of(out["rows"][-1]["layout"])))
    finally:
        flow_objective.deactivate()
    print("FLOW OBJECTIVE CHECKS PASSED")


if __name__ == "__main__":
    main()
