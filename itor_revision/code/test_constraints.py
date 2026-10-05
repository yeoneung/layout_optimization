"""Checks for obstacles and dock anchors: feasibility by construction, objective
consistency, obstacle avoidance of every stored layout, and all method families."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code_submission" / "rl_layout" / "experiments"))
sys.path.insert(0, str(HERE))

import bench  # noqa: E402
from certified_decode import FirstFitWitness  # noqa: E402
from repair_baseline import run_adaptive_repair  # noqa: E402
import constraints  # noqa: E402
import flow_objective  # noqa: E402
import methods_ext  # noqa: E402


def brute_cost(problem, layout):
    pos = {i: (c + problem.inst.w[0, i] / 2.0, r + problem.inst.h[0, i] / 2.0) for i, r, c in layout}
    keys = sorted(pos)
    total = 0.0
    for a, i in enumerate(keys):
        for j in keys[a + 1:]:
            total += problem.flow[i, j] * (abs(pos[i][0] - pos[j][0]) + abs(pos[i][1] - pos[j][1]))
    for anchor in problem.anchors:
        for i in keys:
            total += anchor["flow"][i] * (abs(pos[i][0] - anchor["x"]) + abs(pos[i][1] - anchor["y"]))
    return total


def main():
    for n, fill in ((32, 0.90), (64, 0.90)):
        cell = bench.suite_cell("val", n, fill, 2, tag="eaai_unit_test", max_grid=44)
        inst = cell["inst"].take([0])
        source = cell["witness"][0]
        canvas = max(int(round(inst.gw[0])), int(round(inst.gh[0])))
        flow = flow_objective.generate_flow(n, seed=5)
        blocked, anchors, info = constraints.make_constraints(inst, source, canvas, seed=11)
        assert info["blocked_cells"] > 0 and info["dock"] is not None, info
        # the withheld source packing stays feasible with the obstacles in place
        gen = FirstFitWitness(np.rint(inst.w[0]).astype(int), np.rint(inst.h[0]).astype(int), canvas, canvas,
                              int(round(inst.gw[0])), int(round(inst.gh[0])))
        witness = [(i, int(round(source[i, 1])), int(round(source[i, 0]))) for i in range(n)]
        assert gen.verify(blocked.astype(np.int32), list(range(n)), witness), "source packing must remain feasible"
        problem = constraints.ConstrainedFlowLayoutProblem(inst, flow, blocked, anchors)
        ok, w0, _ = problem.initial_best_contact()
        print("n={} fill={} free {} blocked {} dock {} effective fill {:.3f} init {}".format(
            n, fill, info["free_cells"], info["blocked_cells"], info["dock"], info["effective_fill"], ok))
        if not ok:
            continue
        assert problem.verifies_obstacles(w0)
        assert abs(problem.cost_of(w0) - brute_cost(problem, w0)) < 1e-9
        # marginal gain includes the dock term
        k = int(w0[-1][0])
        prefix = [item for item in w0 if item[0] != k]
        placed = np.zeros((1, n), dtype=bool)
        x = np.zeros((1, n)); y = np.zeros((1, n))
        for j, r, c in prefix:
            placed[0, j] = True
            x[0, j] = c + inst.w[0, j] / 2.0
            y[0, j] = r + inst.h[0, j] / 2.0
        gain = problem.greedy._gain(k, x, y, placed)[0]
        rk, ck = [(r, c) for i, r, c in w0 if i == k][0]
        assert abs(gain[rk, ck] + (brute_cost(problem, w0) - brute_cost(problem, prefix))) < 1e-9
        constraints.activate()
        try:
            budgets = [0.0, 0.5, 1.5]
            for token in ("M0", "M1", "CAP:8", "INTERLEAVE", "BANDIT:8", "BANDIT2:8"):
                out = methods_ext.run_anytime_ext(problem, w0, token, budgets, seed=3)
                for row in out["rows"]:
                    assert problem.verifies_obstacles(row["layout"]), token + " placed a facility on an obstacle"
                    assert abs(problem.score(row["layout"], already_verified=True) - row["J"]) < 1e-8
                print("  {:<10s} cost {:.1f} -> {:.1f}".format(token, problem.cost_of(w0), problem.cost_of(out["rows"][-1]["layout"])))
            out = run_adaptive_repair(problem, w0, budgets, 5, "small_beam", initial_value=problem.score(w0))
            for row in out["rows"]:
                assert problem.verifies_obstacles(row["layout"]), "ALNS placed a facility on an obstacle"
            print("  {:<10s} cost {:.1f} -> {:.1f}".format("ALNS", problem.cost_of(w0), problem.cost_of(out["rows"][-1]["layout"])))
        finally:
            constraints.deactivate()
    print("CONSTRAINT CHECKS PASSED")


if __name__ == "__main__":
    main()
