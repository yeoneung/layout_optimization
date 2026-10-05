"""Feasibility check for the learned constructor wrapper (runs on E-cores)."""
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "code_submission" / "rl_layout" / "experiments"))
sys.path.insert(0, str(HERE))

import psutil  # noqa: E402

# Pinning is optional for correctness checks; keep this portable.
if hasattr(psutil.Process(), "cpu_affinity"):
    allowed = psutil.Process().cpu_affinity()
    if 17 in allowed:
        psutil.Process().cpu_affinity([17])

import bench  # noqa: E402
from completion_search import LayoutProblem, run_anytime  # noqa: E402
from layout_verifier import verify_complete_layout  # noqa: E402
import learned_methods  # noqa: E402


def main():
    for n, fill, ckpt in ((32, 0.95, "S_dord_seed1"), (64, 0.95, "S64_seed1"), (64, 0.90, "S64_seed1")):
        cell = bench.suite_cell("val", n, fill, 2, tag="eaai_unit_test", max_grid=44)
        problem = LayoutProblem(cell["inst"].take([0]))
        ok, witness, _ = problem.initial_best_contact()
        assert ok
        budgets = [0.0, 1.0, 3.0, 6.0]
        t = time.perf_counter()
        out = learned_methods.run_learned_anytime(problem, witness, ckpt, budgets, seed=281474976710655)  # 48-bit seed
        wall = time.perf_counter() - t
        for row in out["rows"]:
            check = verify_complete_layout(problem.iw, problem.ih, problem.gw, problem.gh, row["layout"])
            assert check.valid, check.reason
            assert abs(problem.score(row["layout"], already_verified=True) - row["J"]) < 1e-8
        st = out["stats"]
        print("n={} fill={} {}: decodes={} abstained={} load={:.2f}s decode_total={:.2f}s wall={:.1f}s "
              "imp@1={:.2f}% imp@3={:.2f}% imp@6={:.2f}% rejects={} fallbacks={} verify_fail={}".format(
                  n, fill, ckpt, st["decodes"], st["abstained_decodes"], st["policy_load_seconds"],
                  st["decode_seconds"], wall, out["rows"][1]["improvement_pct"], out["rows"][2]["improvement_pct"],
                  out["rows"][3]["improvement_pct"], st["policy_rejects"], st["policy_witness_fallbacks"],
                  st["final_verification_failures"]), flush=True)
        m0 = run_anytime(problem, witness, "M0", budgets, seed=1, kappa=32, repair_cap=4)
        print("   M0 same core: imp@1={:.2f}% imp@3={:.2f}% imp@6={:.2f}% reconstructions={}".format(
            m0["rows"][1]["improvement_pct"], m0["rows"][2]["improvement_pct"],
            m0["rows"][3]["improvement_pct"], m0["stats"]["reconstructions"]), flush=True)
    print("LEARNED METHOD CHECKS PASSED")


if __name__ == "__main__":
    main()
