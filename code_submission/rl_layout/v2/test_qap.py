"""
Correctness gate for qap_bench.py.

The delta formulas are where QAP implementations go wrong (asymmetric matrices,
diagonals, the k = u,v correction terms), so every one of them is checked
against brute-force full evaluation on random asymmetric instances with
non-zero diagonals -- strictly harder than anything in the benchmark set.
"""

import numpy as np

import qap_bench as Q


def rand_inst(n, seed):
    rng = np.random.default_rng(seed)
    return {"name": f"rand{n}", "n": n, "opt": None,
            "A": rng.integers(0, 9, (n, n)).astype(float),
            "B": rng.integers(0, 9, (n, n)).astype(float)}


def test_role_exchange():
    inst = rand_inst(7, 0)
    p = np.random.default_rng(1).permutation(7)
    pinv = np.argsort(p)
    sw = {"name": "", "n": 7, "opt": None, "A": inst["B"], "B": inst["A"]}
    assert abs(Q.cost(inst, p) - Q.cost(sw, pinv)) < 1e-9
    print("role exchange           OK")


def test_swap_delta():
    inst = rand_inst(8, 2)
    rng = np.random.default_rng(3)
    worst = 0.0
    for _ in range(200):
        p = rng.permutation(8)
        Bp = inst["B"][np.ix_(p, p)]
        u, v = rng.choice(8, 2, replace=False)
        d = Q.swap_delta(inst, p, Bp, u, v, Q.Meter(8))
        q = p.copy()
        q[[u, v]] = q[[v, u]]
        worst = max(worst, abs(d - (Q.cost(inst, q) - Q.cost(inst, p))))
    assert worst < 1e-9, worst
    print(f"swap delta              OK (worst {worst:.1e})")


def test_all_deltas():
    for n, seed in [(6, 4), (9, 5), (13, 6)]:
        inst = rand_inst(n, seed)
        p = np.random.default_rng(seed + 10).permutation(n)
        Bp = inst["B"][np.ix_(p, p)]
        D = Q.all_deltas(inst, p, Bp)
        c0 = Q.cost(inst, p)
        worst = 0.0
        for u in range(n):
            for v in range(u + 1, n):
                q = p.copy()
                q[[u, v]] = q[[v, u]]
                worst = max(worst, abs(D[u, v] - (Q.cost(inst, q) - c0)))
        assert worst < 1e-9, (n, worst)
    print(f"all-pairs deltas        OK (worst {worst:.1e})")


def test_marginal_sums_to_cost():
    inst = rand_inst(9, 7)
    rng = np.random.default_rng(8)
    for _ in range(20):
        order = rng.permutation(9)
        locs = rng.permutation(9)
        pf, pl, tot = [], [], 0.0
        for i, l in zip(order, locs):
            m = Q.marginal(inst, pf, pl, int(i), [int(l)], Q.Meter(9))
            tot += float(m[0])
            pf.append(int(i))
            pl.append(int(l))
        p = np.empty(9, dtype=int)
        p[pf] = pl
        assert abs(tot - Q.cost(inst, p)) < 1e-9
    print("marginal telescopes     OK")


def test_beam1_equals_greedy():
    for seed in range(5):
        inst = rand_inst(10, 20 + seed)
        pg = Q.greedy(inst, Q.Meter(10))
        pb = Q.beam(inst, 1, Q.Meter(10))
        assert np.array_equal(pg, pb)
    print("beam-1 == greedy        OK")


def test_loader():
    inst = Q.load("../../benchmarks/qaplib/nug12.dat")
    assert inst["n"] == 12 and inst["opt"] == 578
    assert inst["A"].min() >= 0 and inst["B"].min() >= 0
    print("loader                  OK (nug12, opt 578)")


if __name__ == "__main__":
    test_role_exchange()
    test_swap_delta()
    test_all_deltas()
    test_marginal_sums_to_cost()
    test_beam1_equals_greedy()
    test_loader()
    print("\nall QAP tests passed")
