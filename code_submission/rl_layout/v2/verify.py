"""
Chain-of-trust check for the v2 objective.

    comb_high.objective()  ==  layout_env.LayoutObjective  ==  v2.BatchObjective

The middle equality is already covered by ../test_objective.py.  This script
closes the right-hand one on both reference scenarios, on random layouts, on
grid-aligned layouts, and on layouts with heavy overlap, and additionally spot
checks v2 straight against the original pure-Python implementation so the
verification does not route through v1 alone.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from layout_env import build_spec, LayoutObjective, load_reference_module   # noqa: E402
import core                                                                # noqa: E402


def check_against_v1(scenario, flip, rng, B=64):
    spec = build_spec(scenario, flip_adj=flip)
    o1 = LayoutObjective(spec)
    inst = core.from_spec(spec, B)
    o2 = core.BatchObjective(inst)

    worst = 0.0
    for tag, mk in (
        ("random", lambda: (rng.uniform(o1.xmin, o1.xmax, (B, o1.n)),
                            rng.uniform(o1.ymin, o1.ymax, (B, o1.n)))),
        ("grid-aligned", lambda: (
            np.clip(np.rint(rng.uniform(o1.xmin, o1.xmax, (B, o1.n)) - o1.xmin) + o1.xmin,
                    o1.xmin, o1.xmax),
            np.clip(np.rint(rng.uniform(o1.ymin, o1.ymax, (B, o1.n)) - o1.ymin) + o1.ymin,
                    o1.ymin, o1.ymax))),
        ("piled up (heavy overlap)", lambda: (
            np.broadcast_to(((o1.xmin + o1.xmax) / 2)[None, :], (B, o1.n)).copy()
            + rng.normal(0, 0.3, (B, o1.n)),
            np.broadcast_to(((o1.ymin + o1.ymax) / 2)[None, :], (B, o1.n)).copy()
            + rng.normal(0, 0.3, (B, o1.n)))),
        ("wall-flush", lambda: (
            np.where(rng.random((B, o1.n)) < 0.5, o1.xmin, o1.xmax),
            np.where(rng.random((B, o1.n)) < 0.5, o1.ymin, o1.ymax))),
    ):
        x, y = mk()
        x, y = o1.clip(x, y)
        t1, c1 = o1.evaluate(x, y, components=True)
        t2, c2 = o2.evaluate(x, y, components=True)
        e = float(np.abs(t1 - t2).max())
        worst = max(worst, e)
        parts = "  ".join(f"{k}:{np.abs(c1[k]-c2[k]).max():.2e}" for k in c1)
        print(f"  {scenario:9s} flip={int(flip)} {tag:24s} max|dJ| = {e:.3e}   ({parts})")
    return worst


def check_against_original(scenario, rng, B=8):
    """v2 vs the original pure-Python objective(), no v1 in the path."""
    m = load_reference_module(scenario)
    spec = build_spec(scenario, flip_adj=False)
    inst = core.from_spec(spec, B)
    o2 = core.BatchObjective(inst)
    x = rng.uniform(o2.xmin, o2.xmax)
    y = rng.uniform(o2.ymin, o2.ymax)
    t2 = o2.evaluate(x, y)

    worst = 0.0
    for b in range(B):
        layout = {name: m.Room(float(x[b, i]), float(y[b, i]),
                               float(spec.w[i]), float(spec.h[i]))
                  for i, name in enumerate(spec.room_order)}
        t1, _ = m.objective(layout)
        worst = max(worst, abs(t1 - float(t2[b])))
    print(f"  {scenario:9s} vs original pure-Python objective()   max|dJ| = {worst:.3e}")
    return worst


def check_generator(rng, n_inst=256, n_rooms=32):
    inst = core.generate_instances(n_inst, n_rooms, rng)
    obj = core.BatchObjective(inst)
    fr = inst.fill_ratio()
    print(f"\n  generated {n_inst} instances x {n_rooms} rooms")
    print(f"    fill ratio      {fr.min():.2f} .. {fr.max():.2f}  (mean {fr.mean():.2f})")
    print(f"    boundary        {inst.gw.min():.0f}x{inst.gh.min():.0f}"
          f" .. {inst.gw.max():.0f}x{inst.gh.max():.0f}")
    print(f"    adj ceiling     {obj.adj_ceiling.min():.0f} .. {obj.adj_ceiling.max():.0f}")
    print(f"    edge ceiling    {obj.edge_ceiling.min():.0f} .. {obj.edge_ceiling.max():.0f}")
    x, y = core.random_layout(obj, rng)
    t = obj.evaluate(x, y)
    assert np.all(np.isfinite(t)), "non-finite objective from generated instances"
    # every room must be placeable
    assert np.all(inst.w <= inst.gw[:, None] + 1e-9)
    assert np.all(inst.h <= inst.gh[:, None] + 1e-9)
    print(f"    random-layout J {t.mean():.0f} +- {t.std():.0f}   (all finite)")
    # reference scenarios must lie inside the generated family's support
    for s in ("comb_high", "hospital"):
        sp = build_spec(s)
        r = float((sp.w * sp.h).sum() / (sp.grid_w * sp.grid_h))
        inside = fr.min() - 0.02 <= r <= fr.max() + 0.02
        print(f"    {s:10s} fill {r:.3f}  inside generated support: {inside}")


def main():
    rng = np.random.default_rng(0)
    print("v2 BatchObjective  vs  v1 LayoutObjective")
    worst = 0.0
    for scenario in ("comb_high", "hospital"):
        for flip in (False, True):
            worst = max(worst, check_against_v1(scenario, flip, rng))
    print(f"\n  worst disagreement over all cases: {worst:.3e}")
    assert worst < 1e-9, "v2 objective does not match v1"

    print("\nv2 BatchObjective  vs  original pure-Python objective()")
    worst2 = 0.0
    for scenario in ("comb_high", "hospital"):
        worst2 = max(worst2, check_against_original(scenario, rng))
    assert worst2 < 1e-6, "v2 objective does not match the original scripts"

    print("\nprocedural instance generator")
    check_generator(rng)
    print("\nOK - v2 objective is the same function the SA baseline optimizes.")


if __name__ == "__main__":
    main()
