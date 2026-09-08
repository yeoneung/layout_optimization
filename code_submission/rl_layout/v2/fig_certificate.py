"""
The two figures the certificate story needs.

`fig_cert_ladder`  what each certificate buys, and what it costs.  The left panel
is feasibility against packing density for the ladder of certificates; the right
places every certificate on a cost/feasibility plane, which is where the shape of
the result lives -- the witness-compatible rollout certificate sits up and to the
left of a tighter test that costs an order of magnitude more and delivers less.

`fig_gallery`  what stranding actually looks like.  A scalar objective hides the
failure mode: the same instance, solved by the same constructor under different
certificates, either completes or leaves facilities piled on top of one another.
Overlapping facilities are outlined in red so the failure is visible rather than
inferred from a number.
"""

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
import numpy as np                                             # noqa: E402
from matplotlib.patches import Rectangle                       # noqa: E402

import bench                                                   # noqa: E402
import core                                                    # noqa: E402
import figures as F                                            # noqa: E402
from viability import ViabilityConstructor                     # noqa: E402

CERTS = [
    ("no certificate", dict(filter_mode="none"), F.MUTED, "x", ":"),
    ("first fit", dict(filter_mode="pack"), F.C[1], "s", "-"),
    ("randomized oracle, 32 restarts", dict(filter_mode="oracle", oracle_tries=32),
     F.C[2], "^", "-"),
    ("rollout of the base heuristic", dict(filter_mode="rollout"), F.C[0], "o", "-"),
]


def sweep(fills, instances, n_rooms=32, n_cand=32, split="test"):
    out = {}
    for label, kw, col, mk, ls in CERTS:
        feas, obj, ms = [], [], []
        for fl in fills:
            cell = bench.suite_cell(split, n_rooms, fl, instances)
            inst = cell["inst"]
            t0 = time.time()
            f, j = [], []
            for b in range(inst.B):
                r = ViabilityConstructor(inst.take(np.array([b])), n_cand=n_cand,
                                         room_rule="area", **kw).solve()
                f.append(r["feasible"])
                j.append(r["best"])
            feas.append(float(np.mean(f)))
            obj.append(float(np.mean(j)))
            ms.append((time.time() - t0) / inst.B * 1000)
            print(f"  {label:32s} fill={fl:.2f} feas={feas[-1]:4.2f} "
                  f"{ms[-1]:7.1f} ms", flush=True)
        out[label] = dict(feas=feas, obj=obj, ms=ms, col=col, mk=mk, ls=ls)
    return out


def fig_cert_ladder(fills, res, name="fig_certificate"):
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.9))
    ax = axes[0]
    for label, d in res.items():
        ax.plot(fills, d["feas"], marker=d["mk"], linestyle=d["ls"], color=d["col"],
                markersize=7, markeredgecolor=F.SURF, markeredgewidth=1.2,
                label=label, zorder=3)
    ax.set_xlabel("fill ratio")
    ax.set_ylabel("fraction of instances completed feasibly")
    ax.set_xticks(fills)
    ax.set_ylim(-0.04, 1.06)
    ax.set_title("What each certificate admits", loc="left", fontsize=9.5,
                 fontweight="bold", pad=8)
    ax.legend(loc="lower left", fontsize=7.5)

    # cost / feasibility plane at the densest cell, where they separate
    ax = axes[1]
    k = -1
    costs = [d["ms"][k] for d in res.values()]
    mid = np.sqrt(min(costs) * max(costs))
    for label, d in res.items():
        # an unfilled 'x' marker rejects an edgecolor, so only fill it when set
        kw = {} if d["mk"] == "x" else dict(edgecolor=F.SURF, linewidth=1.2)
        ax.scatter(d["ms"][k], d["feas"][k], s=90, color=d["col"], marker=d["mk"],
                   zorder=3, **kw)
        # label toward the inside of the plot so the rightmost point stays on it
        right = d["ms"][k] > mid
        ax.annotate(label, (d["ms"][k], d["feas"][k]), textcoords="offset points",
                    xytext=(-10 if right else 10, 8), fontsize=7.5, color=d["col"],
                    fontweight="bold", ha="right" if right else "left")
    ax.set_xscale("log")
    ax.set_xlim(min(costs) * 0.45, max(costs) * 2.2)
    ax.set_xlabel("certificate cost  (ms per layout, log scale)")
    ax.set_ylabel("fraction completed feasibly")
    ax.set_ylim(-0.04, 1.06)
    ax.set_title(f"Cost against benefit at fill {fills[k]:.2f}", loc="left",
                 fontsize=9.5, fontweight="bold", pad=8)
    F.save(fig, name)


def draw(ax, inst, x, y, b, title):
    """Layout panel with overlapping facilities outlined in red."""
    gw, gh = inst.gw[b], inst.gh[b]
    obj = core.BatchObjective(inst)
    ovm = obj.overlap_matrix(x, y)[b]
    bad = ovm.sum(axis=1) > 1e-9
    ax.add_patch(Rectangle((0, 0), gw, gh, fill=False, edgecolor=F.AXIS, lw=1.4))
    for i in range(inst.n):
        w, h = inst.w[b, i], inst.h[b, i]
        col = F.TYPE_COLORS[int(inst.type_id[b, i]) % len(F.TYPE_COLORS)]
        ax.add_patch(Rectangle((x[b, i] - w / 2, y[b, i] - h / 2), w, h,
                               facecolor=col, alpha=0.45 if bad[i] else 0.92,
                               edgecolor="#d21f3c" if bad[i] else F.SURF,
                               lw=1.8 if bad[i] else 1.0, zorder=4 if bad[i] else 3))
    ax.set_xlim(-0.6, gw + 0.6)
    ax.set_ylim(-0.6, gh + 0.6)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=8.5, color=F.INK2, pad=4)


def fig_gallery(fill, instances, pick, n_rooms=32, n_cand=32, split="test",
                name="fig_gallery"):
    cell = bench.suite_cell(split, n_rooms, fill, instances)
    inst = cell["inst"]
    one = inst.take(np.array([pick]))
    obj = core.BatchObjective(one)
    fig, axes = plt.subplots(1, len(CERTS), figsize=(3.0 * len(CERTS), 2.9))
    for ax, (label, kw, _, _, _) in zip(axes, CERTS):
        r = ViabilityConstructor(one, n_cand=n_cand, room_rule="area", **kw).solve()
        tot, c = obj.evaluate(r["x"], r["y"], components=True)
        ov = float(c["overlap_area"][0])
        tag = "feasible" if ov <= 1e-9 else f"overlap {ov:.0f}"
        draw(ax, one, r["x"], r["y"], 0, f"{label}\n$J$={float(tot[0]):.0f}, {tag}")
    # No suptitle: the panels are axis-free, so a figure title leaves a large
    # gap once the bounding box is tightened.  The caption carries the framing.
    fig.subplots_adjust(wspace=0.04)
    F.save(fig, name)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fills", nargs="+", type=float,
                   default=[0.90, 0.93, 0.95, 0.97])
    p.add_argument("--instances", type=int, default=20)
    p.add_argument("--gallery-fill", dest="gfill", type=float, default=0.95)
    p.add_argument("--gallery-pick", dest="gpick", type=int, default=0)
    a = p.parse_args()

    print("certificate sweep")
    res = sweep(a.fills, a.instances)
    fig_cert_ladder(a.fills, res)
    fig_gallery(a.gfill, a.instances, a.gpick)


if __name__ == "__main__":
    main()
