"""
Evaluation shared by every track.

Rules kept constant across methods:
  * identical objective (verify.py),
  * results reported per objective evaluation *and* per second, because the two
    tell different stories and only quoting one of them is how this literature
    usually goes wrong,
  * RL training cost is never hidden -- it is reported separately and folded into
    the amortized-cost curve,
  * the polish that SA gets is offered to RL too, and vice versa, so no method
    wins on a post-processing step the other was denied.
"""

import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import core                                                    # noqa: E402
import baselines2 as B2                                        # noqa: E402
from construct_env import ConstructEnv, N_PLANES, N_GLOB_C     # noqa: E402
from improve_env import ImproveEnv, N_GLOB                     # noqa: E402
from nets import ConstructPolicy, ImprovePolicy                # noqa: E402
from train_improve import RunningNorm, OP_LOOKUP               # noqa: E402


# ---------------------------------------------------------------------------
def load_construct(path, inst, dev, canvas="auto", resample=None, seed=0):
    ck = torch.load(path, map_location=dev, weights_only=False)
    if canvas == "auto":
        # the canvas fixes the coordinate planes' normalization, so evaluation
        # must use the canvas the policy was trained on
        canvas = ck["args"].get("canvas") if ck["args"].get("instances") == "random" else None
    env = ConstructEnv(inst, seed=seed, canvas=canvas, resample=resample,
                       order=ck["args"].get("order", "policy"))
    m = ConstructPolicy(env.feat_dim, N_GLOB_C, N_PLANES,
                        d_model=ck["args"]["d_model"], n_blocks=ck["args"]["n_blocks"],
                        ch=ck["args"]["ch"]).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    nrm = RunningNorm(env.feat_dim)
    nrm.load(ck["norm"])
    return m, nrm, env, ck["args"]


def load_improve(path, inst, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    env = ImproveEnv(inst, horizon=a["horizon"], action_mode=a["action_mode"],
                     delta_max=a["delta_max"], reward=a["reward"], accept=a["accept"])
    m = ImprovePolicy(env.obj.feat_dim, N_GLOB, d_model=a["d_model"],
                      n_blocks=a["n_blocks"], ops=tuple(ck["ops"])).to(dev)
    m.load_state_dict(ck["model"])
    m.eval()
    nrm = RunningNorm(env.obj.feat_dim)
    nrm.load(ck["norm"])
    return m, nrm, env, a


# ---------------------------------------------------------------------------
@torch.no_grad()
def run_construct(model, nrm, env, dev, deterministic=False, temperature=1.0,
                  new_instance=False):
    """One layout per instance in the batch.  n objective evaluations each."""
    t0 = time.time()
    obs, glob = env.reset(new_instance=new_instance)
    T = torch.as_tensor
    for _ in range(env.n):
        cache = {}

        def planes_fn(room_np):
            P, legal, dead, fb = env.planes(room_np)
            cache.update(P=P, legal=legal, dead=dead, fb=fb)
            return T(P, device=dev), T(legal, device=dev), T(dead, device=dev)

        fr = env.forced_room()
        a = model.act(T(nrm(obs), device=dev), T(glob, device=dev),
                      T(~env.placed, device=dev), planes_fn,
                      deterministic=deterministic, temperature=temperature,
                      forced_room=None if fr is None else T(fr, device=dev))
        room = a["room"].cpu().numpy()
        pos = a["pos"].cpu().numpy()
        if cache["dead"].any():
            pos = np.where(cache["dead"], cache["fb"].reshape(env.B, -1).argmax(axis=1), pos)
        obs, glob, _, _ = env.step(room, pos)
    x, y, tot, c = env.result()
    return {"x": x, "y": y, "best": tot, "overlap_area": c["overlap_area"],
            "adj": c["adj"], "edge": c["edge"],
            "evals": env.n * env.B, "wall_time": time.time() - t0}


@torch.no_grad()
def run_improve(model, nrm, env, dev, horizon, x0=None, y0=None, deterministic=False,
                trace_every=0):
    t0 = time.time()
    env.horizon = horizon
    obs, glob = env.reset(x0, y0)
    T = torch.as_tensor
    trace = []
    for t in range(horizon):
        a = model.act(T(nrm(obs), device=dev), T(glob, device=dev),
                      deterministic=deterministic)
        act = {k: a[k].cpu().numpy() for k in ("room", "op", "swap", "delta", "jump")}
        obs, glob, _, _ = env.step(act)
        if trace_every and (t + 1) % trace_every == 0:
            tot, _ = env.exact_best()
            trace.append({"evals_per_inst": t + 1, "best_mean": float(tot.mean()),
                          "wall": time.time() - t0})
    tot, c = env.exact_best()
    return {"x": env.best_x.copy(), "y": env.best_y.copy(), "best": tot,
            "overlap_area": c["overlap_area"], "adj": c["adj"], "edge": c["edge"],
            "evals": horizon * env.B, "wall_time": time.time() - t0, "trace": trace}


def best_of_k(runs):
    """Fold K independent samples per instance into one best-of-K result."""
    best = np.stack([r["best"] for r in runs])            # (K, B)
    k = best.argmax(axis=0)
    b = np.arange(best.shape[1])
    return {"best": best.max(axis=0),
            "x": np.stack([r["x"] for r in runs])[k, b],
            "y": np.stack([r["y"] for r in runs])[k, b],
            "overlap_area": np.stack([r["overlap_area"] for r in runs])[k, b],
            "adj": np.stack([r["adj"] for r in runs])[k, b],
            "edge": np.stack([r["edge"] for r in runs])[k, b],
            "evals": sum(r["evals"] for r in runs),
            "wall_time": sum(r["wall_time"] for r in runs)}


# ---------------------------------------------------------------------------
def type_maps(inst, x, y):
    """(B, GH, GW) map of room-type id per cell; -1 empty.  Permutation invariant
    within a type, so relabelling two identical rooms is not counted as variety."""
    B, n = x.shape
    GW, GH = int(inst.gw.max()), int(inst.gh.max())
    maps = np.full((B, GH, GW), -1, dtype=np.int16)
    for b in range(B):
        for i in range(n):
            x0 = int(round(x[b, i] - inst.w[b, i] / 2))
            y0 = int(round(y[b, i] - inst.h[b, i] / 2))
            maps[b, max(0, y0):y0 + int(inst.h[b, i]),
                 max(0, x0):x0 + int(inst.w[b, i])] = inst.type_id[b, i]
    return maps


def diversity(inst, x, y, max_pairs=400, rng=None):
    m = type_maps(inst, x, y).reshape(x.shape[0], -1)
    K = m.shape[0]
    pairs = [(i, j) for i in range(K) for j in range(i + 1, K)]
    if len(pairs) > max_pairs:
        rng = rng or np.random.default_rng(0)
        pairs = [pairs[t] for t in rng.choice(len(pairs), max_pairs, replace=False)]
    return float(np.mean([np.mean(m[i] != m[j]) for i, j in pairs]))


def summarize(name, obj, x, y, evals, wall, n_inst, extra=None):
    tot, c = obj.evaluate(x, y, components=True)
    row = {"method": name, "mean": float(tot.mean()), "std": float(tot.std()),
           "median": float(np.median(tot)), "max": float(tot.max()),
           "min": float(tot.min()),
           "feasible_rate": float(np.mean(c["overlap_area"] <= 1e-9)),
           "adj": float(c["adj"].mean()), "edge": float(c["edge"].mean()),
           "evals_per_inst": float(evals / n_inst),
           "wall_s": float(wall), "ms_per_layout": float(wall / n_inst * 1000)}
    if extra:
        row.update(extra)
    return row


def print_row(r):
    print(f"{r['method']:38s} J={r['mean']:8.1f}+-{r['std']:5.1f} med={r['median']:8.1f} "
          f"max={r['max']:8.1f} feas={r['feasible_rate']:4.2f} "
          f"adj={r['adj']:7.1f} edge={r['edge']:5.1f} "
          f"ev/inst={r['evals_per_inst']:8.0f} {r['ms_per_layout']:8.1f} ms/layout")
