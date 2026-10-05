"""Learned constructive policy as a shared-incumbent anytime method (X17).

Token LRN:<checkpoint stem>, for example LRN:S64_seed1.  Each reconstruction is
one certified decode of an archive policy: the policy ranks candidate positions
and a first-fit completion certificate admits the first viable one (the learned
analogue of full regeneration, M0).  Decodes repeat until the deadline; the first
is greedy and later ones sample.  Every complete layout is independently
verified and scored before it can enter the anytime record, and the common
best-contact initial layout is the incumbent that no output can fall below.

The policies were trained on a 44 x 44 canvas, so this method applies to the
n = 32 and n = 64 cells (plate side at most 44).  Torch runs single-threaded on
the CPU core the worker is pinned to.
"""

import os
import time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import torch

torch.set_num_threads(1)

from certified_decode import FirstFitWitness  # noqa: E402
from completion_search import _empty_stats  # noqa: E402
from evaluate2 import load_construct  # noqa: E402
from layout_verifier import verify_complete_layout  # noqa: E402

ARCHIVE = Path(__file__).resolve().parents[2] / "code_submission" / "rl_layout" / "experiments"
CHECKPOINTS = ARCHIVE / "runs"
LOCAL_POLICIES = Path.home() / "layout_eaai_results" / "policies"   # policies trained for the EAAI track
CANVAS = 44
_MODEL_CACHE = {}


def checkpoint_path(checkpoint):
    for directory in (CHECKPOINTS, LOCAL_POLICIES):
        path = directory / (checkpoint + ".pt")
        if path.exists():
            return path
    raise FileNotFoundError(checkpoint + ".pt not found in the archive runs or the local policies directory")


@torch.no_grad()
def certified_decode_from_witness(model, nrm, env, dev, inst, initial_witness, n_cand=32,
                                  temperature=1.0, deterministic=False, seed=0, step_tries=1):
    """The archive certified decoder (certified_decode.run_certified_decode) for
    one instance, seeded with a verified complete layout instead of first-fit at
    the empty plate, so that every compared method shares the same initializer.
    The policy chooses the room and ranks positions; the first candidate whose
    successor state has a first-fit completion is committed; otherwise the
    carried witness position is used."""
    B, n = env.B, env.n
    if B != 1:
        raise ValueError("single-instance decoding only")
    obs, glob = env.reset()
    T = torch.as_tensor
    iw = np.rint(inst.w).astype(np.int64)
    ih = np.rint(inst.h).astype(np.int64)
    wit = FirstFitWitness(iw[0], ih[0], env.GH, env.GW, int(round(inst.gw[0])), int(round(inst.gh[0])))
    occ = np.zeros((env.GH, env.GW), dtype=np.int32)
    placed = np.zeros(n, dtype=bool)
    witness = [(int(i), int(r), int(c)) for i, r, c in initial_witness]
    if not wit.verify(occ, list(range(n)), witness):
        raise RuntimeError("initial witness is not a valid completion on the policy canvas")
    rng = np.random.default_rng(seed)
    n_witness_used = 0
    n_reject = 0
    for _ in range(n):
        tok, pooled, _, rcat = model.room_dist(T(nrm(obs), device=dev), T(glob, device=dev),
                                               T(~env.placed, device=dev))
        forced = env.forced_room()
        if forced is not None:
            room = T(forced, device=dev)
        elif deterministic:
            room = rcat.logits.argmax(-1)
        else:
            room = torch.distributions.Categorical(logits=rcat.logits / temperature).sample()
        room_np = room.cpu().numpy()
        P, legal, dead, fb = env.planes(room_np)
        logits = model.pos_logits(tok, pooled, room, T(P, device=dev), T(legal, device=dev),
                                  T(dead, device=dev))
        order = torch.argsort(logits, dim=-1, descending=True).cpu().numpy()
        i = int(room_np[0])
        rem = [j for j in range(n) if not placed[j] and j != i]
        wit_pos = next((r * env.GW + c for (j, r, c) in witness if j == i), None)
        chosen = None
        for k in range(min(n_cand, order.shape[1])):
            f = int(order[0, k])
            if not np.isfinite(logits[0, f].item()):
                break
            r, c = f // env.GW, f % env.GW
            if not wit._legal(occ, i)[r, c]:
                n_reject += 1
                continue
            trial = occ.copy()
            trial[r:r + ih[0, i], c:c + iw[0, i]] += 1
            ok, w = wit.certify(trial, rem, tries=step_tries, rng=rng)
            if ok:
                chosen = (f, trial, w)
                break
            n_reject += 1
        if chosen is None and wit_pos is not None:
            r, c = wit_pos // env.GW, wit_pos % env.GW
            trial = occ.copy()
            trial[r:r + ih[0, i], c:c + iw[0, i]] += 1
            w = [(j, wr, wc) for (j, wr, wc) in witness if j != i]
            if not wit.verify(trial, rem, w):
                raise RuntimeError("carried witness invariant failed for facility {}".format(i))
            chosen = (wit_pos, trial, w)
            n_witness_used += 1
        if chosen is None:
            raise RuntimeError("certified decoder lost its witness for facility {}".format(i))
        pos = np.array([chosen[0]], dtype=np.int64)
        occ, witness = chosen[1], chosen[2]
        placed[i] = True
        obs, glob, _, _ = env.step(room_np, pos)
    x, y, tot, c = env.result()
    return {"x": x, "y": y, "best": tot, "witness_used": n_witness_used, "rejects": n_reject}


def checkpoint_args(checkpoint):
    return torch.load(str(checkpoint_path(checkpoint)), map_location="cpu", weights_only=False)["args"]


def policy_canvas(checkpoint):
    """Canvas the policy was trained on (its coordinate planes are normalized by it)."""
    return int(checkpoint_args(checkpoint).get("canvas") or CANVAS)


def load_policy(checkpoint, inst, seed):
    key = (checkpoint, int(inst.n), float(inst.gw[0]), float(inst.gh[0]))
    if key not in _MODEL_CACHE:
        path = checkpoint_path(checkpoint)
        _MODEL_CACHE.clear()
        _MODEL_CACHE[key] = load_construct(str(path), inst, torch.device("cpu"),
                                           canvas=policy_canvas(checkpoint), seed=seed)
    return _MODEL_CACHE[key]


def _layout_from_centers(problem, x, y):
    layout = []
    for i in range(problem.n):
        col = int(round(x[0, i] - problem.inst.w[0, i] / 2.0))
        row = int(round(y[0, i] - problem.inst.h[0, i] / 2.0))
        layout.append((i, row, col))
    return layout


def run_learned_anytime(problem, initial_witness, checkpoint, budgets, seed=0,
                        n_cand=32, temperature=1.0, initial_value=None):
    budgets = sorted(float(b) for b in budgets)
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets must be nonnegative")
    canvas = policy_canvas(checkpoint)
    if problem.gw > canvas or problem.gh > canvas:
        raise ValueError("plate exceeds the policy canvas ({} > {})".format(max(problem.gw, problem.gh), canvas))
    inst = problem.inst
    trained_n = checkpoint_args(checkpoint).get("n_rooms")
    if trained_n is not None and int(trained_n) != int(problem.n):
        raise ValueError("policy {} was trained for n = {} and cannot decode n = {}".format(
            checkpoint, trained_n, problem.n))
    if initial_value is None:
        initial_value = problem.score(initial_witness)
    stats = _empty_stats()
    stats.update({"decodes": 0, "abstained_decodes": 0, "policy_rejects": 0,
                  "policy_witness_fallbacks": 0, "decode_seconds": 0.0,
                  "checkpoint": checkpoint})
    start = time.perf_counter()
    deadline = start + budgets[-1]
    model, nrm, env, _ = load_policy(checkpoint, inst, seed)
    load_seconds = time.perf_counter() - start
    stats["policy_load_seconds"] = load_seconds
    events = [(0.0, initial_value, list(initial_witness), "initial")]
    incumbent = list(initial_witness)
    best_value = initial_value
    iteration = 0
    seed_witness = [tuple(int(v) for v in item) for item in initial_witness]
    while time.perf_counter() < deadline:
        # Search seeds are 48-bit integers; keep the derived seeds within 31 bits.
        decode_seed = (int(seed) * 100003 + iteration) % 2147483647
        torch.manual_seed(decode_seed)
        decode_start = time.perf_counter()
        # Later decodes restart from the incumbent, as M0 and M1 do.
        out = certified_decode_from_witness(model, nrm, env, torch.device("cpu"), inst,
                                            incumbent if iteration else seed_witness,
                                            n_cand=n_cand, temperature=temperature,
                                            deterministic=(iteration == 0),
                                            seed=decode_seed)
        stats["decode_seconds"] += time.perf_counter() - decode_start
        stats["decodes"] += 1
        stats["policy_rejects"] += int(out["rejects"])
        stats["policy_witness_fallbacks"] += int(out["witness_used"])
        iteration += 1
        layout = _layout_from_centers(problem, out["x"], out["y"])
        now = time.perf_counter()
        if now >= deadline:
            stats["timeouts"] += 1
            break
        stats["verifier_calls"] += 1
        verify_start = time.perf_counter()
        check = verify_complete_layout(problem.iw, problem.ih, problem.gw, problem.gh, layout)
        stats["final_verification_seconds"] += time.perf_counter() - verify_start
        if not check.valid:
            stats["final_verification_failures"] += 1
            continue
        score_start = time.perf_counter()
        value = problem.score(layout, already_verified=True)
        stats["objective_seconds"] += time.perf_counter() - score_start
        completed = time.perf_counter()
        if completed > deadline:
            stats["late_completed_events"] += 1
            stats["timeouts"] += 1
            break
        events.append((completed - start, value, list(layout), "learned-certified"))
        counts = stats["verified_events_by_branch"]
        counts["learned-certified"] = counts.get("learned-certified", 0) + 1
        if value > best_value + 1e-12:
            gain = value - best_value
            best_value = value
            incumbent = list(layout)
            improvements = stats["incumbent_improvements_by_branch"]
            improvements["learned-certified"] = improvements.get("learned-certified", 0) + 1
            gains = stats["incumbent_gain_by_branch"]
            gains["learned-certified"] = gains.get("learned-certified", 0.0) + gain
    finished = time.perf_counter()
    stats["actual_search_elapsed_s"] = finished - start
    stats["overrun_s"] = max(0.0, finished - deadline)
    stats["max_admitted_event_s"] = max(event[0] for event in events)
    rows = []
    for budget in budgets:
        eligible = [(value, layout) for elapsed, value, layout, _ in events if elapsed <= budget + 1e-12]
        value, layout = max(eligible, key=lambda item: item[0])
        check = verify_complete_layout(problem.iw, problem.ih, problem.gw, problem.gh, layout)
        if not check.valid:
            raise RuntimeError("anytime incumbent failed final verification")
        rows.append({"budget_s": budget, "J": value, "improvement": value - initial_value,
                     "improvement_pct": (100.0 * (value - initial_value) / initial_value
                                         if initial_value > 0 else None),
                     "layout": [[int(i), int(r), int(c)] for i, r, c in layout]})
    stats["events"] = len(events)
    stats["iterations"] = iteration
    stats["mean_displaced"] = None
    stats["max_displaced"] = None
    stats["mean_destroy_size"] = None
    trace = [{"elapsed_s": float(elapsed), "J": float(value), "branch": branch}
             for elapsed, value, _, branch in events]
    return {"method": "LRN:" + checkpoint, "initial_J": initial_value, "rows": rows,
            "trace": trace, "stats": stats}
