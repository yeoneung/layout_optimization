"""Shared-incumbent controls and completion-aware search for dense layout.

All methods in this module start from the same verified complete layout.  They
keep that layout as an incumbent and use the independent final verifier before
an objective value can enter the anytime record.  The implementations are kept
training-free so the computational effect of completion handling is isolated.

Methods
-------
B0  return the initial layout;
B1  unfiltered randomized construction with final verification only;
B2  incumbent-preserving multi-facility destroy and repair;
B2S the same control with destroy size restricted to 2--4;
B3  fortified rollout, ranking actions by verified completed-layout value;
M0  marginally ranked construction with full completion regeneration;
R0  marginally ranked construction with unchanged-witness reuse only;
M1  the same candidate order with bounded repair of the carried completion.
"""

import time

import numpy as np

import core
from certified_decode import FirstFitWitness
from greedy_construct import GreedyConstructor
from layout_verifier import verify_complete_layout, verify_witness_completion
from witness_frontier import best_contact


METHODS = ("B0", "B1", "B2", "B2S", "B3", "M0", "R0", "M1")


def _overlap(a, b, iw, ih):
    ia, ra, ca = a
    ib, rb, cb = b
    return (ca < cb + int(iw[ib]) and cb < ca + int(iw[ia])
            and ra < rb + int(ih[ib]) and rb < ra + int(ih[ia]))


class LayoutProblem:
    """One fixed-shape instance and the geometry helpers used by all methods."""

    def __init__(self, inst):
        if inst.B != 1:
            raise ValueError("LayoutProblem requires one instance")
        self.inst = inst
        self.n = inst.n
        self.iw = np.rint(inst.w[0]).astype(np.int64)
        self.ih = np.rint(inst.h[0]).astype(np.int64)
        self.gw = int(round(inst.gw[0]))
        self.gh = int(round(inst.gh[0]))
        self.canvas = max(self.gw, self.gh)
        self.generator = FirstFitWitness(self.iw, self.ih,
                                         self.canvas, self.canvas,
                                         self.gw, self.gh)
        self.greedy = GreedyConstructor(inst, canvas=self.canvas)
        self.objective = core.BatchObjective(inst)

    def occupancy(self, placements):
        occ = np.zeros((self.canvas, self.canvas), dtype=np.int32)
        for facility, row, column in placements:
            occ[row:row + self.ih[facility],
                column:column + self.iw[facility]] += 1
        return occ

    def score(self, placements, already_verified=False):
        if not already_verified:
            check = verify_complete_layout(self.iw, self.ih, self.gw, self.gh,
                                           placements)
            if not check.valid:
                raise RuntimeError("attempted to score invalid layout: " + check.reason)
        px = np.zeros(self.n)
        py = np.zeros(self.n)
        for facility, row, column in placements:
            px[facility] = column + self.inst.w[0, facility] / 2.0
            py[facility] = row + self.inst.h[0, facility] / 2.0
        return float(self.objective.evaluate(px[None, :], py[None, :])[0])

    def initial_best_contact(self):
        start = time.perf_counter()
        order = sorted(range(self.n),
                       key=lambda i: -(self.iw[i] * self.ih[i]))
        ok, witness = best_contact(self.generator, order)
        if ok:
            check = verify_complete_layout(self.iw, self.ih, self.gw, self.gh,
                                           witness)
            ok = check.valid
        elapsed = time.perf_counter() - start
        return ok, witness if ok else None, elapsed

    def candidate_positions(self, prefix, facility, limit, rng, noise=0.0):
        occ = self.occupancy(prefix)[None, :, :]
        placed = np.zeros((1, self.n), dtype=bool)
        x = np.zeros((1, self.n))
        y = np.zeros((1, self.n))
        for j, row, column in prefix:
            placed[0, j] = True
            x[0, j] = column + self.inst.w[0, j] / 2.0
            y[0, j] = row + self.inst.h[0, j] / 2.0
        legal = self.greedy._legal(occ, facility)[0][0]
        gain = self.greedy._gain(facility, x, y, placed)[0]
        if noise:
            finite = gain[legal]
            scale = max(float(np.std(finite)) if finite.size else 0.0, 1.0)
            gain = gain + rng.gumbel(size=gain.shape) * noise * scale
        score = np.where(legal, gain, -np.inf).reshape(-1)
        order = np.argsort(-score, kind="mergesort")
        output = []
        for flat in order:
            if not np.isfinite(score[flat]) or len(output) >= limit:
                break
            output.append((int(flat // self.canvas), int(flat % self.canvas)))
        return output


def _complete(prefix, witness):
    return list(prefix) + list(witness)


def _full_regeneration(problem, prefix, remaining, rng, tries, stats):
    stats["full_completion_calls"] += 1
    started = time.perf_counter()
    ok, witness = problem.generator.certify(problem.occupancy(prefix), remaining,
                                            tries=tries, rng=rng)
    stats["completion_generation_seconds"] += time.perf_counter() - started
    if not ok:
        return None
    stats["verifier_calls"] += 1
    started = time.perf_counter()
    if not verify_witness_completion(problem.iw, problem.ih,
                                     problem.gw, problem.gh,
                                     prefix, witness).valid:
        raise RuntimeError("full completion failed independent verification")
    stats["intermediate_verification_seconds"] += time.perf_counter() - started
    stats["verified_completions"] += 1
    return witness


def _partial_repair(problem, prefix, old_witness, facility, position,
                    remaining, rng, cap, log_cap, tries, stats):
    proposal = (facility, position[0], position[1])
    retained = []
    displaced = []
    for item in old_witness:
        if item[0] == facility:
            continue
        if _overlap(proposal, item, problem.iw, problem.ih):
            displaced.append(item[0])
        else:
            retained.append(item)
    stats["displaced_sizes"].append(len(displaced))
    if not displaced:
        stats["proposals_D0"] += 1
    elif len(displaced) <= log_cap:
        stats["proposals_D1_to_cap"] += 1
    else:
        stats["proposals_D_above_cap"] += 1
    if not displaced:
        witness = retained
        stats["unchanged_reuses"] += 1
        branch = "reuse"
    else:
        if len(displaced) > cap:
            return None
        stats["repair_attempts"] += 1
        obstacles = list(prefix) + [proposal] + retained
        started = time.perf_counter()
        ok, repaired = problem.generator.certify(problem.occupancy(obstacles),
                                                 displaced, tries=tries, rng=rng)
        elapsed = time.perf_counter() - started
        stats["completion_generation_seconds"] += elapsed
        stats["repair_generation_seconds"] += elapsed
        if not ok:
            return None
        witness = retained + repaired
        stats["repair_successes"] += 1
        branch = "repair"
    successor = list(prefix) + [proposal]
    stats["verifier_calls"] += 1
    started = time.perf_counter()
    if not verify_witness_completion(problem.iw, problem.ih,
                                     problem.gw, problem.gh,
                                     successor, witness).valid:
        raise RuntimeError("partial repair failed independent verification")
    stats["intermediate_verification_seconds"] += time.perf_counter() - started
    stats["verified_completions"] += 1
    return witness, branch


def _empty_stats():
    return {
        "full_completion_calls": 0,
        "verifier_calls": 0,
        "verified_completions": 0,
        "unchanged_reuses": 0,
        "repair_attempts": 0,
        "repair_successes": 0,
        "displaced_sizes": [],
        "proposals_D0": 0,
        "proposals_D1_to_cap": 0,
        "proposals_D_above_cap": 0,
        "fallback_reuses": 0,
        "destroy_sizes": [],
        "completion_generation_seconds": 0.0,
        "repair_generation_seconds": 0.0,
        "intermediate_verification_seconds": 0.0,
        "final_verification_seconds": 0.0,
        "objective_seconds": 0.0,
        "late_completed_events": 0,
        "verified_events_by_branch": {},
        "incumbent_improvements_by_branch": {},
        "incumbent_gain_by_branch": {},
        "reconstructions": 0,
        "final_verification_failures": 0,
        "timeouts": 0,
    }


def _record(problem, placements, start, deadline, events, stats, branch):
    now = time.perf_counter()
    if now >= deadline:
        stats["timeouts"] += 1
        return False
    stats["verifier_calls"] += 1
    verify_start = time.perf_counter()
    check = verify_complete_layout(problem.iw, problem.ih,
                                   problem.gw, problem.gh, placements)
    stats["final_verification_seconds"] += time.perf_counter() - verify_start
    if not check.valid:
        stats["final_verification_failures"] += 1
        return False
    score_start = time.perf_counter()
    value = problem.score(placements, already_verified=True)
    stats["objective_seconds"] += time.perf_counter() - score_start
    completed = time.perf_counter()
    if completed > deadline:
        stats["late_completed_events"] += 1
        stats["timeouts"] += 1
        return False
    events.append((completed - start, value, list(placements), branch))
    counts = stats["verified_events_by_branch"]
    counts[branch] = counts.get(branch, 0) + 1
    return True


def _reconstruct(problem, initial_witness, mode, rng, start, deadline,
                 events, stats, kappa, repair_cap, tries, random_order):
    """Run one completion-aware reconstruction and log verified completions."""
    prefix = []
    witness = list(initial_witness)
    area_order = sorted(range(problem.n),
                        key=lambda i: -(problem.iw[i] * problem.ih[i]))
    order = list(rng.permutation(problem.n)) if random_order else area_order
    noise = 0.08 if random_order else 0.0
    stats["reconstructions"] += 1

    for facility in order:
        if time.perf_counter() >= deadline:
            stats["timeouts"] += 1
            return
        positions = problem.candidate_positions(prefix, facility, kappa,
                                                rng, noise=noise)
        witness_position = next((r, c) for i, r, c in witness
                                if i == facility)
        if witness_position not in positions:
            positions.append(witness_position)
        remaining = [i for i, _, _ in witness if i != facility]
        choices = []

        for position in positions:
            if time.perf_counter() >= deadline:
                stats["timeouts"] += 1
                break
            proposal = (facility, position[0], position[1])
            successor = prefix + [proposal]
            if position == witness_position:
                candidate_witness = [item for item in witness
                                     if item[0] != facility]
                branch = "fallback"
                stats["fallback_reuses"] += 1
            elif mode in ("M1", "R0"):
                repaired = _partial_repair(
                    problem, prefix, witness, facility, position, remaining,
                    rng, repair_cap if mode == "M1" else 0,
                    repair_cap, tries, stats)
                if repaired is None:
                    continue
                candidate_witness, branch = repaired
            else:
                candidate_witness = _full_regeneration(
                    problem, successor, remaining, rng, tries, stats)
                branch = "full-regeneration"
            if candidate_witness is None:
                continue
            full = _complete(successor, candidate_witness)
            if not _record(problem, full, start, deadline, events, stats, branch):
                if time.perf_counter() > deadline:
                    return
                continue
            value = events[-1][1]
            choices.append((value, proposal, candidate_witness))
            # Marginally ranked M0/M1 commits the first feasible proposal.
            if mode != "B3":
                break

        if choices:
            if mode == "B3":
                # The carried witness action is in the choice set.  Maximizing
                # completed value therefore gives the fortified non-worsening
                # property without a separate acceptance exception.
                _, proposal, candidate_witness = max(choices, key=lambda item: item[0])
            else:
                _, proposal, candidate_witness = choices[0]
        else:
            # Timeout can occur before the explicit witness action is reached.
            # The full incumbent has already been logged, so return it safely.
            return
        prefix.append(proposal)
        witness = candidate_witness


def _unfiltered_construction(problem, rng, start, deadline, events, stats,
                             random_order):
    prefix = []
    order = sorted(range(problem.n),
                   key=lambda i: -(problem.iw[i] * problem.ih[i]))
    if random_order:
        order = list(rng.permutation(problem.n))
    for facility in order:
        if time.perf_counter() >= deadline:
            stats["timeouts"] += 1
            return
        positions = problem.candidate_positions(prefix, facility, 8, rng,
                                                noise=0.15 if random_order else 0.0)
        if not positions:
            return
        # Later restarts diversify among the strongest marginal positions.
        pick = 0 if not random_order else int(rng.integers(min(4, len(positions))))
        row, column = positions[pick]
        prefix.append((facility, row, column))
    _record(problem, prefix, start, deadline, events, stats,
            "final-verification")


def _destroy_repair(problem, incumbent, rng, start, deadline, events, stats,
                    repair_cap):
    qmax = min(repair_cap, problem.n)
    if qmax < 2:
        return
    q = int(rng.integers(2, qmax + 1))
    stats["destroy_sizes"].append(q)
    pos = {i: (r, c) for i, r, c in incumbent}
    anchor = int(rng.integers(problem.n))
    centers = np.array([[pos[i][1] + problem.iw[i] / 2.0,
                         pos[i][0] + problem.ih[i] / 2.0]
                        for i in range(problem.n)])
    distance = ((centers - centers[anchor]) ** 2).sum(axis=1)
    destroyed = list(np.argsort(distance)[:q])
    kept = [item for item in incumbent if item[0] not in set(destroyed)]
    prefix = list(kept)
    order = sorted(destroyed,
                   key=lambda i: -(problem.iw[i] * problem.ih[i]))
    if rng.random() < 0.5:
        rng.shuffle(order)
    for facility in order:
        if time.perf_counter() >= deadline:
            stats["timeouts"] += 1
            return
        positions = problem.candidate_positions(prefix, facility, 12, rng,
                                                noise=0.10)
        if not positions:
            return
        # Randomized restricted candidate list is the repair-search component.
        pick = int(rng.integers(min(4, len(positions))))
        row, column = positions[pick]
        prefix.append((facility, row, column))
    _record(problem, prefix, start, deadline, events, stats, "destroy-repair")


def run_anytime(problem, initial_witness, method, budgets, seed=0, kappa=16,
                repair_cap=8, lns_cap=16, completion_tries=1,
                initial_value=None):
    """Run one method once to the largest budget and sample its anytime trace."""
    if method not in METHODS:
        raise ValueError("unknown method: " + method)
    budgets = sorted(float(b) for b in budgets)
    if not budgets or budgets[0] < 0:
        raise ValueError("budgets must be nonnegative")
    rng = np.random.default_rng(seed)
    stats = _empty_stats()
    if initial_value is None:
        initial_value = problem.score(initial_witness)
    start = time.perf_counter()
    deadline = start + budgets[-1]
    events = [(0.0, initial_value, list(initial_witness), "initial")]
    incumbent = list(initial_witness)
    best_value = initial_value
    iteration = 0

    while method != "B0" and time.perf_counter() < deadline:
        before = len(events)
        if method == "B1":
            _unfiltered_construction(problem, rng, start, deadline, events,
                                     stats, random_order=iteration > 0)
        elif method in ("B2", "B2S"):
            _destroy_repair(problem, incumbent, rng, start, deadline, events,
                            stats, repair_cap=(4 if method == "B2S"
                                               else max(4, lns_cap)))
        else:
            _reconstruct(problem, incumbent, method, rng, start, deadline,
                         events, stats, kappa, repair_cap, completion_tries,
                         random_order=iteration > 0)
        for _, value, layout, branch in events[before:]:
            if value > best_value + 1e-12:
                gain = value - best_value
                best_value = value
                incumbent = list(layout)
                counts = stats["incumbent_improvements_by_branch"]
                counts[branch] = counts.get(branch, 0) + 1
                gains = stats["incumbent_gain_by_branch"]
                gains[branch] = gains.get(branch, 0.0) + gain
        iteration += 1

    finished = time.perf_counter()
    stats["actual_search_elapsed_s"] = finished - start
    stats["overrun_s"] = max(0.0, finished - deadline)
    stats["max_admitted_event_s"] = max(event[0] for event in events)
    rows = []
    for budget in budgets:
        eligible = [(value, layout) for elapsed, value, layout, _ in events
                    if elapsed <= budget + 1e-12]
        value, layout = max(eligible, key=lambda item: item[0])
        check = verify_complete_layout(problem.iw, problem.ih,
                                       problem.gw, problem.gh, layout)
        if not check.valid:
            raise RuntimeError("anytime incumbent failed final verification")
        rows.append({
            "budget_s": budget,
            "J": value,
            "improvement": value - initial_value,
            "improvement_pct": (100.0 * (value - initial_value) / initial_value
                                if initial_value > 0 else None),
            "layout": [[int(i), int(r), int(c)] for i, r, c in layout],
        })
    stats["events"] = len(events)
    stats["iterations"] = iteration
    stats["mean_displaced"] = (float(np.mean(stats["displaced_sizes"]))
                               if stats["displaced_sizes"] else None)
    stats["max_displaced"] = (max(stats["displaced_sizes"])
                               if stats["displaced_sizes"] else None)
    stats["mean_destroy_size"] = (float(np.mean(stats["destroy_sizes"]))
                                  if stats["destroy_sizes"] else None)
    trace = [{"elapsed_s": float(elapsed), "J": float(value),
              "branch": branch}
             for elapsed, value, _, branch in events]
    return {"method": method, "initial_J": initial_value,
            "rows": rows, "trace": trace, "stats": stats}
