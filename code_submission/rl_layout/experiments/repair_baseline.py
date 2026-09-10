"""Adaptive destroy-and-repair control for fixed-shape lattice layouts.

This is a layout-specific ALNS implementation, not a reproduction of a routing
solver. It uses spatial, random and low-contribution removal, with beam and
regret insertion, adaptive operator weights and optional Metropolis acceptance.
Every scored completion uses the same independent verifier as M0 and M1.
"""

import math
import time

import numpy as np

from completion_search import _empty_stats, _record
from core import _pair_distance
from layout_verifier import verify_complete_layout


CONFIGURATIONS = {
    "small_beam": {"destroy_cap": 4, "beam_width": 4,
                   "candidates": 4, "temperature_fraction": 0.0025},
    "medium_beam": {"destroy_cap": 8, "beam_width": 4,
                    "candidates": 4, "temperature_fraction": 0.0025},
    "wide_beam": {"destroy_cap": 8, "beam_width": 8,
                  "candidates": 8, "temperature_fraction": 0.0025},
    "large_beam": {"destroy_cap": 16, "beam_width": 4,
                   "candidates": 4, "temperature_fraction": 0.0025},
}


def ranked_positions(problem, prefix, facility, limit, original_position=None):
    """Legal positions and exact marginal gains, including a legal original slot."""
    occupied = problem.occupancy(prefix)[None, :, :]
    placed = np.zeros((1, problem.n), dtype=bool)
    x = np.zeros((1, problem.n))
    y = np.zeros((1, problem.n))
    for j, row, col in prefix:
        placed[0, j] = True
        x[0, j] = col + problem.iw[j] / 2.0
        y[0, j] = row + problem.ih[j] / 2.0
    legal = problem.greedy._legal(occupied, facility)[0][0]
    gains = problem.greedy._gain(facility, x, y, placed)[0]
    flat_gains = np.where(legal, gains, -np.inf).reshape(-1)
    indices = np.argsort(-flat_gains, kind="mergesort")
    positions = []
    for index in indices:
        if len(positions) >= limit or not np.isfinite(flat_gains[index]):
            break
        row, col = divmod(int(index), problem.canvas)
        positions.append((row, col, float(gains[row, col])))
    if original_position is not None:
        row, col = original_position
        if legal[row, col] and not any((r, c) == (row, col)
                                     for r, c, _ in positions):
            positions.append((row, col, float(gains[row, col])))
    return positions


def facility_contributions(problem, layout):
    """Incident pair rewards plus boundary rewards; the constant bonus is omitted."""
    rows = np.zeros(problem.n, dtype=int)
    cols = np.zeros(problem.n, dtype=int)
    for i, row, col in layout:
        rows[i], cols[i] = row, col
    x, y = cols + problem.iw / 2.0, rows + problem.ih / 2.0
    w, h = problem.iw, problem.ih
    distance = _pair_distance(x[:, None], y[:, None], w[:, None], h[:, None],
                              x[None, :], y[None, :], w[None, :], h[None, :])
    dmax = problem.objective.dmax[0][problem.greedy.pair_idx]
    normalized = np.clip(distance / np.maximum(dmax, 1e-12), 0.0, 1.0)
    normalized **= problem.inst.weights.distance_power
    affinity = problem.inst.adj[0]
    pairs = np.where(affinity >= 0, affinity * normalized,
                     -affinity * (1.0 - normalized))
    np.fill_diagonal(pairs, 0.0)
    result = problem.inst.weights.adj_weight * pairs.sum(axis=1)
    result += ((cols == 0) | (cols + w == problem.gw)) * problem.inst.edge_side[0]
    result += (rows == 0) * problem.inst.edge_bottom[0]
    result += (rows + h == problem.gh) * problem.inst.edge_top[0]
    return result


def destroy(problem, layout, kind, cap, rng):
    count = int(rng.integers(2, min(cap, problem.n) + 1))
    positions = {i: (r, c) for i, r, c in layout}
    if kind == "random":
        removed = list(map(int, rng.choice(problem.n, count, replace=False)))
    elif kind == "spatial":
        centers = np.array([[positions[i][1] + problem.iw[i] / 2.0,
                             positions[i][0] + problem.ih[i] / 2.0]
                            for i in range(problem.n)])
        anchor = int(rng.integers(problem.n))
        distances = ((centers - centers[anchor]) ** 2).sum(axis=1)
        removed = list(map(int, np.argsort(distances, kind="mergesort")[:count]))
    elif kind == "low_contribution":
        ranked = np.argsort(facility_contributions(problem, layout))
        probabilities = np.empty(problem.n)
        probabilities[ranked] = 1.0 / np.sqrt(np.arange(1, problem.n + 1))
        probabilities /= probabilities.sum()
        removed = list(map(int, rng.choice(problem.n, count, replace=False,
                                           p=probabilities)))
    else:
        raise ValueError("unknown removal operator")
    removed_set = set(removed)
    return [x for x in layout if x[0] not in removed_set], removed, positions


def beam_repair(problem, kept, removed, originals, rng, config, deadline):
    order = sorted(removed, key=lambda i: -(problem.iw[i] * problem.ih[i]))
    if rng.random() < 0.25:
        rng.shuffle(order)
    beam = [(0.0, list(kept))]
    for facility in order:
        successors = []
        for value, prefix in beam:
            if time.perf_counter() >= deadline:
                return None
            positions = ranked_positions(problem, prefix, facility,
                                          config["candidates"], originals[facility])
            for row, col, gain in positions:
                successors.append((value + gain, prefix + [(facility, row, col)]))
        if not successors:
            return None
        successors.sort(key=lambda x: x[0], reverse=True)
        beam = successors[:config["beam_width"]]
    return beam[0][1]


def regret_repair(problem, kept, removed, originals, rng, config, deadline):
    prefix, remaining = list(kept), list(removed)
    while remaining:
        choices = []
        for facility in remaining:
            if time.perf_counter() >= deadline:
                return None
            positions = ranked_positions(problem, prefix, facility,
                                          config["candidates"], originals[facility])
            if not positions:
                return None
            positions.sort(key=lambda x: x[2], reverse=True)
            regret = (positions[0][2] - positions[1][2]
                      if len(positions) > 1 else float("inf"))
            choices.append((regret, problem.iw[facility] * problem.ih[facility],
                            facility, positions[0]))
        _, _, facility, (row, col, _) = max(choices, key=lambda x: (x[0], x[1]))
        prefix.append((facility, row, col))
        remaining.remove(facility)
    return prefix


def run_adaptive_repair(problem, initial_witness, budgets, seed, configuration,
                        initial_value=None):
    """Run ALNS with immutable configuration and the common anytime record rule."""
    config = dict(CONFIGURATIONS[configuration])
    budgets = sorted(float(x) for x in budgets)
    if not budgets or budgets[0] < 0 or not all(math.isfinite(x) for x in budgets):
        raise ValueError("budgets must be finite and nonnegative")
    if problem.n < 2:
        raise ValueError("ALNS requires at least two facilities")
    rng = np.random.default_rng(seed)
    if initial_value is None:
        initial_value = problem.score(initial_witness)
    stats = _empty_stats()
    stats.update({"operator_attempts": {}, "operator_completions": {},
                  "accepted_nonimproving": 0, "configuration": config})
    operators = [(a, b) for a in ("spatial", "random", "low_contribution")
                 for b in ("beam", "regret")]
    weights = np.ones(len(operators))
    current, incumbent = list(initial_witness), list(initial_witness)
    current_value = best_value = initial_value
    objective_scale = max(abs(initial_value - problem.inst.weights.no_overlap_bonus), 1.0)
    temperature = config["temperature_fraction"] * objective_scale
    start = time.perf_counter()
    deadline = start + budgets[-1]
    events = [(0.0, initial_value, list(initial_witness), "initial")]
    iteration = 0
    while time.perf_counter() < deadline:
        index = int(rng.choice(len(operators), p=weights / weights.sum()))
        removal, insertion = operators[index]
        name = removal + "/" + insertion
        stats["operator_attempts"][name] = stats["operator_attempts"].get(name, 0) + 1
        kept, removed, originals = destroy(problem, current, removal,
                                             config["destroy_cap"], rng)
        stats["destroy_sizes"].append(len(removed))
        repair = beam_repair if insertion == "beam" else regret_repair
        candidate = repair(problem, kept, removed, originals, rng, config, deadline)
        reward = 0.0
        if candidate is not None and _record(problem, candidate, start, deadline,
                                             events, stats, "adaptive-repair"):
            stats["operator_completions"][name] = stats["operator_completions"].get(name, 0) + 1
            value = events[-1][1]
            delta = value - current_value
            accepted = delta > 1e-12 or rng.random() < math.exp(
                min(0.0, delta / max(temperature, 1e-12)))
            if accepted:
                reward = 3.0 if delta > 1e-12 else 1.0
                if delta <= 1e-12:
                    stats["accepted_nonimproving"] += 1
                current, current_value = candidate, value
            if value > best_value + 1e-12:
                reward = 6.0
                gain = value - best_value
                best_value, incumbent = value, candidate
                counts = stats["incumbent_improvements_by_branch"]
                counts["adaptive-repair"] = counts.get("adaptive-repair", 0) + 1
                gains = stats["incumbent_gain_by_branch"]
                gains["adaptive-repair"] = gains.get("adaptive-repair", 0.0) + gain
        weights[index] = max(0.2, 0.9 * weights[index] + 0.1 * reward)
        temperature *= 0.999
        iteration += 1
    finished = time.perf_counter()
    rows = []
    for budget in budgets:
        _, value, layout, _ = max((event for event in events if event[0] <= budget),
                                  key=lambda event: event[1])
        check = verify_complete_layout(problem.iw, problem.ih, problem.gw,
                                        problem.gh, layout)
        if not check.valid:
            raise RuntimeError("ALNS incumbent verification failed")
        rows.append({"budget_s": budget, "J": value,
                     "improvement": value - initial_value,
                     "improvement_pct": (100.0 * (value - initial_value) / initial_value
                                         if initial_value > 0 else None),
                     "layout": [[int(i), int(r), int(c)] for i, r, c in layout]})
    stats.update({"actual_search_elapsed_s": finished - start,
                  "overrun_s": max(0.0, finished - deadline),
                  "max_admitted_event_s": max(event[0] for event in events),
                  "events": len(events), "iterations": iteration,
                  "mean_displaced": None, "max_displaced": None,
                  "mean_destroy_size": (float(np.mean(stats["destroy_sizes"]))
                                        if stats["destroy_sizes"] else None),
                  "operator_weights": {a + "/" + b: float(weights[k])
                                       for k, (a, b) in enumerate(operators)}})
    trace = [{"elapsed_s": float(t), "J": float(v), "branch": branch}
             for t, v, _, branch in events]
    return {"method": "ALNS", "initial_J": initial_value, "rows": rows,
            "trace": trace, "stats": stats, "configuration_name": configuration}
