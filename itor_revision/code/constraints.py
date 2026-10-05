"""Fixed obstacles and a pre-placed dock for the EAAI track (X7).

Industrial plates are not empty: columns, utilities and pre-placed stations
occupy cells that no facility may use, and receiving or shipping docks are
fixed facilities with material flow to the movable ones.  Both enter the
completion machinery through the occupancy grid: every legality mask, witness
generation, repair and candidate ranking already reads an occupancy array, so
marking blocked cells there is sufficient.  The best-contact initializer is
re-implemented to start from the blocked grid.  Dock flows enter the objective
as fixed anchors.

Feasibility by construction: obstacle cells are drawn from cells that are free
in the withheld source packing of the instance, so that packing remains a
feasible layout of the constrained instance.  Nothing in the archive changes.
"""

import numpy as np

import repair_baseline
from flow_objective import FlowGreedy, FlowLayoutProblem, FlowObjective


class AnchoredFlowObjective(FlowObjective):
    """Flow objective with fixed anchors (dock centers) that carry flow to facilities."""

    def __init__(self, inst, flow, anchors):
        super().__init__(inst, flow)
        self.anchor_x = np.array([a["x"] for a in anchors], dtype=float)
        self.anchor_y = np.array([a["y"] for a in anchors], dtype=float)
        self.anchor_flow = np.array([a["flow"] for a in anchors], dtype=float).reshape(len(anchors), self.n)
        self.constant = self.constant + (self.anchor_flow.sum() * (inst.gw[0] + inst.gh[0]))

    def cost(self, x, y):
        base = super().cost(x, y)
        if self.anchor_flow.size == 0:
            return base
        d = (np.abs(x[:, None, :] - self.anchor_x[None, :, None])
             + np.abs(y[:, None, :] - self.anchor_y[None, :, None]))      # (B, A, n)
        return base + (self.anchor_flow[None, :, :] * d).sum(axis=(1, 2))


class AnchoredFlowGreedy(FlowGreedy):
    def __init__(self, inst, flow, anchors, canvas=None):
        super().__init__(inst, flow, canvas)
        self.anchors = anchors

    def _gain(self, i, x, y, placed, integral=None):
        gain = super()._gain(i, x, y, placed, integral)
        w = self.I.w[:, i][:, None, None]
        h = self.I.h[:, i][:, None, None]
        cx = self._cols + w / 2.0
        cy = self._rows + h / 2.0
        for a in self.anchors:
            f = float(a["flow"][i])
            if f > 0:
                gain = gain - f * (np.abs(cx - a["x"]) + np.abs(cy - a["y"]))
        return gain


def best_contact_with_obstacles(gen, order, blocked):
    """witness_frontier.best_contact starting from a blocked occupancy grid."""
    work = blocked.astype(np.int32).copy()
    wit = []
    for i in order:
        m = gen._legal(work, i)
        if not m.any():
            return False, None
        w, h = int(gen.iw[i]), int(gen.ih[i])
        occ = (work > 0).astype(np.int64)
        C = np.zeros((gen.GH + 1, gen.GW + 2), dtype=np.int64)
        C[1:, 1:-1] = occ.cumsum(axis=0)
        R = np.zeros((gen.GH + 2, gen.GW + 1), dtype=np.int64)
        R[1:-1, 1:] = occ.cumsum(axis=1)
        rows = np.arange(gen.GH)[:, None]
        cols = np.arange(gen.GW)[None, :]
        r1 = np.minimum(rows + h, gen.GH)
        c1 = np.minimum(cols + w, gen.GW)
        left = C[r1, cols] - C[rows, cols]
        right = C[r1, np.minimum(c1 + 1, gen.GW + 1)] - C[rows, np.minimum(c1 + 1, gen.GW + 1)]
        bottom = R[rows, c1] - R[rows, cols]
        top = R[np.minimum(r1 + 1, gen.GH + 1), c1] - R[np.minimum(r1 + 1, gen.GH + 1), cols]
        contact = (left + right + bottom + top
                   + np.where(cols == 0, h, 0) + np.where(cols + w == gen.gw, h, 0)
                   + np.where(rows == 0, w, 0) + np.where(rows + h == gen.gh, w, 0))
        score = np.where(m, contact, -1)
        flat = int(score.argmax())
        r, c = flat // gen.GW, flat % gen.GW
        work[r:r + h, c:c + w] += 1
        wit.append((int(i), r, c))
    return True, wit


class ConstrainedFlowLayoutProblem(FlowLayoutProblem):
    """Flow objective plus blocked cells and dock anchors."""

    def __init__(self, inst, flow, blocked, anchors):
        super().__init__(inst, flow)
        blocked = np.asarray(blocked, dtype=bool)
        if blocked.shape != (self.canvas, self.canvas):
            raise ValueError("blocked mask must cover the canvas")
        self.blocked = blocked
        self.anchors = list(anchors)
        self.objective = AnchoredFlowObjective(inst, self.flow[None, :, :], self.anchors)
        self.greedy = AnchoredFlowGreedy(inst, self.flow[None, :, :], self.anchors, canvas=self.canvas)
        self.objective_name = "flow+obstacles"

    def occupancy(self, placements):
        occ = super().occupancy(placements)
        occ[self.blocked] += 1
        return occ

    def initial_best_contact(self):
        import time
        start = time.perf_counter()
        order = sorted(range(self.n), key=lambda i: -(self.iw[i] * self.ih[i]))
        ok, witness = best_contact_with_obstacles(self.generator, order, self.blocked)
        if ok:
            ok = self.verifies_obstacles(witness)
        elapsed = time.perf_counter() - start
        return ok, witness if ok else None, elapsed

    def verifies_obstacles(self, layout):
        occ = np.zeros((self.canvas, self.canvas), dtype=np.int32)
        for facility, row, column in layout:
            occ[row:row + self.ih[facility], column:column + self.iw[facility]] += 1
        return bool(np.all(occ[self.blocked] == 0)) and bool(occ.max() <= 1)


def source_free_cells(inst, source_witness, canvas):
    """Boolean canvas mask of cells inside the plate not covered by the source packing."""
    gw, gh = int(round(inst.gw[0])), int(round(inst.gh[0]))
    covered = np.zeros((canvas, canvas), dtype=bool)
    for i in range(inst.n):
        x, y = int(round(source_witness[i, 0])), int(round(source_witness[i, 1]))
        w, h = int(round(inst.w[0, i])), int(round(inst.h[0, i]))
        covered[y:y + h, x:x + w] = True
    inside = np.zeros((canvas, canvas), dtype=bool)
    inside[:gh, :gw] = True
    return inside & ~covered


def make_constraints(inst, source_witness, canvas, seed, column_fraction=0.15, dock_shape=(2, 3),
                     dock_density=0.30, dock_sigma=0.8):
    """Blocked mask and dock anchors carved from the free cells of the source packing.

    Returns (blocked, anchors, info).  The dock is the first free rectangle of
    dock_shape found in row-major order (smaller fallbacks if none fits); the
    columns are a random fraction of the remaining free cells.  Dock flows go
    to a random subset of facilities with log-normal magnitudes.
    """
    rng = np.random.default_rng(seed)
    free = source_free_cells(inst, source_witness, canvas)
    blocked = np.zeros_like(free)
    anchors = []
    dock = None
    for dh, dw in (dock_shape, (dock_shape[0], max(1, dock_shape[1] - 1)), (1, 2), (1, 1)):
        rows, cols = np.where(free)
        for r, c in zip(rows, cols):
            if r + dh <= canvas and c + dw <= canvas and free[r:r + dh, c:c + dw].all():
                dock = (int(r), int(c), int(dh), int(dw))
                break
        if dock is not None:
            break
    if dock is not None:
        r, c, dh, dw = dock
        blocked[r:r + dh, c:c + dw] = True
        mask = rng.random(inst.n) < dock_density
        flow = np.where(mask, rng.lognormal(0.0, dock_sigma, inst.n), 0.0)
        anchors.append({"x": c + dw / 2.0, "y": r + dh / 2.0, "flow": flow, "rect": dock})
    remaining = free & ~blocked
    cells = np.argwhere(remaining)
    count = int(round(column_fraction * len(cells)))
    if count > 0:
        chosen = cells[rng.choice(len(cells), count, replace=False)]
        blocked[chosen[:, 0], chosen[:, 1]] = True
    info = {"free_cells": int(free.sum()), "blocked_cells": int(blocked.sum()), "dock": dock,
            "dock_flow_pairs": int(len(anchors) and (anchors[0]["flow"] > 0).sum()),
            "effective_fill": float(1.0 - (free.sum() - blocked.sum()) / max(1, int(round(inst.gw[0])) * int(round(inst.gh[0]))))}
    return blocked, anchors, info


def anchored_flow_facility_contributions(problem, layout):
    rows = np.zeros(problem.n)
    cols = np.zeros(problem.n)
    for i, row, col in layout:
        rows[i], cols[i] = row, col
    x = cols + problem.iw / 2.0
    y = rows + problem.ih / 2.0
    d = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])
    contribution = -(problem.flow * d).sum(axis=1)
    for a in getattr(problem, "anchors", []):
        contribution = contribution - a["flow"] * (np.abs(x - a["x"]) + np.abs(y - a["y"]))
    return contribution


_ORIGINAL_CONTRIBUTIONS = repair_baseline.facility_contributions


def activate():
    repair_baseline.facility_contributions = anchored_flow_facility_contributions


def deactivate():
    repair_baseline.facility_contributions = _ORIGINAL_CONTRIBUTIONS
