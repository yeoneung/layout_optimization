"""Material-flow objective for the EAAI track (X6).

The classical material-handling cost of facility layout is
    cost(g) = sum_{i<j} f_ij * d_ij,   d_ij = |x_i - x_j| + |y_i - y_j|,
the flow-weighted rectilinear distance between facility centroids.  The search
code maximizes an objective, so the objective used by every method is
    J_flow(g) = C - cost(g),   C = sum_{i<j} f_ij * (W + H),
where C bounds the cost from above on the plate, so J_flow >= 0.  Raw gains
J - J(w_0) are cost reductions; the analysis reports them as a percentage of
the initial cost.

Nothing in the archive changes.  FlowLayoutProblem replaces the objective and
the marginal-gain evaluator of completion_search.LayoutProblem, and activate()
replaces the facility-contribution rule that the ALNS control uses for
low-contribution removal.  Witness generation, repair and verification are
geometric and unchanged, so M0, M1, the cap family, the bandit controllers and
ALNS run under the flow objective without modification.

Flow matrices are from-to charts generated per instance: a fraction of the
facility pairs (density) carries a positive flow with log-normal magnitude, the
matrix is symmetric with zero diagonal, and the draw is fixed by a seed.
"""

import numpy as np

import repair_baseline
from completion_search import LayoutProblem
from greedy_construct import GreedyConstructor

FLOW_DENSITY = 0.30
FLOW_SIGMA = 0.8


def generate_flow(n, seed, density=FLOW_DENSITY, sigma=FLOW_SIGMA):
    """Symmetric nonnegative (n, n) flow matrix with zero diagonal."""
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density
    magnitude = rng.lognormal(mean=0.0, sigma=sigma, size=(n, n))
    flow = np.triu(np.where(mask, magnitude, 0.0), 1)
    return flow + flow.T


class FlowObjective:
    """Batched flow objective with the interface completion_search needs."""

    def __init__(self, inst, flow):
        self.I = inst
        self.B, self.n = inst.B, inst.n
        self.flow = np.asarray(flow, dtype=float)
        if self.flow.shape != (self.B, self.n, self.n):
            raise ValueError("flow must have shape (B, n, n)")
        self.iu0, self.iu1 = np.triu_indices(self.n, k=1)
        self.fpair = self.flow[:, self.iu0, self.iu1]
        self.constant = (self.fpair * (inst.gw[:, None] + inst.gh[:, None])).sum(axis=1)

    def cost(self, x, y):
        d = (np.abs(x[:, self.iu0] - x[:, self.iu1]) + np.abs(y[:, self.iu0] - y[:, self.iu1]))
        return (self.fpair * d).sum(axis=1)

    def evaluate(self, x, y, components=False):
        cost = self.cost(x, y)
        total = self.constant - cost
        if components:
            return total, {"cost": cost}
        return total


class FlowGreedy(GreedyConstructor):
    """GreedyConstructor whose marginal gain is the negative flow cost added by a placement."""

    def __init__(self, inst, flow, canvas=None):
        super().__init__(inst, canvas)
        self.flow = np.asarray(flow, dtype=float)

    def _gain(self, i, x, y, placed, integral=None):
        w = self.I.w[:, i][:, None, None]
        h = self.I.h[:, i][:, None, None]
        cx = self._cols + w / 2.0
        cy = self._rows + h / 2.0
        gain = np.zeros((self.B, self.GH, self.GW))
        for j in range(self.n):
            m = placed[:, j]
            if not m.any() or j == i:
                continue
            f = self.flow[:, i, j][:, None, None]
            d = np.abs(cx - x[:, j][:, None, None]) + np.abs(cy - y[:, j][:, None, None])
            gain -= np.where(m[:, None, None], f * d, 0.0)
        return gain


class FlowLayoutProblem(LayoutProblem):
    """One instance under the flow objective; geometry helpers are inherited."""

    def __init__(self, inst, flow):
        super().__init__(inst)
        self.flow = np.asarray(flow, dtype=float)
        if self.flow.shape != (self.n, self.n):
            raise ValueError("flow must have shape (n, n) for one instance")
        self.objective = FlowObjective(inst, self.flow[None, :, :])
        self.greedy = FlowGreedy(inst, self.flow[None, :, :], canvas=self.canvas)
        self.objective_name = "flow"

    def cost_of(self, placements):
        px = np.zeros(self.n)
        py = np.zeros(self.n)
        for facility, row, column in placements:
            px[facility] = column + self.inst.w[0, facility] / 2.0
            py[facility] = row + self.inst.h[0, facility] / 2.0
        return float(self.objective.cost(px[None, :], py[None, :])[0])


def flow_facility_contributions(problem, layout):
    """Negative flow cost incident to each facility (higher is better), for ALNS removal."""
    rows = np.zeros(problem.n)
    cols = np.zeros(problem.n)
    for i, row, col in layout:
        rows[i], cols[i] = row, col
    x = cols + problem.iw / 2.0
    y = rows + problem.ih / 2.0
    d = np.abs(x[:, None] - x[None, :]) + np.abs(y[:, None] - y[None, :])
    return -(problem.flow * d).sum(axis=1)


_ORIGINAL_CONTRIBUTIONS = repair_baseline.facility_contributions


def activate():
    """Route the ALNS low-contribution removal through the flow objective (process-wide)."""
    repair_baseline.facility_contributions = flow_facility_contributions


def deactivate():
    repair_baseline.facility_contributions = _ORIGINAL_CONTRIBUTIONS
