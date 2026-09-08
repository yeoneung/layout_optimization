"""
Standard fixed-outline floorplanning benchmarks, with their own objective.

Everything else in this paper optimizes an adjacency-preference objective on
generated or hand-authored instances.  A reviewer is entitled to ask whether the
findings are an artefact of instances our own generator produced, so this module
brings in the MCNC and GSRC benchmarks --- ami33, ami49, n100, n200, n300 --- with
the objective the floorplanning literature actually uses, half-perimeter
wirelength, and the formulation it actually uses, fixed outline.

The geometry is identical to ours: fixed-shape rectangles, no overlap, a bounded
plate.  Only the objective differs, and it differs in a way that matters for one
of our results.  Our adjacency objective decomposes exactly because a masked
placement disturbs no existing term (Proposition "exact marginal decomposition").
HPWL decomposes too, but for a different reason: placing a block can only *grow*
the bounding box of the nets it belongs to, and that growth depends on the
already-placed pins of those nets, never on the unplaced ones.  So the increment
is again computable in closed form, now in O(nets containing i) rather than
O(placed blocks).  The constructive machinery therefore transfers unchanged,
which is what makes the comparison possible.

Pin model: block pins sit at block centres, terminals at their fixed positions
from the .pl file.  This is the standard simplification and we state it because
published HPWL numbers depend on it.

File formats are GSRC "UCSC blocks 1.0", UCLA "nets 1.0" and UCLA "pl 1.0".
"""

import os
import re
from dataclasses import dataclass, field

import numpy as np

BENCH_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "benchmarks")


# ---------------------------------------------------------------------------
def parse_blocks(path):
    """-> (names, w, h) for hard rectilinear blocks; terminals are skipped."""
    names, W, H = [], [], []
    with open(path) as f:
        for line in f:
            if "hardrectilinear" not in line:
                continue
            name = line.split()[0]
            pts = re.findall(r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)", line)
            xs = [int(a) for a, _ in pts]
            ys = [int(b) for _, b in pts]
            names.append(name)
            W.append(max(xs) - min(xs))
            H.append(max(ys) - min(ys))
    return names, np.array(W, dtype=np.int64), np.array(H, dtype=np.int64)


def parse_pl(path):
    """-> {name: (x, y)} for every entry; blocks are all at the origin."""
    out = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) >= 3 and not line.startswith("UCLA") and not line.startswith("#"):
                try:
                    out[p[0]] = (float(p[1]), float(p[2]))
                except ValueError:
                    pass
    return out


def parse_nets(path, block_index):
    """-> list of (block_ids, terminal_names) per net."""
    nets = []
    with open(path) as f:
        lines = [l.strip() for l in f]
    i = 0
    while i < len(lines):
        m = re.match(r"NetDegree\s*:\s*(\d+)", lines[i])
        if not m:
            i += 1
            continue
        deg = int(m.group(1))
        blocks, terms = [], []
        j = i + 1
        while j < len(lines) and len(blocks) + len(terms) < deg:
            name = lines[j].split()[0] if lines[j] else ""
            if name:
                if name in block_index:
                    blocks.append(block_index[name])
                else:
                    terms.append(name)
            j += 1
        nets.append((blocks, terms))
        i = j
    return nets


# ---------------------------------------------------------------------------
@dataclass
class FloorplanInstance:
    name: str
    w: np.ndarray                  # (n,) block widths
    h: np.ndarray                  # (n,) block heights
    W: int                         # plate width
    H: int                         # plate height
    nets_blocks: list              # per net, array of block ids
    nets_fixed: list               # per net, (min_x, max_x, min_y, max_y) of terminals or None
    names: list = field(default_factory=list)
    scale: int = 1                 # coordinates were divided by this

    @property
    def n(self):
        return len(self.w)

    @property
    def fill(self):
        return float((self.w * self.h).sum()) / float(self.W * self.H)


def load(name, dead_space=0.15, aspect=1.0, scale=None, bench_dir=None):
    """Load a benchmark as a fixed-outline instance.

    `dead_space` fixes the plate area at total block area / (1 - dead_space),
    which is how the fixed-outline literature parameterizes difficulty; 0.15 is
    its usual setting and corresponds to a fill ratio of 0.85 in our terms.

    `scale` divides all coordinates.  The benchmarks are specified on grids of
    up to ~10^3 per side, and our legality masks are dense over the plate, so
    full resolution costs O(WH) = 10^6 per query.  Scaling trades resolution for
    tractability and is reported; `scale=None` picks the smallest divisor that
    brings the plate under 200 cells per side.
    """
    d = bench_dir or BENCH_DIR
    names, w, h = parse_blocks(os.path.join(d, f"{name}.blocks"))
    idx = {nm: i for i, nm in enumerate(names)}
    pl = parse_pl(os.path.join(d, f"{name}.pl"))
    nets = parse_nets(os.path.join(d, f"{name}.nets"), idx)

    if scale is None:
        area = float((w * h).sum()) / (1.0 - dead_space)
        side = np.sqrt(area * aspect)
        scale = max(1, int(np.ceil(side / 200.0)))
    w = np.maximum(1, w // scale)
    h = np.maximum(1, h // scale)

    area = float((w * h).sum()) / (1.0 - dead_space)
    W = int(np.ceil(np.sqrt(area * aspect)))
    H = int(np.ceil(area / W))
    W = max(W, int(w.max()))
    H = max(H, int(h.max()))

    nb, nf = [], []
    for blocks, terms in nets:
        if len(blocks) + len(terms) < 2:
            continue                       # a one-pin net has zero HPWL always
        nb.append(np.array(sorted(set(blocks)), dtype=np.int64))
        pts = [pl[t] for t in terms if t in pl]
        if pts:
            xs = [p[0] / scale for p in pts]
            ys = [p[1] / scale for p in pts]
            nf.append((min(xs), max(xs), min(ys), max(ys)))
        else:
            nf.append(None)

    return FloorplanInstance(name=name, w=w, h=h, W=W, H=H, nets_blocks=nb,
                             nets_fixed=nf, names=names, scale=scale)


# ---------------------------------------------------------------------------
class HPWL:
    """Half-perimeter wirelength with an exact incremental form.

    `net_of[i]` lists the nets that block `i` belongs to, which is what makes the
    marginal cheap: placing `i` can change only those nets' bounding boxes.
    """

    def __init__(self, inst: FloorplanInstance):
        self.I = inst
        self.n = inst.n
        self.net_of = [[] for _ in range(self.n)]
        for k, bl in enumerate(inst.nets_blocks):
            for i in bl:
                self.net_of[i].append(k)

    def total(self, x, y):
        """x, y are block centres, shape (n,).  Returns total HPWL."""
        tot = 0.0
        for k, bl in enumerate(self.I.nets_blocks):
            xs, ys = x[bl], y[bl]
            lo_x, hi_x = xs.min(), xs.max()
            lo_y, hi_y = ys.min(), ys.max()
            f = self.I.nets_fixed[k]
            if f is not None:
                lo_x, hi_x = min(lo_x, f[0]), max(hi_x, f[1])
                lo_y, hi_y = min(lo_y, f[2]), max(hi_y, f[3])
            tot += (hi_x - lo_x) + (hi_y - lo_y)
        return float(tot)

    # -- incremental state -------------------------------------------------
    def fresh(self):
        """Bounding boxes of the pins placed so far, per net."""
        m = len(self.I.nets_blocks)
        box = np.empty((m, 4))
        box[:, 0] = np.inf
        box[:, 1] = -np.inf
        box[:, 2] = np.inf
        box[:, 3] = -np.inf
        for k, f in enumerate(self.I.nets_fixed):
            if f is not None:
                box[k] = f
        return box

    @staticmethod
    def _span(box):
        w = np.where(np.isfinite(box[:, 1] - box[:, 0]), box[:, 1] - box[:, 0], 0.0)
        h = np.where(np.isfinite(box[:, 3] - box[:, 2]), box[:, 3] - box[:, 2], 0.0)
        return w + h

    def marginal_grid(self, i, box, cx, cy):
        """(GH, GW) increase in HPWL from placing block `i` at each centre.

        Returned as a *cost* (positive = worse), because HPWL is minimized while
        the rest of this codebase maximizes; callers negate it.
        """
        # cx and cy are usually a row and a column vector, so the result lives
        # on their broadcast, not on either one
        out = np.zeros(np.broadcast_shapes(np.shape(cx), np.shape(cy)), dtype=float)
        for k in self.net_of[i]:
            lo_x, hi_x, lo_y, hi_y = box[k]
            if not np.isfinite(lo_x):
                new_x = np.zeros_like(cx)            # first pin: zero span
            else:
                new_x = np.maximum(hi_x, cx) - np.minimum(lo_x, cx)
            if not np.isfinite(lo_y):
                new_y = np.zeros_like(cy)
            else:
                new_y = np.maximum(hi_y, cy) - np.minimum(lo_y, cy)
            old = 0.0
            if np.isfinite(lo_x):
                old += (hi_x - lo_x) + (hi_y - lo_y)
            out += (new_x + new_y) - old
        return out

    def commit(self, i, box, cx, cy):
        for k in self.net_of[i]:
            box[k, 0] = min(box[k, 0], cx)
            box[k, 1] = max(box[k, 1], cx)
            box[k, 2] = min(box[k, 2], cy)
            box[k, 3] = max(box[k, 3], cy)
        return box
