import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Dense practical scenario: 32-room outpatient clinic
# - Standalone
# - No matplotlib
# - Saves paper-style PNG using Pillow
# ============================================================

GRID_W, GRID_H = 28, 16
X_START, Y_START = 0.0, 0.0
X_END, Y_END = float(GRID_W), float(GRID_H)
STEP_SIZE = 1.0

ADJ_WEIGHT = 2.0
EDGE_WEIGHT = 2.0
OVERLAP_WEIGHT = -10.0
NO_OVERLAP_BONUS = 450.0
NO_OVERLAP_MIDDLE_BONUS = 200.0
OVERLAP_BONUS_THRESHOLD = -250.0
DISTANCE_POWER = 1.0

# ============================================================
# Room inventory (32 rooms)
# ============================================================
ROOM_TYPES = {
    "Reception": "reception",
    "Waiting1": "waiting",
    "Waiting2": "waiting",
    "Triage1": "triage",
    "Triage2": "triage",
    "Consult1": "consult",
    "Consult2": "consult",
    "Consult3": "consult",
    "Consult4": "consult",
    "Consult5": "consult",
    "Consult6": "consult",
    "Consult7": "consult",
    "Consult8": "consult",
    "Exam1": "exam",
    "Exam2": "exam",
    "Exam3": "exam",
    "Exam4": "exam",
    "Exam5": "exam",
    "Exam6": "exam",
    "Treat1": "treatment",
    "Treat2": "treatment",
    "Treat3": "treatment",
    "Proc1": "procedure",
    "Proc2": "procedure",
    "Nurse1": "nurse",
    "Nurse2": "nurse",
    "Med1": "med",
    "Clean1": "clean",
    "Soiled1": "soiled",
    "Storage1": "storage",
    "Rest1": "restroom",
    "Rest2": "restroom",
}
ROOM_ORDER = list(ROOM_TYPES.keys())

TYPE_SIZES = {
    "reception": (4, 3),
    "waiting": (4, 4),
    "triage": (3, 3),
    "consult": (3, 3),
    "exam": (3, 3),
    "treatment": (4, 3),
    "procedure": (4, 3),
    "nurse": (3, 3),
    "med": (2, 3),
    "clean": (2, 3),
    "soiled": (2, 3),
    "storage": (3, 3),
    "restroom": (2, 3),
}

ROOM_SIZES: Dict[str, Tuple[int, int]] = {
    room: TYPE_SIZES[rtype] for room, rtype in ROOM_TYPES.items()
}

# ============================================================
# Practical clinic adjacency rules (symmetric)
# positive -> prefer close
# negative -> prefer far
# ============================================================
TYPE_ADJ = {
    ("reception", "waiting"): 10,
    ("reception", "triage"): 8,
    ("reception", "consult"): 4,
    ("reception", "exam"): 2,
    ("reception", "treatment"): 1,
    ("reception", "procedure"): 0,
    ("reception", "nurse"): 3,
    ("reception", "med"): -2,
    ("reception", "clean"): -4,
    ("reception", "soiled"): -10,
    ("reception", "storage"): -4,
    ("reception", "restroom"): 4,

    ("waiting", "waiting"): 4,
    ("waiting", "triage"): 8,
    ("waiting", "consult"): 6,
    ("waiting", "exam"): 3,
    ("waiting", "treatment"): -2,
    ("waiting", "procedure"): -3,
    ("waiting", "nurse"): 0,
    ("waiting", "med"): -3,
    ("waiting", "clean"): -5,
    ("waiting", "soiled"): -10,
    ("waiting", "storage"): -3,
    ("waiting", "restroom"): 6,

    ("triage", "triage"): 2,
    ("triage", "consult"): 6,
    ("triage", "exam"): 7,
    ("triage", "treatment"): 5,
    ("triage", "procedure"): 3,
    ("triage", "nurse"): 6,
    ("triage", "med"): 1,
    ("triage", "clean"): 2,
    ("triage", "soiled"): -3,
    ("triage", "storage"): 1,
    ("triage", "restroom"): 2,

    ("consult", "consult"): 3,
    ("consult", "exam"): 9,
    ("consult", "treatment"): 4,
    ("consult", "procedure"): 3,
    ("consult", "nurse"): 5,
    ("consult", "med"): 1,
    ("consult", "clean"): 1,
    ("consult", "soiled"): -4,
    ("consult", "storage"): 1,
    ("consult", "restroom"): 2,

    ("exam", "exam"): 4,
    ("exam", "treatment"): 8,
    ("exam", "procedure"): 6,
    ("exam", "nurse"): 8,
    ("exam", "med"): 3,
    ("exam", "clean"): 4,
    ("exam", "soiled"): -2,
    ("exam", "storage"): 2,
    ("exam", "restroom"): 1,

    ("treatment", "treatment"): 3,
    ("treatment", "procedure"): 7,
    ("treatment", "nurse"): 10,
    ("treatment", "med"): 9,
    ("treatment", "clean"): 8,
    ("treatment", "soiled"): 4,
    ("treatment", "storage"): 3,
    ("treatment", "restroom"): 1,

    ("procedure", "procedure"): 2,
    ("procedure", "nurse"): 8,
    ("procedure", "med"): 6,
    ("procedure", "clean"): 6,
    ("procedure", "soiled"): 5,
    ("procedure", "storage"): 2,
    ("procedure", "restroom"): 0,

    ("nurse", "nurse"): 3,
    ("nurse", "med"): 8,
    ("nurse", "clean"): 7,
    ("nurse", "soiled"): 3,
    ("nurse", "storage"): 2,
    ("nurse", "restroom"): 1,

    ("med", "med"): 2,
    ("med", "clean"): 5,
    ("med", "soiled"): -4,
    ("med", "storage"): 2,
    ("med", "restroom"): -1,

    ("clean", "clean"): 1,
    ("clean", "soiled"): -8,
    ("clean", "storage"): 4,
    ("clean", "restroom"): -1,

    ("soiled", "soiled"): 1,
    ("soiled", "storage"): 2,
    ("soiled", "restroom"): 0,

    ("storage", "storage"): 2,
    ("storage", "restroom"): 0,

    ("restroom", "restroom"): 2,
}


def get_affinity(t1: str, t2: str) -> float:
    if (t1, t2) in TYPE_ADJ:
        return TYPE_ADJ[(t1, t2)]
    if (t2, t1) in TYPE_ADJ:
        return TYPE_ADJ[(t2, t1)]
    return 0.0


ADJ: Dict[str, Dict[str, float]] = {}
for r1 in ROOM_ORDER:
    ADJ[r1] = {}
    for r2 in ROOM_ORDER:
        if r1 == r2:
            continue
        ADJ[r1][r2] = get_affinity(ROOM_TYPES[r1], ROOM_TYPES[r2])

# ============================================================
# Colors / labels
# ============================================================
TYPE_COLORS = {
    "reception": (220, 190, 225),
    "waiting": (235, 220, 180),
    "triage": (255, 210, 130),
    "consult": (141, 211, 199),
    "exam": (128, 177, 211),
    "treatment": (251, 128, 114),
    "procedure": (255, 170, 170),
    "nurse": (190, 186, 218),
    "med": (245, 160, 120),
    "clean": (179, 222, 105),
    "soiled": (120, 120, 120),
    "storage": (188, 158, 158),
    "restroom": (255, 255, 179),
}
ROOM_COLORS = {room: TYPE_COLORS[ROOM_TYPES[room]] for room in ROOM_ORDER}

DISPLAY_NAMES = {
    "Reception": "Rec",
    "Waiting1": "Wait1",
    "Waiting2": "Wait2",
    "Triage1": "Tri1",
    "Triage2": "Tri2",
    "Consult1": "Con1",
    "Consult2": "Con2",
    "Consult3": "Con3",
    "Consult4": "Con4",
    "Consult5": "Con5",
    "Consult6": "Con6",
    "Consult7": "Con7",
    "Consult8": "Con8",
    "Exam1": "Ex1",
    "Exam2": "Ex2",
    "Exam3": "Ex3",
    "Exam4": "Ex4",
    "Exam5": "Ex5",
    "Exam6": "Ex6",
    "Treat1": "Tr1",
    "Treat2": "Tr2",
    "Treat3": "Tr3",
    "Proc1": "P1",
    "Proc2": "P2",
    "Nurse1": "N1",
    "Nurse2": "N2",
    "Med1": "Med",
    "Clean1": "Clean",
    "Soiled1": "Soil",
    "Storage1": "Store",
    "Rest1": "Rest1",
    "Rest2": "Rest2",
}


@dataclass
class Room:
    x: float
    y: float
    w: float
    h: float


def copy_layout(layout: Dict[str, Dict[str, float]]) -> Dict[str, Room]:
    return {
        name: Room(
            float(layout[name]["x"]),
            float(layout[name]["y"]),
            float(layout[name]["width"]),
            float(layout[name]["height"]),
        )
        for name in ROOM_ORDER
    }


def to_serializable(layout: Dict[str, Room]) -> Dict[str, Dict[str, float]]:
    return {
        k: {"x": float(v.x), "y": float(v.y), "width": float(v.w), "height": float(v.h)}
        for k, v in layout.items()
    }


def valid_center_range(w: float, h: float):
    return (X_START + w / 2.0, X_END - w / 2.0, Y_START + h / 2.0, Y_END - h / 2.0)


def clamp_room(room: Room) -> Room:
    xmin, xmax, ymin, ymax = valid_center_range(room.w, room.h)
    room.x = min(max(room.x, xmin), xmax)
    room.y = min(max(room.y, ymin), ymax)
    return room


def validate_initial_layout(layout: Dict[str, Dict[str, float]]):
    for name in ROOM_ORDER:
        if name not in layout:
            raise ValueError(f"Missing room: {name}")
        x = float(layout[name]["x"])
        y = float(layout[name]["y"])
        w = float(layout[name]["width"])
        h = float(layout[name]["height"])
        exp_w, exp_h = ROOM_SIZES[name]
        if (w, h) != (exp_w, exp_h):
            raise ValueError(f"Size mismatch for {name}: got {(w, h)}, expected {(exp_w, exp_h)}")
        xmin, xmax, ymin, ymax = valid_center_range(w, h)
        if not (xmin <= x <= xmax and ymin <= y <= ymax):
            raise ValueError(
                f"Out of bounds for {name}: (x={x}, y={y}) not in x[{xmin},{xmax}] y[{ymin},{ymax}]"
            )


def overlap_area(a: Room, b: Room) -> float:
    ax0, ax1 = a.x - a.w / 2.0, a.x + a.w / 2.0
    ay0, ay1 = a.y - a.h / 2.0, a.y + a.h / 2.0
    bx0, bx1 = b.x - b.w / 2.0, b.x + b.w / 2.0
    by0, by1 = b.y - b.h / 2.0, b.y + b.h / 2.0
    ox = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    oy = max(0.0, min(ay1, by1) - max(ay0, by0))
    return ox * oy


def cal_abs_tangent(x1: float, y1: float, x2: float, y2: float) -> float:
    return abs((y2 - y1) / (x2 - x1))


def cal_junk_distance(tan: float, w: float, h: float) -> float:
    half_w = w / 2.0
    half_h = h / 2.0
    if tan > h / w:
        kw = half_h / tan
        kh = half_h
    else:
        kw = half_w
        kh = half_w * tan
    return math.sqrt(kw * kw + kh * kh)


def between_distance(a: Room, b: Room) -> float:
    x1, y1, w1, h1 = a.x, a.y, a.w, a.h
    x2, y2, w2, h2 = b.x, b.y, b.w, b.h
    if abs(x1 - x2) <= 1e-12:
        return abs(y2 - y1) - (h1 + h2) / 2.0
    if abs(y1 - y2) <= 1e-12:
        return abs(x2 - x1) - (w1 + w2) / 2.0
    tan = cal_abs_tangent(x1, y1, x2, y2)
    jd1 = cal_junk_distance(tan, w1, h1)
    jd2 = cal_junk_distance(tan, w2, h2)
    return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) - jd1 - jd2


def max_distance_for_pair(a: Room, b: Room) -> float:
    aa = Room(X_START + a.w / 2.0, Y_START + a.h / 2.0, a.w, a.h)
    bb = Room(X_END - b.w / 2.0, Y_END - b.h / 2.0, b.w, b.h)
    return between_distance(aa, bb)


def total_overlap_area(layout: Dict[str, Room]) -> float:
    total = 0.0
    for i, ni in enumerate(ROOM_ORDER):
        for nj in ROOM_ORDER[i + 1:]:
            total += overlap_area(layout[ni], layout[nj])
    return total


def overlap_term(layout: Dict[str, Room]) -> float:
    penalty = OVERLAP_WEIGHT * total_overlap_area(layout)
    if abs(penalty) <= 1e-12:
        bonus = NO_OVERLAP_BONUS
    elif OVERLAP_BONUS_THRESHOLD <= penalty < 0.0:
        bonus = NO_OVERLAP_MIDDLE_BONUS * (1.0 - penalty / OVERLAP_BONUS_THRESHOLD) ** 2
    else:
        bonus = 0.0
    return penalty + bonus


def adjacency_term(layout: Dict[str, Room]) -> float:
    total = 0.0
    for i, ni in enumerate(ROOM_ORDER):
        for nj in ROOM_ORDER[i + 1:]:
            wij = ADJ.get(ni, {}).get(nj, 0.0)
            if overlap_area(layout[ni], layout[nj]) > 0.0:
                continue
            dmax = max_distance_for_pair(layout[ni], layout[nj])
            nd = between_distance(layout[ni], layout[nj]) / dmax if dmax > 0 else 0.0
            nd = min(max(nd, 0.0), 1.0) ** DISTANCE_POWER
            if wij >= 0:
                total += wij * nd
            else:
                total += (-wij) * (1.0 - nd)
    return ADJ_WEIGHT * total


def edge_bonus_for_room(name: str, room: Room, layout: Dict[str, Room]) -> float:
    rtype = ROOM_TYPES[name]
    if any(overlap_area(room, layout[other]) > 0.0 for other in ROOM_ORDER if other != name):
        return 0.0

    x0, x1 = room.x - room.w / 2.0, room.x + room.w / 2.0
    y0, y1 = room.y - room.h / 2.0, room.y + room.h / 2.0

    touches_left = abs(x0 - X_START) < 1e-12
    touches_right = abs(x1 - X_END) < 1e-12
    touches_bottom = abs(y0 - Y_START) < 1e-12
    touches_top = abs(y1 - Y_END) < 1e-12

    bonus = 0.0

    # public-facing rooms near "front" (top boundary in our figure)
    if rtype in {"reception", "waiting", "triage"}:
        if touches_top:
            bonus += 1.0 * EDGE_WEIGHT * room.w

    # utility / sanitary rooms like wall access
    if rtype in {"restroom", "clean", "soiled", "storage", "med"}:
        if touches_left or touches_right:
            bonus += 0.8 * EDGE_WEIGHT * room.h
        if touches_bottom or touches_top:
            bonus += 0.8 * EDGE_WEIGHT * room.w

    # consult / exam / treatment mildly like side boundaries
    if rtype in {"consult", "exam", "treatment", "procedure", "nurse"}:
        if touches_left or touches_right:
            bonus += 0.25 * EDGE_WEIGHT * room.h

    return bonus


def edge_term(layout: Dict[str, Room]) -> float:
    return sum(edge_bonus_for_room(name, layout[name], layout) for name in ROOM_ORDER)


def objective(layout: Dict[str, Room]):
    adj = adjacency_term(layout)
    edge = edge_term(layout)
    overlap = overlap_term(layout)
    total = adj + edge + overlap
    return total, {
        "adj": adj,
        "edge": edge,
        "overlap": overlap,
        "overlap_area": total_overlap_area(layout),
    }


def propose_move(layout: Dict[str, Room], rng: random.Random) -> Dict[str, Room]:
    new_layout = {k: Room(v.x, v.y, v.w, v.h) for k, v in layout.items()}
    room_name = rng.choice(ROOM_ORDER)
    room = new_layout[room_name]

    u = rng.random()
    if u < 0.80:
        dx, dy = rng.choice([
            (STEP_SIZE, 0.0),
            (-STEP_SIZE, 0.0),
            (0.0, STEP_SIZE),
            (0.0, -STEP_SIZE),
        ])
        room.x += dx
        room.y += dy
        clamp_room(room)
    else:
        xmin, xmax, ymin, ymax = valid_center_range(room.w, room.h)
        room.x = rng.uniform(xmin, xmax)
        room.y = rng.uniform(ymin, ymax)

    return new_layout


def greedy_polish(layout: Dict[str, Room], max_passes: int = 25) -> Dict[str, Room]:
    current = {k: Room(v.x, v.y, v.w, v.h) for k, v in layout.items()}
    current_score, _ = objective(current)
    directions = [
        (STEP_SIZE, 0.0),
        (-STEP_SIZE, 0.0),
        (0.0, STEP_SIZE),
        (0.0, -STEP_SIZE),
    ]

    for _ in range(max_passes):
        improved = False
        best_local_score = current_score
        best_local_layout = current

        for name in ROOM_ORDER:
            for dx, dy in directions:
                cand = {k: Room(v.x, v.y, v.w, v.h) for k, v in current.items()}
                cand[name].x += dx
                cand[name].y += dy
                clamp_room(cand[name])
                score, _ = objective(cand)
                if score > best_local_score + 1e-12:
                    best_local_score = score
                    best_local_layout = cand
                    improved = True

        if not improved:
            break
        current = best_local_layout
        current_score = best_local_score

    return current


def sample_uniform_random_layout(seed: int) -> Dict[str, Dict[str, float]]:
    rng = random.Random(seed)
    layout = {}
    for name in ROOM_ORDER:
        w, h = ROOM_SIZES[name]
        xmin, xmax, ymin, ymax = valid_center_range(w, h)
        layout[name] = {
            "x": rng.uniform(xmin, xmax),
            "y": rng.uniform(ymin, ymax),
            "width": w,
            "height": h,
        }
    return layout


SCENARIO32_INIT_CONFIGS = [
    sample_uniform_random_layout(101),
    sample_uniform_random_layout(202),
    sample_uniform_random_layout(303),
]


def simulated_annealing(
    init_layout: Dict[str, Dict[str, float]],
    seed: int,
    n_steps: int = 50000,
    t0: float = 32.0,
    t1: float = 0.08,
    greedy_every: int = 5000,
):
    validate_initial_layout(init_layout)
    rng = random.Random(seed)
    current = copy_layout(init_layout)
    current_score, current_info = objective(current)

    best = {k: Room(v.x, v.y, v.w, v.h) for k, v in current.items()}
    best_score = current_score
    best_info = dict(current_info)

    for step in range(1, n_steps + 1):
        frac = (step - 1) / max(1, n_steps - 1)
        temp = t0 * ((t1 / t0) ** frac)

        cand = propose_move(current, rng)
        cand_score, cand_info = objective(cand)
        delta = cand_score - current_score

        if delta >= 0.0 or rng.random() < math.exp(delta / max(temp, 1e-12)):
            current = cand
            current_score = cand_score

            if cand_score > best_score:
                best = {k: Room(v.x, v.y, v.w, v.h) for k, v in cand.items()}
                best_score = cand_score
                best_info = dict(cand_info)

        if greedy_every > 0 and step % greedy_every == 0:
            polished = greedy_polish(best, max_passes=20)
            polished_score, polished_info = objective(polished)
            if polished_score > best_score:
                best = polished
                best_score = polished_score
                best_info = dict(polished_info)

    best = greedy_polish(best, max_passes=35)
    best_score, best_info = objective(best)
    return best, {"total": best_score, **best_info}


def ascii_layout(layout: Dict[str, Room]) -> str:
    grid = [["." for _ in range(GRID_W)] for _ in range(GRID_H)]
    labels = {}
    for i, name in enumerate(ROOM_ORDER):
        labels[name] = chr(65 + (i % 26))

    for name in ROOM_ORDER:
        r = layout[name]
        x0 = int(round(r.x - r.w / 2.0))
        x1 = int(round(r.x + r.w / 2.0))
        y0 = int(round(r.y - r.h / 2.0))
        y1 = int(round(r.y + r.h / 2.0))
        for yy in range(max(0, y0), min(GRID_H, y1)):
            for xx in range(max(0, x0), min(GRID_W, x1)):
                if grid[yy][xx] == ".":
                    grid[yy][xx] = labels[name]
                else:
                    grid[yy][xx] = "X"

    return "\n".join(" ".join(row) for row in reversed(grid))


def try_load_font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                pass
    return ImageFont.load_default()


def draw_panel(
    draw: ImageDraw.ImageDraw,
    layout: Dict[str, Room],
    panel_x: int,
    panel_y: int,
    panel_w: int,
    panel_h: int,
    title: str,
    font,
    title_font,
):
    margin = 28
    title_h = 30
    left = panel_x + margin
    top = panel_y + title_h + 10
    width = panel_w - 2 * margin
    height = panel_h - title_h - 2 * margin

    draw.text((panel_x + 8, panel_y + 5), title, fill=(0, 0, 0), font=title_font)
    draw.rectangle([left, top, left + width, top + height], outline=(0, 0, 0), width=2)

    cell_w = width / GRID_W
    cell_h = height / GRID_H

    for gx in range(GRID_W + 1):
        x = left + gx * cell_w
        draw.line([(x, top), (x, top + height)], fill=(215, 215, 215), width=1)
    for gy in range(GRID_H + 1):
        y = top + gy * cell_h
        draw.line([(left, y), (left + width, y)], fill=(215, 215, 215), width=1)

    for name in ROOM_ORDER:
        r = layout[name]
        x0 = left + (r.x - r.w / 2.0) * cell_w
        x1 = left + (r.x + r.w / 2.0) * cell_w
        y0 = top + (GRID_H - (r.y + r.h / 2.0)) * cell_h
        y1 = top + (GRID_H - (r.y - r.h / 2.0)) * cell_h

        draw.rectangle([x0, y0, x1, y1], fill=ROOM_COLORS[name], outline=(0, 0, 0), width=2)

        label = DISPLAY_NAMES[name]
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        tx = (x0 + x1) / 2 - tw / 2
        ty = (y0 + y1) / 2 - th / 2
        draw.text((tx, ty), label, fill=(0, 0, 0), font=font)


def save_paper_figure(all_inits, all_results, out_path="clinic32_layouts_for_paper.png"):
    canvas_w = 2400
    canvas_h = 1300
    img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = try_load_font(15)
    title_font = try_load_font(26)

    cols = 3
    rows = 2
    panel_w = canvas_w // cols
    panel_h = canvas_h // rows

    for i in range(3):
        init_layout = copy_layout(all_inits[i])
        result_layout = all_results[i][0]
        info = all_results[i][1]

        draw_panel(
            draw,
            init_layout,
            i * panel_w,
            0,
            panel_w,
            panel_h,
            f"Initial configuration {i+1}",
            font,
            title_font,
        )

        draw_panel(
            draw,
            result_layout,
            i * panel_w,
            panel_h,
            panel_w,
            panel_h,
            f"Optimized layout {i+1} | Reward = {info['total']:.2f}",
            font,
            title_font,
        )

    abs_path = os.path.abspath(out_path)
    img.save(abs_path)
    print(f"[saved] {abs_path}")


def print_room_summary():
    type_count = {}
    for r in ROOM_ORDER:
        t = ROOM_TYPES[r]
        type_count[t] = type_count.get(t, 0) + 1

    print("Room counts by type:")
    for k in sorted(type_count.keys()):
        print(f"  {k}: {type_count[k]}")
    print(f"Total rooms: {len(ROOM_ORDER)}")
    print(f"Grid size: {GRID_W} x {GRID_H}")
    total_area = sum(ROOM_SIZES[r][0] * ROOM_SIZES[r][1] for r in ROOM_ORDER)
    print(f"Total room area: {total_area}")
    print(f"Floor area: {GRID_W * GRID_H}")
    print(f"Occupancy ratio: {total_area / (GRID_W * GRID_H):.3f}")


def main():
    print("=" * 100)
    print("Dense outpatient clinic scenario (32 rooms) combinatorial optimization baseline")
    print("=" * 100)
    print_room_summary()

    all_results = []

    for idx, init_cfg in enumerate(SCENARIO32_INIT_CONFIGS, start=1):
        print(f"\nRunning init {idx} ...")
        best_layout, info = simulated_annealing(init_cfg, seed=700 + idx)
        all_results.append((best_layout, info))

        print(
            f"[Init {idx}] total={info['total']:.4f} | "
            f"adj={info['adj']:.4f} | edge={info['edge']:.4f} | "
            f"overlap={info['overlap']:.4f} | overlap_area={info['overlap_area']:.4f}"
        )
        print("layout dict =")
        print(to_serializable(best_layout))
        print("-" * 100)

    save_paper_figure(
        SCENARIO32_INIT_CONFIGS,
        all_results,
        out_path="clinic32_layouts_for_paper.png",
    )


if __name__ == "__main__":
    main()