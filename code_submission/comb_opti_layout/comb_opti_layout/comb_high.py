import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Large office scenario (32 rooms)
# - Standalone
# - No matplotlib
# - Saves paper-style PNG using Pillow
# ============================================================

GRID_W, GRID_H = 36, 24
X_START, Y_START = 0.0, 0.0
X_END, Y_END = float(GRID_W), float(GRID_H)
STEP_SIZE = 1.0

ADJ_WEIGHT = 2.0
EDGE_WEIGHT = 2.0
OVERLAP_WEIGHT = -8.0
NO_OVERLAP_BONUS = 400.0
NO_OVERLAP_MIDDLE_BONUS = 200.0
OVERLAP_BONUS_THRESHOLD = -200.0
DISTANCE_POWER = 1.0

# ------------------------------------------------------------
# Room types and names
# ------------------------------------------------------------
ROOM_TYPES = {
    "CEO1": "executive",
    "CEO2": "executive",
    "Meet1": "meeting",
    "Meet2": "meeting",
    "Meet3": "meeting",
    "Meet4": "meeting",
    "Meet5": "meeting",
    "Open1": "open_office",
    "Open2": "open_office",
    "Open3": "open_office",
    "Open4": "open_office",
    "Open5": "open_office",
    "Open6": "open_office",
    "Open7": "open_office",
    "Open8": "open_office",
    "Phone1": "phone_booth",
    "Phone2": "phone_booth",
    "Phone3": "phone_booth",
    "Phone4": "phone_booth",
    "Bath1": "restroom",
    "Bath2": "restroom",
    "Bath3": "restroom",
    "Pantry1": "pantry",
    "Pantry2": "pantry",
    "Storage1": "storage",
    "Storage2": "storage",
    "Util1": "utility",
    "Util2": "utility",
    "Reception": "reception",
    "Server": "server",
    "Print": "print",
    "Lounge": "lounge",
}

ROOM_ORDER = list(ROOM_TYPES.keys())

TYPE_SIZES = {
    "executive": (3, 4),
    "meeting": (5, 4),
    "open_office": (6, 4),
    "phone_booth": (2, 3),
    "restroom": (2, 4),
    "pantry": (3, 3),
    "storage": (3, 3),
    "utility": (2, 4),
    "reception": (4, 4),
    "server": (3, 3),
    "print": (2, 3),
    "lounge": (4, 4),
}

ROOM_SIZES: Dict[str, Tuple[int, int]] = {
    room: TYPE_SIZES[rtype] for room, rtype in ROOM_TYPES.items()
}

# ------------------------------------------------------------
# Type-based affinity rules (symmetric)
# positive -> prefer close
# negative -> prefer far
# ------------------------------------------------------------
TYPE_ADJ = {
    ("executive", "executive"): 6,
    ("executive", "meeting"): 8,
    ("executive", "open_office"): 3,
    ("executive", "phone_booth"): 2,
    ("executive", "restroom"): -6,
    ("executive", "pantry"): 2,
    ("executive", "storage"): -2,
    ("executive", "utility"): -2,
    ("executive", "reception"): 5,
    ("executive", "server"): -6,
    ("executive", "print"): -2,
    ("executive", "lounge"): 4,

    ("meeting", "meeting"): 3,
    ("meeting", "open_office"): 8,
    ("meeting", "phone_booth"): 5,
    ("meeting", "restroom"): 2,
    ("meeting", "pantry"): 3,
    ("meeting", "storage"): 0,
    ("meeting", "utility"): 0,
    ("meeting", "reception"): 5,
    ("meeting", "server"): -3,
    ("meeting", "print"): 2,
    ("meeting", "lounge"): 4,

    ("open_office", "open_office"): 5,
    ("open_office", "phone_booth"): 8,
    ("open_office", "restroom"): 5,
    ("open_office", "pantry"): 6,
    ("open_office", "storage"): -2,
    ("open_office", "utility"): -2,
    ("open_office", "reception"): 2,
    ("open_office", "server"): -8,
    ("open_office", "print"): 6,
    ("open_office", "lounge"): 5,

    ("phone_booth", "phone_booth"): 2,
    ("phone_booth", "restroom"): 1,
    ("phone_booth", "pantry"): 1,
    ("phone_booth", "storage"): -1,
    ("phone_booth", "utility"): -1,
    ("phone_booth", "reception"): 0,
    ("phone_booth", "server"): -3,
    ("phone_booth", "print"): 2,
    ("phone_booth", "lounge"): 2,

    ("restroom", "restroom"): 3,
    ("restroom", "pantry"): -3,
    ("restroom", "storage"): 1,
    ("restroom", "utility"): 2,
    ("restroom", "reception"): 1,
    ("restroom", "server"): -1,
    ("restroom", "print"): 1,
    ("restroom", "lounge"): 1,

    ("pantry", "pantry"): 1,
    ("pantry", "storage"): -2,
    ("pantry", "utility"): -2,
    ("pantry", "reception"): 2,
    ("pantry", "server"): -4,
    ("pantry", "print"): 2,
    ("pantry", "lounge"): 7,

    ("storage", "storage"): 2,
    ("storage", "utility"): 4,
    ("storage", "reception"): -3,
    ("storage", "server"): 2,
    ("storage", "print"): 2,
    ("storage", "lounge"): -2,

    ("utility", "utility"): 3,
    ("utility", "reception"): -3,
    ("utility", "server"): 2,
    ("utility", "print"): 1,
    ("utility", "lounge"): -2,

    ("reception", "reception"): 0,
    ("reception", "server"): -5,
    ("reception", "print"): 1,
    ("reception", "lounge"): 4,

    ("server", "server"): 1,
    ("server", "print"): 2,
    ("server", "lounge"): -4,

    ("print", "print"): 1,
    ("print", "lounge"): 1,

    ("lounge", "lounge"): 2,
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

# ------------------------------------------------------------
# Colors / labels
# ------------------------------------------------------------
TYPE_COLORS = {
    "executive": (141, 211, 199),
    "meeting": (253, 180, 98),
    "open_office": (128, 177, 211),
    "phone_booth": (190, 186, 218),
    "restroom": (255, 255, 179),
    "pantry": (251, 128, 114),
    "storage": (188, 158, 158),
    "utility": (179, 222, 105),
    "reception": (240, 200, 240),
    "server": (160, 160, 160),
    "print": (220, 220, 120),
    "lounge": (255, 220, 180),
}

ROOM_COLORS = {room: TYPE_COLORS[ROOM_TYPES[room]] for room in ROOM_ORDER}
DISPLAY_NAMES = {room: room for room in ROOM_ORDER}


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

    if rtype in {"restroom", "utility", "storage", "server", "print"}:
        if touches_left or touches_right:
            bonus += EDGE_WEIGHT * room.h
        if touches_bottom or touches_top:
            bonus += EDGE_WEIGHT * room.w

    elif rtype in {"executive", "reception"}:
        if touches_top:
            bonus += 0.8 * EDGE_WEIGHT * room.w

    elif rtype in {"open_office", "meeting", "lounge", "pantry"}:
        if touches_left or touches_right:
            bonus += 0.3 * EDGE_WEIGHT * room.h
        if touches_bottom or touches_top:
            bonus += 0.3 * EDGE_WEIGHT * room.w

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

    move_type = rng.random()
    if move_type < 0.80:
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
        # occasional larger jump helps for larger scenario
        xmin, xmax, ymin, ymax = valid_center_range(room.w, room.h)
        room.x = rng.uniform(xmin, xmax)
        room.y = rng.uniform(ymin, ymax)

    return new_layout


def greedy_polish(layout: Dict[str, Room], max_passes: int = 30) -> Dict[str, Room]:
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
    sample_uniform_random_layout(11),
    sample_uniform_random_layout(22),
    sample_uniform_random_layout(33),
]


def simulated_annealing(
    init_layout: Dict[str, Dict[str, float]],
    seed: int,
    n_steps: int = 45000,
    t0: float = 30.0,
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

    best = greedy_polish(best, max_passes=40)
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
    margin = 35
    title_h = 30
    left = panel_x + margin
    top = panel_y + title_h + 10
    width = panel_w - 2 * margin
    height = panel_h - title_h - 2 * margin

    draw.text((panel_x + 10, panel_y + 5), title, fill=(0, 0, 0), font=title_font)
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


def save_paper_figure(all_inits, all_results, out_path="scenario32_layouts_for_paper.png"):
    canvas_w = 2400
    canvas_h = 1400
    img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = try_load_font(18)
    title_font = try_load_font(28)

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


def main():
    print("=" * 100)
    print("Large office scenario (32 rooms) combinatorial optimization baseline")
    print("=" * 100)
    print_room_summary()

    all_results = []

    for idx, init_cfg in enumerate(SCENARIO32_INIT_CONFIGS, start=1):
        print(f"\nRunning init {idx} ...")
        best_layout, info = simulated_annealing(init_cfg, seed=100 + idx)
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
        out_path="scenario32_layouts_for_paper.png",
    )


if __name__ == "__main__":
    main()