import math
import os
import random
from dataclasses import dataclass
from typing import Dict, Tuple

from PIL import Image, ImageDraw, ImageFont

# ============================================================
# Scenario 1 combinatorial optimization baseline
# - Standalone
# - No matplotlib
# - Saves paper-style PNG using Pillow
# ============================================================

GRID_W, GRID_H = 14, 10
X_START, Y_START = 0.0, 0.0
X_END, Y_END = float(GRID_W), float(GRID_H)
STEP_SIZE = 1.0

ROOM_ORDER = [
    "CEO R",
    "Bathroom",
    "Cafeteria",
    "Office1",
    "Office2",
    "MeetingRoom",
    "Utility R",
]

ROOM_SIZES: Dict[str, Tuple[int, int]] = {
    "CEO R": (3, 4),
    "Bathroom": (2, 4),
    "Cafeteria": (3, 4),
    "Office1": (3, 4),
    "Office2": (8, 4),
    "MeetingRoom": (5, 4),
    "Utility R": (2, 4),
}

ADJ = {
    "CEO R": {"Bathroom": 10, "Cafeteria": 10, "Office1": 0, "Office2": 5, "MeetingRoom": 0, "Utility R": 10},
    "Bathroom": {"CEO R": 10, "Cafeteria": -10, "Office1": 0, "Office2": 5, "MeetingRoom": 0, "Utility R": -10},
    "Cafeteria": {"CEO R": 10, "Bathroom": -10, "Office1": 0, "Office2": 5, "MeetingRoom": 0, "Utility R": 0},
    "Office1": {"CEO R": 0, "Bathroom": 0, "Cafeteria": 0, "Office2": 5, "MeetingRoom": -10, "Utility R": 0},
    "Office2": {"CEO R": 5, "Bathroom": 5, "Cafeteria": 5, "Office1": 5, "MeetingRoom": 5, "Utility R": 5},
    "MeetingRoom": {"CEO R": 0, "Bathroom": 0, "Cafeteria": 0, "Office1": -10, "Office2": 5, "Utility R": 0},
    "Utility R": {"CEO R": 10, "Bathroom": -10, "Cafeteria": 0, "Office1": 0, "Office2": 5, "MeetingRoom": 0},
}

ADJ_WEIGHT = 2.0
EDGE_WEIGHT = 3.0
OVERLAP_WEIGHT = -5.0
NO_OVERLAP_BONUS = 200.0
NO_OVERLAP_MIDDLE_BONUS = 200.0
OVERLAP_BONUS_THRESHOLD = -100.0
DISTANCE_POWER = 1.0

SCENARIO1_INIT_CONFIGS = [
    # Init 1
    {
        "CEO R":       {"x": 6.0,  "y": 6.0, "width": 3, "height": 4},
        "Bathroom":    {"x": 7.5,  "y": 6.0, "width": 2, "height": 4},
        "Cafeteria":   {"x": 9.5,  "y": 4.5, "width": 3, "height": 4},
        "Office1":     {"x": 10.0, "y": 3.5, "width": 3, "height": 4},
        "Office2":     {"x": 8.0,  "y": 6.0, "width": 8, "height": 4},
        "MeetingRoom": {"x": 5.0,  "y": 2.5, "width": 5, "height": 4},
        "Utility R":   {"x": 11.5, "y": 8.0, "width": 2, "height": 4},
    },

    # Init 2
    {
        "CEO R":       {"x": 4.0,  "y": 8.0, "width": 3, "height": 4},
        "Bathroom":    {"x": 10.0, "y": 8.0, "width": 2, "height": 4},
        "Cafeteria":   {"x": 6.0,  "y": 5.5, "width": 3, "height": 4},
        "Office1":     {"x": 8.5,  "y": 4.5, "width": 3, "height": 4},
        "Office2":     {"x": 4.5,  "y": 2.5, "width": 8, "height": 4},
        "MeetingRoom": {"x": 9.5,  "y": 5.5, "width": 5, "height": 4},
        "Utility R":   {"x": 11.5, "y": 6.5, "width": 2, "height": 4},
    },

    # Init 3
    {
        "CEO R":       {"x": 2.5,  "y": 5.0, "width": 3, "height": 4},
        "Bathroom":    {"x": 3.0,  "y": 8.0, "width": 2, "height": 4},
        "Cafeteria":   {"x": 3.0,  "y": 3.0, "width": 3, "height": 4},
        "Office1":     {"x": 8.0,  "y": 3.5, "width": 3, "height": 4},
        "Office2":     {"x": 4.5,  "y": 2.5, "width": 8, "height": 4},
        "MeetingRoom": {"x": 8.5,  "y": 4.0, "width": 5, "height": 4},
        "Utility R":   {"x": 12.0, "y": 6.5, "width": 2, "height": 4},
    },
]

ROOM_COLORS = {
    "CEO R": (141, 211, 199),
    "Bathroom": (255, 255, 179),
    "Cafeteria": (190, 186, 218),
    "Office1": (251, 128, 114),
    "Office2": (128, 177, 211),
    "MeetingRoom": (253, 180, 98),
    "Utility R": (179, 222, 105),
}

DISPLAY_NAMES = {
    "CEO R": "CEO",
    "Bathroom": "Bath",
    "Cafeteria": "Cafe",
    "Office1": "Off1",
    "Office2": "Off2",
    "MeetingRoom": "Meet",
    "Utility R": "Util",
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


def edge_term(layout: Dict[str, Room]) -> float:
    total = 0.0
    for ni in ROOM_ORDER:
        a = layout[ni]
        if any(overlap_area(a, layout[nj]) > 0.0 for nj in ROOM_ORDER if nj != ni):
            continue
        x0, x1 = a.x - a.w / 2.0, a.x + a.w / 2.0
        y0, y1 = a.y - a.h / 2.0, a.y + a.h / 2.0
        if abs(x0 - X_START) < 1e-12 or abs(x1 - X_END) < 1e-12:
            total += EDGE_WEIGHT * a.h
        if abs(y0 - Y_START) < 1e-12 or abs(y1 - Y_END) < 1e-12:
            total += EDGE_WEIGHT * a.w
    return total


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
    dx, dy = rng.choice([
        (STEP_SIZE, 0.0),
        (-STEP_SIZE, 0.0),
        (0.0, STEP_SIZE),
        (0.0, -STEP_SIZE),
    ])
    room.x += dx
    room.y += dy
    clamp_room(room)
    return new_layout


def greedy_polish(layout: Dict[str, Room], max_passes: int = 40) -> Dict[str, Room]:
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


def simulated_annealing(
    init_layout: Dict[str, Dict[str, float]],
    seed: int,
    n_steps: int = 20000,
    t0: float = 20.0,
    t1: float = 0.05,
    greedy_every: int = 2000,
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

    best = greedy_polish(best, max_passes=50)
    best_score, best_info = objective(best)
    return best, {"total": best_score, **best_info}


def ascii_layout(layout: Dict[str, Room]) -> str:
    grid = [["." for _ in range(GRID_W)] for _ in range(GRID_H)]
    labels = {
        "CEO R": "C",
        "Bathroom": "B",
        "Cafeteria": "F",
        "Office1": "1",
        "Office2": "2",
        "MeetingRoom": "M",
        "Utility R": "U",
    }
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


def draw_panel(draw: ImageDraw.ImageDraw, layout: Dict[str, Room], panel_x: int, panel_y: int,
               panel_w: int, panel_h: int, title: str, font, title_font):
    margin = 35
    title_h = 28
    left = panel_x + margin
    top = panel_y + title_h + 10
    width = panel_w - 2 * margin
    height = panel_h - title_h - 2 * margin

    draw.text((panel_x + 10, panel_y + 5), title, fill=(0, 0, 0), font=title_font)

    # Border
    draw.rectangle([left, top, left + width, top + height], outline=(0, 0, 0), width=2)

    cell_w = width / GRID_W
    cell_h = height / GRID_H

    # Grid lines
    for gx in range(GRID_W + 1):
        x = left + gx * cell_w
        draw.line([(x, top), (x, top + height)], fill=(210, 210, 210), width=1)
    for gy in range(GRID_H + 1):
        y = top + gy * cell_h
        draw.line([(left, y), (left + width, y)], fill=(210, 210, 210), width=1)

    # Rooms
    for name in ROOM_ORDER:
        r = layout[name]
        x0 = left + (r.x - r.w / 2.0) * cell_w
        x1 = left + (r.x + r.w / 2.0) * cell_w
        # y-axis flipped for image coordinates
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


def save_paper_figure(all_inits, all_results, out_path="scenario1_layouts_for_paper.png"):
    canvas_w = 1800
    canvas_h = 1000
    img = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    font = try_load_font(22)
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


def main():
    print("=" * 80)
    print("Scenario 1 combinatorial optimization baseline")
    print("=" * 80)

    all_results = []

    for idx, init_cfg in enumerate(SCENARIO1_INIT_CONFIGS, start=1):
        best_layout, info = simulated_annealing(init_cfg, seed=100 + idx)
        all_results.append((best_layout, info))

        print(
            f"[Init {idx}] total={info['total']:.4f} | "
            f"adj={info['adj']:.4f} | edge={info['edge']:.4f} | "
            f"overlap={info['overlap']:.4f} | overlap_area={info['overlap_area']:.4f}"
        )
        print("layout dict =")
        print(to_serializable(best_layout))
        print("ascii layout =")
        print(ascii_layout(best_layout))
        print("-" * 80)

    save_paper_figure(
        SCENARIO1_INIT_CONFIGS,
        all_results,
        out_path="scenario1_layouts_for_paper.png",
    )


if __name__ == "__main__":
    main()