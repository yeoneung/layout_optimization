"""Learned switch time (X15): predict when to move from partial repair to full
regeneration from instance features that need no search.

Training data are runs of SW:fixed<tau> for a grid of tau on a train-split
phase directory.  For each instance the label is the tau with the best anytime
score (mean improvement over the recorded budgets from 1 s upward, or the
60-second value).  A gradient-boosted regressor on log(tau) is fitted; at
prediction time the output is clipped to the training grid range.

Usage
-----
python learned_switch.py fit <phase_dir> [<phase_dir> ...] --out ../models/swl_v1.pkl
python learned_switch.py evaluate <phase_dir> ...        (leave-one-size-out and
                                                          leave-one-cell-out on the same data)

The token SWL:<name> (methods_ext) loads ../models/<name>.pkl, computes the
features of the instance at hand, predicts tau and runs SW:fixed<tau>.
"""

import argparse
import json
import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARCHIVE = HERE.parents[1] / "code_submission" / "rl_layout" / "experiments"
sys.path.insert(0, str(ARCHIVE))
sys.path.insert(0, str(HERE))

MODELS = HERE.parent / "models"
FEATURE_NAMES = ["log_n", "achieved_fill", "log_plate", "plate_aspect", "mean_area_norm",
                 "cv_area", "max_area_norm", "mean_facility_aspect", "initial_J_per_n",
                 "affinity_density", "mean_abs_affinity", "wall_pref_fraction"]
SW_PATTERN = re.compile(r"^SW:fixed(\d+(?:\.\d+)?)$")


def instance_features(problem, initial_value):
    """Search-free features of one instance (LayoutProblem with B = 1)."""
    iw = problem.iw.astype(float)
    ih = problem.ih.astype(float)
    area = iw * ih
    plate = float(problem.gw * problem.gh)
    n = problem.n
    adj = problem.inst.adj[0]
    off = ~np.eye(n, dtype=bool)
    wall = (problem.inst.edge_side[0] > 0) | (problem.inst.edge_bottom[0] > 0) | (problem.inst.edge_top[0] > 0)
    return {
        "log_n": float(np.log2(n)),
        "achieved_fill": float(area.sum() / plate),
        "log_plate": float(np.log(plate)),
        "plate_aspect": float(max(problem.gw, problem.gh) / max(min(problem.gw, problem.gh), 1)),
        "mean_area_norm": float(area.mean() / plate * n),
        "cv_area": float(area.std() / max(area.mean(), 1e-9)),
        "max_area_norm": float(area.max() / plate),
        "mean_facility_aspect": float(np.mean(np.maximum(iw, ih) / np.maximum(np.minimum(iw, ih), 1))),
        "initial_J_per_n": float(initial_value / n),
        "affinity_density": float(np.mean(adj[off] != 0)),
        "mean_abs_affinity": float(np.mean(np.abs(adj[off]))),
        "wall_pref_fraction": float(np.mean(wall)),
    }


def feature_vector(features):
    return np.array([features[name] for name in FEATURE_NAMES], dtype=float)


def _load_phase(directory):
    from extended_protocol import generate_cell
    from completion_search import LayoutProblem

    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))["settings"]
    inits = {}
    for path in (directory / "initializations").glob("*.json"):
        init = json.loads(path.read_text(encoding="utf-8"))
        if init["initialized"]:
            inits[(init["geometry"], init["n"], init["fill"], init["instance"])] = init
    runs = defaultdict(lambda: defaultdict(list))   # instance -> tau -> [anytime rows]
    for path in (directory / "runs").glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        match = SW_PATTERN.match(run["method"])
        if not match:
            continue
        key = (run["geometry"], run["n"], run["fill"], run["instance"])
        runs[key][float(match.group(1))].append({r["budget_s"]: r["improvement_pct"] for r in run["rows"]})
    cells = {}
    for cell in protocol["cells"]:
        cells[(cell["geometry"], cell["n"], cell["fill"])] = generate_cell(protocol, cell)
    records = []
    for key, by_tau in runs.items():
        if key not in inits:
            continue
        generated = cells[(key[0], key[1], key[2])]
        problem = LayoutProblem(generated["inst"].take([key[3]]))
        feats = instance_features(problem, inits[key]["initial_J"])
        records.append({"key": key, "features": feats, "by_tau": by_tau})
    return protocol, records


def _label(by_tau, objective):
    scores = {}
    for tau, rows in by_tau.items():
        budgets = sorted(rows[0])
        if objective == "final":
            scores[tau] = float(np.mean([r[budgets[-1]] for r in rows]))
        else:
            use = [b for b in budgets if b >= 1.0]
            scores[tau] = float(np.mean([np.mean([r[b] for b in use]) for r in rows]))
    best = max(scores, key=lambda t: scores[t])
    return best, scores


def build_dataset(directories, objective):
    X, y, groups, meta = [], [], [], []
    for directory in directories:
        _, records = _load_phase(directory)
        for rec in records:
            best, scores = _label(rec["by_tau"], objective)
            X.append(feature_vector(rec["features"]))
            y.append(np.log(best))
            groups.append(rec["key"])
            meta.append(scores)
    return np.array(X), np.array(y), groups, meta


def make_model():
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                     subsample=0.8, random_state=0)


def fit(directories, out, objective="auc"):
    X, y, groups, meta = build_dataset(directories, objective)
    if len(y) < 8:
        raise RuntimeError("too few training instances: {}".format(len(y)))
    model = make_model().fit(X, y)
    taus = sorted({t for scores in meta for t in scores})
    bundle = {"model": model, "features": FEATURE_NAMES, "objective": objective,
              "tau_min": min(taus), "tau_max": max(taus), "tau_grid": taus,
              "training_instances": len(y), "training_dirs": [str(Path(d).resolve()) for d in directories],
              "label_distribution": {str(t): int(np.sum(np.isclose(np.exp(y), t))) for t in taus}}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump(bundle, handle)
    print("saved", out, "instances", len(y), "labels", bundle["label_distribution"])
    return bundle


def load_bundle(name):
    path = MODELS / (name + ".pkl")
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_tau(bundle, features):
    x = feature_vector(features)[None, :]
    tau = float(np.exp(bundle["model"].predict(x)[0]))
    return float(min(max(tau, bundle["tau_min"]), bundle["tau_max"]))


ARM_METHODS = {"M1": ("M1", 4), "CAP:8": ("M1", 8), "CAP:16": ("M1", 16), "M0": ("M0", None)}


def _load_arm_runs(directory):
    """instance key -> arm method -> {budget: improvement_pct} (seed-averaged)."""
    from extended_protocol import generate_cell
    from completion_search import LayoutProblem

    directory = Path(directory)
    protocol = json.loads((directory / "protocol.json").read_text(encoding="utf-8"))["settings"]
    inits = {}
    for path in (directory / "initializations").glob("*.json"):
        init = json.loads(path.read_text(encoding="utf-8"))
        if init["initialized"]:
            inits[(init["geometry"], init["n"], init["fill"], init["instance"])] = init
    runs = defaultdict(lambda: defaultdict(list))
    for path in (directory / "runs").glob("*.json"):
        run = json.loads(path.read_text(encoding="utf-8"))
        if run["method"] not in ARM_METHODS:
            continue
        key = (run["geometry"], run["n"], run["fill"], run["instance"])
        runs[key][run["method"]].append({r["budget_s"]: r["improvement_pct"] for r in run["rows"]})
    cells = {(c["geometry"], c["n"], c["fill"]): generate_cell(protocol, c) for c in protocol["cells"]}
    records = []
    for key, by_method in runs.items():
        if key not in inits or set(by_method) != set(ARM_METHODS):
            continue
        problem = LayoutProblem(cells[(key[0], key[1], key[2])]["inst"].take([key[3]]))
        feats = instance_features(problem, inits[key]["initial_J"])
        curves = {m: {b: float(np.mean([r[b] for r in rows])) for b in rows[0]} for m, rows in by_method.items()}
        records.append({"key": key, "features": feats, "curves": curves})
    return records


def arm_targets(curves, budget):
    """Normalized early reward per arm: improvement at the budget over the best arm."""
    values = np.array([max(curves[m][budget], 0.0) for m in ARM_METHODS])
    top = values.max()
    return values / top if top > 1e-9 else np.ones(len(values))


def fit_arm_priors(directories, out, budget=3.0):
    records = []
    for directory in directories:
        records.extend(_load_arm_runs(directory))
    if len(records) < 8:
        raise RuntimeError("too few training instances: {}".format(len(records)))
    X = np.array([feature_vector(r["features"]) for r in records])
    Y = np.array([arm_targets(r["curves"], budget) for r in records])
    from sklearn.ensemble import GradientBoostingRegressor
    models = []
    for k in range(Y.shape[1]):
        model = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, random_state=k)
        models.append(model.fit(X, Y[:, k]))
    bundle = {"kind": "arm_priors", "models": models, "arms": list(ARM_METHODS), "features": FEATURE_NAMES,
              "budget_s": budget, "training_instances": len(records),
              "training_dirs": [str(Path(d).resolve()) for d in directories],
              "mean_targets": Y.mean(axis=0).tolist()}
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as handle:
        pickle.dump(bundle, handle)
    print("saved", out, "instances", len(records), "mean targets", dict(zip(ARM_METHODS, np.round(Y.mean(axis=0), 3))))
    return bundle


def predict_arm_priors(bundle, features):
    """Predicted normalized early reward per arm, clipped to [0, 1]."""
    x = feature_vector(features)[None, :]
    pred = np.array([float(m.predict(x)[0]) for m in bundle["models"]])
    return np.clip(pred, 0.0, 1.0)


def evaluate_arm_priors(directories, budget=3.0):
    """Leave-one-size-out accuracy of the predicted best arm and its realized value."""
    records = []
    for directory in directories:
        records.extend(_load_arm_runs(directory))
    X = np.array([feature_vector(r["features"]) for r in records])
    Y = np.array([arm_targets(r["curves"], budget) for r in records])
    sizes = np.array([r["key"][1] for r in records])
    from sklearn.ensemble import GradientBoostingRegressor
    pred = np.zeros_like(Y)
    for held in sorted(set(sizes)):
        train, test = sizes != held, sizes == held
        for k in range(Y.shape[1]):
            model = GradientBoostingRegressor(n_estimators=150, max_depth=2, learning_rate=0.05,
                                              subsample=0.8, random_state=k).fit(X[train], Y[train, k])
            pred[test, k] = model.predict(X[test])
    arms = list(ARM_METHODS)
    report = {"budget_s": budget, "instances": len(records), "per_size": {}}
    for size in sorted(set(sizes)):
        mask = sizes == size
        best_true = Y[mask].argmax(axis=1)
        best_pred = pred[mask].argmax(axis=1)
        realized = np.mean([Y[i, best_pred[j]] for j, i in enumerate(np.where(mask)[0])])
        report["per_size"][int(size)] = {"top1_accuracy": float(np.mean(best_true == best_pred)),
                                         "realized_normalized_reward": float(realized),
                                         "always_M0": float(Y[mask, arms.index("M0")].mean()),
                                         "always_M1": float(Y[mask, arms.index("M1")].mean()),
                                         "predicted_best_arm_counts": {a: int(np.sum(best_pred == k)) for k, a in enumerate(arms)}}
    print(json.dumps(report, indent=1))
    return report


def evaluate(directories, objective="auc"):
    """Cross-validated value of the predicted tau versus fixed taus and the oracle."""
    X, y, groups, meta = build_dataset(directories, objective)
    sizes = np.array([g[1] for g in groups])
    cells = np.array(["{}_{}".format(g[1], g[2]) for g in groups])
    taus = sorted({t for scores in meta for t in scores})

    def realized(pred_tau, scores):
        # value of running the tau on the grid nearest to the prediction (log scale)
        nearest = min(scores, key=lambda t: abs(np.log(t) - np.log(pred_tau)))
        return scores[nearest]

    report = {}
    for scheme, labels in (("leave_one_size_out", sizes), ("leave_one_cell_out", cells)):
        pred = np.zeros(len(y))
        for held in sorted(set(labels)):
            train = labels != held
            test = labels == held
            model = make_model().fit(X[train], y[train])
            pred[test] = np.exp(model.predict(X[test]))
        value_pred = np.mean([realized(p, s) for p, s in zip(pred, meta)])
        value_oracle = np.mean([max(s.values()) for s in meta])
        value_fixed = {str(t): float(np.mean([s[t] for s in meta])) for t in taus}
        per_size = {}
        for size in sorted(set(sizes)):
            mask = sizes == size
            per_size[int(size)] = {"predicted": float(np.mean([realized(p, s) for p, s, m in zip(pred, meta, mask) if m])),
                                   "oracle": float(np.mean([max(s.values()) for s, m in zip(meta, mask) if m])),
                                   "fixed": {str(t): float(np.mean([s[t] for s, m in zip(meta, mask) if m])) for t in taus},
                                   "median_predicted_tau": float(np.median(pred[mask]))}
        report[scheme] = {"predicted": float(value_pred), "oracle": float(value_oracle),
                          "fixed": value_fixed, "per_size": per_size}
    print(json.dumps(report, indent=1))
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["fit", "evaluate", "fit-priors", "evaluate-priors"])
    parser.add_argument("directories", nargs="+")
    parser.add_argument("--out", default=str(MODELS / "swl_v1.pkl"))
    parser.add_argument("--objective", choices=["auc", "final"], default="auc")
    parser.add_argument("--budget", type=float, default=3.0, help="early window for arm priors")
    args = parser.parse_args()
    if args.command == "fit":
        fit(args.directories, args.out, args.objective)
    elif args.command == "evaluate":
        evaluate(args.directories, args.objective)
    elif args.command == "fit-priors":
        fit_arm_priors(args.directories, args.out, args.budget)
    else:
        evaluate_arm_priors(args.directories, args.budget)


if __name__ == "__main__":
    main()
