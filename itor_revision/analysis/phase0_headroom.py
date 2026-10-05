"""Phase 0 (zero compute cost): what do the existing 2,124 runs say about switching?

1. Oracle headroom: per instance, the better of M0 and M1 (seed-averaged) at each
   budget; how far a per-instance choice could move each cell above the best
   single method.
2. Predictability: can the early M1 trace (first 1 or 3 seconds) plus instance
   features predict whether M1 or M0 wins at 60 seconds?  Evaluated with random
   stratified folds and with leave-one-(n, fill)-cell-out folds (transfer).

Reads the audited test directory of the published study; never writes there.
"""
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
TEST = ROOT / "code_submission" / "results" / "completion_extension_full_cpu" / "test"
OUT = Path(__file__).resolve().parent / "phase0"
OUT.mkdir(exist_ok=True)
CACHE = OUT / "runs_compact.pkl"
BUDGETS = [0.0, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 60.0]


def pin_to_efficiency_cores():
    try:
        import psutil
        allowed = psutil.Process().cpu_affinity()
        e_cores = [c for c in allowed if c >= 16]
        if e_cores:
            psutil.Process().cpu_affinity(e_cores)
    except Exception:
        pass


def load_runs():
    if CACHE.exists():
        return pd.read_pickle(CACHE)
    records = []
    files = sorted((TEST / "runs").glob("*.json"))
    started = time.perf_counter()
    for k, path in enumerate(files):
        run = json.loads(path.read_text(encoding="utf-8"))
        row = {key: run[key] for key in ("method", "geometry", "n", "fill", "instance",
                                         "evaluation_seed", "initial_J", "initialization_s")}
        for r in run["rows"]:
            row["imp_{:g}".format(r["budget_s"])] = r["improvement_pct"]
        row["trace"] = [(e["elapsed_s"], e["J"], e["branch"]) for e in run["trace"]]
        st = run["stats"]
        for key in ("repair_attempts", "repair_successes", "unchanged_reuses", "reconstructions",
                    "full_completion_calls", "proposals_D0", "proposals_D1_to_cap",
                    "proposals_D_above_cap", "fallback_reuses", "mean_displaced", "iterations"):
            row[key] = st.get(key)
        row["incumbent_improvements_by_branch"] = st.get("incumbent_improvements_by_branch", {})
        row["incumbent_gain_by_branch"] = st.get("incumbent_gain_by_branch", {})
        records.append(row)
        if (k + 1) % 200 == 0:
            print("loaded {}/{} ({:.0f} s)".format(k + 1, len(files), time.perf_counter() - started), flush=True)
    frame = pd.DataFrame.from_records(records)
    frame.to_pickle(CACHE)
    return frame


def instance_features():
    rows = []
    for path in sorted((TEST / "initializations").glob("*.json")):
        init = json.loads(path.read_text(encoding="utf-8"))
        if not init["initialized"]:
            continue
        rows.append({"geometry": init["geometry"], "n": init["n"], "fill": init["fill"],
                     "instance": init["instance"], "gw": init["gw"], "gh": init["gh"],
                     "achieved_fill": init["achieved_fill"],
                     "mean_facility_area": init["mean_facility_area"],
                     "initial_J": init["initial_J"]})
    return pd.DataFrame(rows)


def early_trace_features(trace, t0):
    """Features computable by a policy that has run M1 for t0 seconds."""
    events = [e for e in trace if 0 < e[0] <= t0]
    initial = trace[0][1]
    best = initial
    improvements = 0
    by_branch = defaultdict(int)
    for elapsed, value, branch in events:
        by_branch[branch] += 1
        if value > best + 1e-12:
            best = value
            improvements += 1
    imp_t0 = 100 * (best - initial) / initial
    best_third = initial
    for elapsed, value, _ in events:
        if elapsed <= t0 / 3.0:
            best_third = max(best_third, value)
    imp_third = 100 * (best_third - initial) / initial
    total = max(len(events), 1)
    return {"imp_t0": imp_t0, "slope": (imp_t0 - imp_third) / (t0 - t0 / 3.0),
            "events": len(events), "improvements": improvements,
            "improve_rate": improvements / total,
            "frac_repair": by_branch.get("repair", 0) / total,
            "frac_reuse": by_branch.get("reuse", 0) / total,
            "frac_fallback": by_branch.get("fallback", 0) / total,
            "last_gain_s": max([e[0] for e in events if e[1] >= best - 1e-12] or [0.0])}


def seed_average(frame, columns):
    keys = ["geometry", "n", "fill", "instance", "method"]
    return frame.groupby(keys)[columns].mean().reset_index()


def gmean(series):
    """Equal weight for the two geometry families."""
    labels = series.index.get_level_values("geometry")
    return float(np.mean([series[labels == g].mean() for g in ("guillotine", "nonslicing")]))


def headroom(frame):
    imp_cols = ["imp_{:g}".format(b) for b in BUDGETS]
    avg = seed_average(frame, imp_cols)
    wide = avg.pivot_table(index=["geometry", "n", "fill", "instance"], columns="method", values=imp_cols)
    out_rows = []
    for (n, fill), sub in wide.groupby(level=["n", "fill"]):
        for b in BUDGETS:
            col = "imp_{:g}".format(b)
            m0 = sub[(col, "M0")]
            m1 = sub[(col, "M1")]
            al = sub[(col, "ALNS:small_beam")]
            oracle2 = np.maximum(m0, m1)
            oracle3 = np.maximum(oracle2, al)
            best_single = max(gmean(m0), gmean(m1))
            out_rows.append({"n": n, "fill": fill, "budget_s": b, "instances": len(sub),
                             "M0": gmean(m0), "M1": gmean(m1), "ALNS": gmean(al),
                             "oracle_M0_M1": gmean(oracle2), "oracle_all3": gmean(oracle3),
                             "headroom_pp": gmean(oracle2) - best_single,
                             "regret_always_M0": gmean(oracle2) - gmean(m0),
                             "regret_always_M1": gmean(oracle2) - gmean(m1),
                             "M1_win_fraction": float(np.mean(m1 > m0 + 1e-9))})
    return pd.DataFrame(out_rows), wide


def predictability(frame, wide, feats):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    m1 = frame[frame.method == "M1"]
    results = {}
    for t0 in (1.0, 3.0):
        rows = []
        for (geometry, n, fill, instance), sub in m1.groupby(["geometry", "n", "fill", "instance"]):
            acc = defaultdict(list)
            for trace in sub.trace:
                for key, value in early_trace_features(trace, t0).items():
                    acc[key].append(value)
            row = {k: float(np.mean(v)) for k, v in acc.items()}
            row.update({"geometry": geometry, "n": n, "fill": fill, "instance": instance})
            rows.append(row)
        data = pd.DataFrame(rows).merge(feats, on=["geometry", "n", "fill", "instance"])
        idx = pd.MultiIndex.from_frame(data[["geometry", "n", "fill", "instance"]])
        d60 = (wide[("imp_60", "M1")] - wide[("imp_60", "M0")]).reindex(idx).values
        d10 = (wide[("imp_10", "M1")] - wide[("imp_10", "M0")]).reindex(idx).values
        m0_60 = wide[("imp_60", "M0")].reindex(idx).values
        m1_60 = wide[("imp_60", "M1")].reindex(idx).values
        data["is_nonslicing"] = (data.geometry == "nonslicing").astype(float)
        data["log_n"] = np.log2(data.n)
        data["plate"] = data.gw * data.gh
        data["initial_J_per_n"] = data.initial_J / data.n
        early_cols = ["imp_t0", "slope", "events", "improvements", "improve_rate",
                      "frac_repair", "frac_reuse", "frac_fallback", "last_gain_s"]
        inst_cols = ["log_n", "fill", "is_nonslicing", "plate", "mean_facility_area",
                     "achieved_fill", "initial_J_per_n"]
        feature_sets = {"instance_only": inst_cols, "early_M1_only": early_cols,
                        "instance_plus_early": inst_cols + early_cols}
        y = (d60 > 0).astype(int)
        cell_labels = np.array(["{}_{}".format(a, b) for a, b in zip(data.n, data.fill)])
        out = {"t0": t0, "instances": int(len(y)), "positive_rate": float(y.mean()), "schemes": {}}
        models = {
            "logistic": lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)),
            "hist_gb": lambda: HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05,
                                                                max_iter=200, min_samples_leaf=8),
        }
        for fs_name, cols in feature_sets.items():
            X = data[cols].values.astype(float)
            for model_name, factory in models.items():
                for scheme in ("stratified5", "leave_one_cell_out", "leave_one_size_out"):
                    pred = np.full(len(y), np.nan)
                    prob = np.full(len(y), np.nan)
                    if scheme == "stratified5":
                        splitter = StratifiedKFold(5, shuffle=True, random_state=0).split(X, y)
                    elif scheme == "leave_one_cell_out":
                        splitter = ((np.where(cell_labels != c)[0], np.where(cell_labels == c)[0])
                                    for c in sorted(set(cell_labels)))
                    else:
                        sizes = data.n.values
                        splitter = ((np.where(sizes != s)[0], np.where(sizes == s)[0])
                                    for s in sorted(set(sizes)))
                    for train, test in splitter:
                        if len(set(y[train])) < 2:
                            pred[test] = int(round(y[train].mean()))
                            prob[test] = y[train].mean()
                            continue
                        model = factory()
                        model.fit(X[train], y[train])
                        prob[test] = model.predict_proba(X[test])[:, 1]
                        pred[test] = (prob[test] > 0.5).astype(int)
                    chosen = np.where(pred == 1, m1_60, m0_60)
                    oracle = np.maximum(m0_60, m1_60)
                    per_cell = {}
                    for c in sorted(set(cell_labels)):
                        mask = cell_labels == c
                        per_cell[c] = {"policy": float(chosen[mask].mean()), "M0": float(m0_60[mask].mean()),
                                       "M1": float(m1_60[mask].mean()), "oracle": float(oracle[mask].mean()),
                                       "accuracy": float((pred[mask] == y[mask]).mean())}
                    try:
                        auc = float(roc_auc_score(y, prob))
                    except ValueError:
                        auc = None
                    out["schemes"]["{}|{}|{}".format(fs_name, model_name, scheme)] = {
                        "accuracy": float((pred == y).mean()), "auc": auc,
                        "policy_mean_60s": float(chosen.mean()), "M0_mean_60s": float(m0_60.mean()),
                        "M1_mean_60s": float(m1_60.mean()), "oracle_mean_60s": float(oracle.mean()),
                        "per_cell": per_cell}
        table = {}
        for c in sorted(set(cell_labels)):
            mask = cell_labels == c
            table[c] = int(y[mask].mean() > 0.5)
        pred = np.array([table[c] for c in cell_labels])
        chosen = np.where(pred == 1, m1_60, m0_60)
        out["cell_lookup_in_sample"] = {"accuracy": float((pred == y).mean()),
                                        "policy_mean_60s": float(chosen.mean())}
        out["univariate_auc"] = {}
        for col in inst_cols + early_cols:
            try:
                out["univariate_auc"][col] = float(roc_auc_score(y, data[col].values))
            except ValueError:
                pass
        out["label_summary"] = {"d60_mean": float(np.mean(d60)), "d10_mean": float(np.mean(d10)),
                                "corr_d10_d60": float(np.corrcoef(d10, d60)[0, 1])}
        results["t0={:g}".format(t0)] = out
        data.assign(d60=d60, d10=d10, y=y).to_csv(OUT / "features_t0_{:g}.csv".format(t0), index=False)
    return results


def write_report(head, pred):
    lines = ["# Phase 0: oracle headroom and predictability from the published 60-second study", "",
             "Source: `code_submission/results/completion_extension_full_cpu/test` (2,124 runs, audited).",
             "Seeds averaged within instance; geometries equally weighted.", "",
             "## 1. Oracle headroom (percentage points of initial-objective improvement)", "",
             "oracle_M0_M1 = per-instance better of M0 and M1. headroom = oracle minus the better *cell-level* single method.",
             "regret_always_X = oracle minus X. A switching policy can at most recover the headroom.", ""]
    for b in (10.0, 60.0):
        sub = head[head.budget_s == b]
        lines += ["### {:g} seconds".format(b), "",
                  "| n | fill | M0 | M1 | ALNS | oracle(M0,M1) | oracle(all 3) | headroom | regret M0 | regret M1 | M1 win frac |",
                  "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in sub.iterrows():
            lines.append("| {n} | {fill:.2f} | {M0:.2f} | {M1:.2f} | {ALNS:.2f} | {oracle_M0_M1:.2f} | {oracle_all3:.2f} | "
                         "**{headroom_pp:.2f}** | {regret_always_M0:.2f} | {regret_always_M1:.2f} | {M1_win_fraction:.2f} |".format(**r))
        lines.append("")
    lines += ["## 2. Can the first seconds of M1 predict the 60-second winner?", "",
              "Label: M1 beats M0 at 60 s (seed-averaged). Policy value = mean 60 s improvement when the",
              "predicted winner is run alone; compare with M0-only, M1-only and the oracle.", ""]
    for key, out in pred.items():
        lines += ["### Early window {} (positive rate {:.2f}, {} instances)".format(key, out["positive_rate"], out["instances"]), "",
                  "| features | model | CV scheme | accuracy | AUC | policy | M0 | M1 | oracle |",
                  "|---|---|---|---:|---:|---:|---:|---:|---:|"]
        for name, s in out["schemes"].items():
            fs, model, scheme = name.split("|")
            lines.append("| {} | {} | {} | {:.3f} | {} | {:.2f} | {:.2f} | {:.2f} | {:.2f} |".format(
                fs, model, scheme, s["accuracy"], "{:.3f}".format(s["auc"]) if s["auc"] is not None else "n/a",
                s["policy_mean_60s"], s["M0_mean_60s"], s["M1_mean_60s"], s["oracle_mean_60s"]))
        lines += ["", "In-sample (n, fill) lookup table: accuracy {:.3f}, policy {:.2f}.".format(
            out["cell_lookup_in_sample"]["accuracy"], out["cell_lookup_in_sample"]["policy_mean_60s"]), "",
            "Univariate AUC (>0.5 favors M1 when large): " + ", ".join(
                "{} {:.2f}".format(k, v) for k, v in sorted(out["univariate_auc"].items(), key=lambda kv: -abs(kv[1] - 0.5))), "",
            "corr(M1-M0 at 10 s, M1-M0 at 60 s) = {:.2f}".format(out["label_summary"]["corr_d10_d60"]), ""]
        best = max(out["schemes"].values(), key=lambda s: s["policy_mean_60s"])
        lines += ["Per-cell values for the best scheme by policy value:", "",
                  "| cell | policy | M0 | M1 | oracle | accuracy |", "|---|---:|---:|---:|---:|---:|"]
        for c, v in best["per_cell"].items():
            lines.append("| {} | {policy:.2f} | {M0:.2f} | {M1:.2f} | {oracle:.2f} | {accuracy:.2f} |".format(c, **v))
        lines.append("")
    (OUT / "phase0_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    pin_to_efficiency_cores()
    frame = load_runs()
    print("runs:", len(frame), "methods:", sorted(frame.method.unique()), flush=True)
    feats = instance_features()
    head, wide = headroom(frame)
    head.to_csv(OUT / "headroom.csv", index=False)
    pred = predictability(frame, wide, feats)
    (OUT / "predictability.json").write_text(json.dumps(pred, indent=1), encoding="utf-8")
    write_report(head, pred)
    print("PHASE 0 DONE ->", OUT / "phase0_report.md", flush=True)


if __name__ == "__main__":
    main()
