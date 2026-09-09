"""
Train and evaluate a learned predictor of a rollout certificate.

The target is the greedy-rollout verdict of `viability_data.py`: does the
constructor's own largest-first rollout complete the state?  (An earlier
version of the data script used a randomized-completion label; the shipped
labels are the rollout verdict.)  That verdict is
computable but costs one full rollout per query.  The
predictor sees only the fixed-length description of free space produced by
`viability.state_features`, which costs one pass of integral-image queries, and
must decide admissibility in one forward pass.

The operating point matters more than the accuracy.  Used as a filter, a false
positive admits a placement that strands a facility, and a false negative
discards a survivable one.  We therefore do not report a single threshold: we
sweep it and report, at each achieved feasibility level, how large an admission
rate the predictor sustains against what the first-fit certificate sustains.
That is the comparison the paper makes -- tightness at fixed feasibility, and
cost.
"""

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from viability import N_FEATURES


class ViabilityNet(nn.Module):
    def __init__(self, d=N_FEATURES, h=64):
        super().__init__()
        self.f = nn.Sequential(
            nn.Linear(d, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, 1))

    def forward(self, x):
        return self.f(x).squeeze(-1)


def load(path):
    d = np.load(path)
    if "group" not in d:
        raise ValueError(
            "dataset has no instance-group identifiers; regenerate it with "
            "the current viability_data.py")
    return d["X"], d["Y"], d["FF"], d["fill"], d["group"]


def grouped_split(groups, fills, val_frac=0.2, seed=0):
    """Return an instance-disjoint split, stratified by density cell."""
    rng = np.random.default_rng(seed)
    val_groups = []
    for fl in np.unique(fills):
        cell_groups = np.unique(groups[fills == fl])
        cell_groups = rng.permutation(cell_groups)
        n_val = max(1, int(round(val_frac * len(cell_groups))))
        val_groups.extend(cell_groups[:n_val].tolist())
    is_val = np.isin(groups, np.asarray(val_groups))
    vi = np.flatnonzero(is_val)
    ti = np.flatnonzero(~is_val)
    if not len(vi) or not len(ti):
        raise ValueError("grouped split produced an empty train or validation set")
    return ti, vi, np.asarray(val_groups, dtype=np.int64)


def train(X, Y, groups, fills, epochs=200, lr=1e-3, val_frac=0.2,
          seed=0, device="cpu"):
    torch.manual_seed(seed)
    ti, vi, val_groups = grouped_split(groups, fills, val_frac, seed)
    mu, sd = X[ti].mean(0), X[ti].std(0) + 1e-6

    def prep(a):
        return torch.tensor((a - mu) / sd, dtype=torch.float32, device=device)

    xt, yt = prep(X[ti]), torch.tensor(Y[ti], device=device)
    xv, yv = prep(X[vi]), torch.tensor(Y[vi], device=device)

    net = ViabilityNet().to(device)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    # positives are rare at the densities where the question is live
    pw = torch.tensor(max((yt == 0).sum().item(), 1) / max((yt == 1).sum().item(), 1),
                      device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    best, best_state = np.inf, None
    for e in range(epochs):
        net.train()
        opt.zero_grad()
        loss = lossf(net(xt), yt)
        loss.backward()
        opt.step()
        net.eval()
        with torch.no_grad():
            vl = lossf(net(xv), yv).item()
        if vl < best:
            best = vl
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    return net, mu, sd, (xv, yv), ti, vi, val_groups


def auc(scores, labels):
    o = np.argsort(scores)
    r = np.empty(len(scores))
    r[o] = np.arange(1, len(scores) + 1)
    p, n = labels.sum(), (1 - labels).sum()
    if p == 0 or n == 0:
        return float("nan")
    return float((r[labels == 1].sum() - p * (p + 1) / 2) / (p * n))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="viability_train.npz")
    p.add_argument("--out", default="runs/viability_net.pt")
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val-frac", dest="val_frac", type=float, default=0.2)
    a = p.parse_args()

    X, Y, FF, fill, groups = load(a.data)
    print(f"{len(X)} samples, oracle positive {Y.mean():.3f}, "
          f"first-fit positive {FF.mean():.3f}")

    t0 = time.time()
    net, mu, sd, (xv, yv), ti, vi, val_groups = train(
        X, Y, groups, fill, epochs=a.epochs, val_frac=a.val_frac, seed=a.seed)
    elapsed = time.time() - t0
    with torch.no_grad():
        s = net(xv).numpy()
    yv = yv.numpy()
    ffv = FF[vi]                       # first-fit verdicts on the same states
    print(f"validation AUC {auc(s, yv):.4f}")
    print(f"instance-disjoint split: {len(np.unique(groups[ti]))} train and "
          f"{len(val_groups)} validation instances; {len(ti)} train and "
          f"{len(vi)} validation states")

    ff_adm = float(ffv.mean())
    ff_rec = float(((ffv == 1) & (yv == 1)).sum() / max((yv == 1).sum(), 1))
    ff_pre = float(((ffv == 1) & (yv == 1)).sum() / max((ffv == 1).sum(), 1))
    print(f"\nfirst-fit certificate: admits {ff_adm:.3f}, recall {ff_rec:.3f}, "
          f"precision {ff_pre:.3f}")
    print("\nlearned predictive filter, threshold sweep")
    print(f"{'admit rate':>11s} {'recall':>8s} {'precision':>10s}")
    quantiles = (0.99, 0.95, 0.925, 0.9, 0.85, 0.8, 0.7)
    operating_points = []
    for q in quantiles:
        thr = float(np.quantile(s, q))
        adm = s >= thr
        rec = float((adm & (yv == 1)).sum() / max((yv == 1).sum(), 1))
        pre = float((adm & (yv == 1)).sum() / max(adm.sum(), 1))
        operating_points.append({"quantile": float(q), "threshold": thr,
                                 "admission": float(adm.mean()),
                                 "recall": rec, "precision": pre})
        mark = "  <- matches first-fit precision" if abs(pre - ff_pre) < 0.02 else ""
        print(f"{adm.mean():11.3f} {rec:8.3f} {pre:10.3f}{mark}")

    metadata = {
        "target": "largest-first greedy-rollout verdict",
        "split": "instance-disjoint, stratified by fill",
        "seed": int(a.seed), "val_fraction": float(a.val_frac),
        "epochs": int(a.epochs), "elapsed_seconds": float(elapsed),
        "n_train_states": int(len(ti)), "n_validation_states": int(len(vi)),
        "n_train_instances": int(len(np.unique(groups[ti]))),
        "n_validation_instances": int(len(val_groups)),
        "validation_auc": auc(s, yv),
        "first_fit": {"admission": ff_adm, "recall": ff_rec,
                      "precision": ff_pre},
        "operating_points": operating_points,
        "thresholds": [op["threshold"] for op in operating_points],
    }
    torch.save({"state": net.state_dict(), "mu": mu, "sd": sd,
                "metadata": metadata}, a.out)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
