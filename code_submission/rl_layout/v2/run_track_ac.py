"""
Tracks A and C plus the v1 control, run back to back through one trainer.

The point of the ordering is that each row changes exactly one thing relative to
the row above it, so the resulting table is an ablation rather than a collection
of separately tuned systems.

  A0  delta reward, translate only, gamma=0.995   <- the legacy formulation
  A0g delta reward, translate only, gamma=0.0     <- v1's own best discount
  A1  best-so-far reward, translate only, gamma=1 <- reward fix alone
  A2  best-so-far reward, + jump + swap           <- move-set fix on top
  C1  A2 + Metropolis accept/reject               <- learned proposal for annealing
"""

import sys

from train_improve import get_parser, train

RUNS = [
    dict(tag="A0_v1_delta_g995", reward="delta", ops="translate", gamma=0.995, lam=0.96),
    dict(tag="A0_v1_delta_g0", reward="delta", ops="translate", gamma=0.0, lam=0.0),
    dict(tag="A1_best_translate", reward="best", ops="translate", gamma=1.0, lam=0.95),
    dict(tag="A2_best_allops", reward="best", ops="translate,jump,swap", gamma=1.0, lam=0.95),
    dict(tag="C1_neural_sa", reward="best", ops="translate,jump,swap", gamma=1.0, lam=0.95,
         accept="metropolis"),
]


def main():
    base = get_parser().parse_args(sys.argv[1:])
    for cfg in RUNS:
        args = get_parser().parse_args(sys.argv[1:])
        for k, v in cfg.items():
            setattr(args, k, v)
        print(f"\n{'='*78}\n{args.tag}\n{'='*78}", flush=True)
        train(args)


if __name__ == "__main__":
    main()
