"""
The action-space axis (discrete vs. continuous) re-run *inside the corrected
improvement MDP*, at 32 rooms.

The predecessor study's contribution was a controlled comparison of a discrete
micro-move head against a continuous displacement head, on eight-room scenarios,
inside the legacy reward-penalty MDP.  That axis is worth keeping, but the interesting
question once the formulation is repaired is whether it still discriminates.
So both parameterizations are trained here on the identical corrected
configuration -- best-so-far reward, jump and swap moves, Metropolis ratchet --
at the same budget as every other row of the ablation.

  C1        lattice     displacements rounded to integer cells (discrete)
  C1cont    continuous  real-valued displacements, plus a snap-to-lattice op
"""

import sys

from train_improve import get_parser, train

RUNS = [
    dict(tag="C1c_neural_sa_continuous", reward="best",
         ops="translate,jump,swap,snap", gamma=1.0, lam=0.95,
         accept="metropolis", action_mode="continuous"),
]


def main():
    for cfg in RUNS:
        args = get_parser().parse_args(sys.argv[1:])
        for k, v in cfg.items():
            setattr(args, k, v)
        print(f"\n{'='*78}\n{args.tag}\n{'='*78}", flush=True)
        train(args)


if __name__ == "__main__":
    main()
