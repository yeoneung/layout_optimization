"""
Track E training: two mechanisms that turn the constructive policy from a
function into a distribution, so its quality/diversity front is traced by
retraining rather than only by raising the sampling temperature.

  B3  the commit order is drawn by the environment, not chosen by the policy.
      The placement rule stays sharp; variety comes from the order, exactly the
      way an annealer's variety comes from its seed.  This is the mechanism that
      should give variety at little cost in objective.
  B4  the policy still chooses the order, but the entropy bonus is held up
      instead of annealed to zero.  Variety then comes from a blunter policy and
      should cost more objective -- the contrast is the point.
"""

import sys

import train_construct as TC

RUNS = [
    dict(tag="B3_constr_randorder", order="random", ent_floor=0.05,
         total_steps=900_000),
    dict(tag="B4_constr_high_entropy", order="policy", ent_coef=0.04, ent_floor=1.0,
         total_steps=900_000),
]


def main():
    for cfg in RUNS:
        args = TC.get_parser().parse_args(sys.argv[1:])
        for k, v in cfg.items():
            setattr(args, k, v)
        print(f"\n{'='*78}\n{args.tag}\n{'='*78}", flush=True)
        TC.train(args)


if __name__ == "__main__":
    main()
