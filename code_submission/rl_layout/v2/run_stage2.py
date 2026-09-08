"""
Second batch of training runs: the instance-distribution policy (Track D) and a
second fixed scenario (the clinic), run back to back.
"""

import sys

import train_construct as TC

RUNS = [
    # Track B on the second legacy scenario: 68% fill, where the earlier study reported SA
    # tying a no-search heuristic and reaching zero overlap in only 20% of runs.
    # Cheap (small floor plate), so it runs first.
    dict(tag="B2_constr_hospital", instances="fixed", scenario="hospital",
         total_steps=900_000),
    # Track D: never sees comb_high or hospital during training
    dict(tag="D1_constr_random", instances="random", n_rooms=32, canvas=44,
         total_steps=1_200_000, ch=48, minibatch=256, epochs=2),
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
