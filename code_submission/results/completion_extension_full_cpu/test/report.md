# Extended comparison: test

Complete phase; independent saved-layout audit passed.

Attempted: 240; initialized: 236; runs: 2124; verified saved layouts: 17228.

Seeds are averaged within instance. Intervals are descriptive, geometry-stratified
paired bootstrap intervals, not multiplicity-adjusted significance tests.

| Geometry | n | Nominal fill | Initialized / attempted | Achieved fill range |
|---|---:|---:|---:|---:|
| guillotine | 32 | 0.90 | 10/10 | 0.89815 to 0.90000 |
| guillotine | 32 | 0.95 | 18/20 | 0.94318 to 0.95000 |
| guillotine | 64 | 0.90 | 10/10 | 0.89855 to 0.89959 |
| guillotine | 64 | 0.95 | 20/20 | 0.94792 to 0.95000 |
| guillotine | 128 | 0.90 | 10/10 | 0.89938 to 0.90000 |
| guillotine | 128 | 0.95 | 20/20 | 0.94885 to 0.95000 |
| guillotine | 256 | 0.90 | 10/10 | 0.89965 to 0.90000 |
| guillotine | 256 | 0.95 | 20/20 | 0.94943 to 0.95000 |
| nonslicing | 32 | 0.90 | 10/10 | 0.89583 to 0.90000 |
| nonslicing | 32 | 0.95 | 18/20 | 0.94603 to 0.95000 |
| nonslicing | 64 | 0.90 | 10/10 | 0.89836 to 0.90000 |
| nonslicing | 64 | 0.95 | 20/20 | 0.94796 to 0.94994 |
| nonslicing | 128 | 0.90 | 10/10 | 0.89890 to 0.90000 |
| nonslicing | 128 | 0.95 | 20/20 | 0.94915 to 0.95000 |
| nonslicing | 256 | 0.90 | 10/10 | 0.89942 to 0.90000 |
| nonslicing | 256 | 0.95 | 20/20 | 0.94933 to 0.95000 |

## Primary contrasts

| n | Fill | Seconds | Control | Instances | M1 minus control (pp) | 95% interval |
|---:|---:|---:|---|---:|---:|---:|
| 32 | 0.90 | 10 | M0 | 20 | -1.60 | [-3.05, -0.28] |
| 32 | 0.90 | 60 | M0 | 20 | -0.10 | [-1.22, +0.95] |
| 32 | 0.90 | 10 | ALNS:small_beam | 20 | +1.01 | [-0.57, +2.52] |
| 32 | 0.90 | 60 | ALNS:small_beam | 20 | +0.66 | [-0.64, +2.03] |
| 32 | 0.95 | 10 | M0 | 36 | +1.35 | [+0.34, +2.28] |
| 32 | 0.95 | 60 | M0 | 36 | +1.23 | [+0.04, +2.31] |
| 32 | 0.95 | 10 | ALNS:small_beam | 36 | +3.45 | [+2.47, +4.44] |
| 32 | 0.95 | 60 | ALNS:small_beam | 36 | +2.97 | [+2.09, +3.87] |
| 64 | 0.90 | 10 | M0 | 20 | -1.98 | [-3.21, -0.78] |
| 64 | 0.90 | 60 | M0 | 20 | -0.79 | [-1.78, +0.18] |
| 64 | 0.90 | 10 | ALNS:small_beam | 20 | +1.36 | [+0.35, +2.41] |
| 64 | 0.90 | 60 | ALNS:small_beam | 20 | -1.74 | [-2.76, -0.79] |
| 64 | 0.95 | 10 | M0 | 40 | -1.01 | [-2.24, +0.19] |
| 64 | 0.95 | 60 | M0 | 40 | +0.52 | [-0.57, +1.52] |
| 64 | 0.95 | 10 | ALNS:small_beam | 40 | +4.16 | [+3.39, +5.01] |
| 64 | 0.95 | 60 | ALNS:small_beam | 40 | +1.68 | [+0.98, +2.44] |
| 128 | 0.90 | 10 | M0 | 20 | -2.22 | [-3.84, -1.08] |
| 128 | 0.90 | 60 | M0 | 20 | -2.59 | [-3.74, -1.68] |
| 128 | 0.90 | 10 | ALNS:small_beam | 20 | +5.04 | [+3.61, +6.43] |
| 128 | 0.90 | 60 | ALNS:small_beam | 20 | -0.41 | [-1.66, +0.88] |
| 128 | 0.95 | 10 | M0 | 40 | -0.98 | [-2.31, +0.37] |
| 128 | 0.95 | 60 | M0 | 40 | -2.35 | [-3.51, -1.20] |
| 128 | 0.95 | 10 | ALNS:small_beam | 40 | +5.10 | [+4.19, +6.04] |
| 128 | 0.95 | 60 | ALNS:small_beam | 40 | +3.08 | [+2.33, +3.90] |
| 256 | 0.90 | 10 | M0 | 20 | +3.98 | [+2.79, +5.25] |
| 256 | 0.90 | 60 | M0 | 20 | -4.70 | [-6.37, -3.16] |
| 256 | 0.90 | 10 | ALNS:small_beam | 20 | +10.00 | [+8.98, +11.08] |
| 256 | 0.90 | 60 | ALNS:small_beam | 20 | +3.93 | [+2.69, +5.18] |
| 256 | 0.95 | 10 | M0 | 40 | +2.90 | [+2.08, +3.77] |
| 256 | 0.95 | 60 | M0 | 40 | -1.80 | [-2.89, -0.74] |
| 256 | 0.95 | 10 | ALNS:small_beam | 40 | +5.91 | [+5.35, +6.46] |
| 256 | 0.95 | 60 | ALNS:small_beam | 40 | +4.12 | [+3.71, +4.54] |

Full quality, target-time and paired-contrast records are in the adjacent CSV files.
Non-hits contribute the cutoff to restricted mean search time, rather than being dropped.
Initialization is included in the companion total-time column.
