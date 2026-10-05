# Phase 0b: M0 versus M1 improvement-rate crossover (published traces)

level crossover = first grid time at which the mean M0 curve stays above the mean M1 curve;
rate crossover = first time M0 improves faster than M1 for three consecutive grid points;
per-instance crossover = time after which M0 stays above M1 for that instance (inf if never).

| n | fill | level cross (s) | rate cross (s) | M0 never overtakes | inst. cross median [q25, q75] (s) | M0@60 | M1@60 |
|---:|---:|---:|---:|---:|---|---:|---:|
| 32 | 0.90 | 0.2 | 0.2 | 0.70 | 0.2 [0.2, 0.3] | 24.43 | 24.32 |
| 32 | 0.95 | 0.2 | 0.2 | 0.69 | 0.1 [0.1, 17.0] | 17.58 | 18.81 |
| 64 | 0.90 | 0.7 | 0.5 | 0.40 | 0.7 [0.5, 1.0] | 18.27 | 17.49 |
| 64 | 0.95 | 1.5 | 0.5 | 0.65 | 0.6 [0.1, 1.5] | 14.86 | 15.38 |
| 128 | 0.90 | 3.5 | 1.5 | 0.00 | 3.8 [2.5, 6.5] | 18.83 | 16.23 |
| 128 | 0.95 | 0.1 | 2.0 | 0.33 | 5.0 [2.2, 12.0] | 14.33 | 11.97 |
| 256 | 0.90 | 0.1 | 6.5 | 0.00 | 16.0 [12.0, 21.5] | 19.39 | 14.70 |
| 256 | 0.95 | 0.1 | 0.1 | 0.33 | 14.0 [11.0, 20.0] | 10.38 | 8.58 |
