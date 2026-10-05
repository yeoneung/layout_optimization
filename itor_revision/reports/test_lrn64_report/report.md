# EAAI phase report: test

Runs present: 720 of 720 expected on 60 initialized instances (60 attempted so far); instances with all methods and seeds: 60.

Methods: M0, BANDIT:8, LRN:S64_seed1, LRN:E64_f9095_seed1

Seeds averaged within instance; geometries equally weighted; intervals are descriptive
geometry-stratified paired bootstrap intervals.

## Improvement over the common initial layout at 10 s (%)

| n | fill | inst | M0 | BANDIT:8 | LRN:S64_seed1 | LRN:E64_f9095_seed1 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.90 | 20 | **17.21** | 16.85 | 14.12 | 14.37 | n/a |
| 64 | 0.95 | 40 | 13.17 | **13.99** | 11.26 | 12.63 | n/a |

### Summary across cells at 10 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| M0 | - | 1.89 | 1.04 | 1 |
| BANDIT:8 | 1/1/0 | 1.77 | 0.80 | 1 |
| LRN:S64_seed1 | 0/0/2 | 3.56 | 3.54 | 0 |
| LRN:E64_f9095_seed1 | 0/1/1 | 2.78 | 2.73 | 0 |

### Method minus M0 at 10 s (pp, 95% interval)

| n | fill | BANDIT:8 | LRN:S64_seed1 | LRN:E64_f9095_seed1 |
|---:|---:|---:|---:|---:|
| 64 | 0.90 | -0.36 [-0.90,+0.19] | -3.09 [-3.85,-2.33] - | -2.84 [-3.82,-1.87] - |
| 64 | 0.95 | +0.83 [+0.30,+1.36] + | -1.91 [-2.80,-1.02] - | -0.53 [-1.69,+0.52] |

## Improvement over the common initial layout at 60 s (%)

| n | fill | inst | M0 | BANDIT:8 | LRN:S64_seed1 | LRN:E64_f9095_seed1 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | 0.90 | 20 | 18.57 | **19.71** | 15.70 | 16.05 | n/a |
| 64 | 0.95 | 40 | 15.14 | **17.86** | 13.42 | 14.91 | n/a |

### Summary across cells at 60 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| M0 | - | 2.36 | 1.99 | 0 |
| BANDIT:8 | 2/0/0 | 1.12 | 0.06 | 2 |
| LRN:S64_seed1 | 0/0/2 | 3.60 | 4.28 | 0 |
| LRN:E64_f9095_seed1 | 0/1/1 | 2.91 | 3.37 | 0 |

### Method minus M0 at 60 s (pp, 95% interval)

| n | fill | BANDIT:8 | LRN:S64_seed1 | LRN:E64_f9095_seed1 |
|---:|---:|---:|---:|---:|
| 64 | 0.90 | +1.14 [+0.78,+1.51] + | -2.87 [-3.66,-2.10] - | -2.52 [-3.53,-1.45] - |
| 64 | 0.95 | +2.71 [+2.14,+3.30] + | -1.73 [-2.46,-1.03] - | -0.23 [-1.21,+0.67] |

## Switching diagnostics

| n | fill | method | runs | first M0 start (s) | M0 time fraction | reconstructions |
|---:|---:|---|---:|---:|---:|---:|
| 64 | 0.90 | BANDIT:8 | 60 | 0.92 | 0.28 | 174.4 |
| 64 | 0.95 | BANDIT:8 | 120 | 0.99 | 0.40 | 137.3 |

CPU/wall ratio: mean 0.993, min 0.937.
