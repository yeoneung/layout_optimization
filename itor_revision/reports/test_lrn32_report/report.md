# EAAI phase report: test

Runs present: 672 of 672 expected on 56 initialized instances (60 attempted so far); instances with all methods and seeds: 56.

Methods: M0, BANDIT:8, LRN:S_dord_seed1, LRN:E32_f9095_seed1

Seeds averaged within instance; geometries equally weighted; intervals are descriptive
geometry-stratified paired bootstrap intervals.

## Improvement over the common initial layout at 10 s (%)

| n | fill | inst | M0 | BANDIT:8 | LRN:S_dord_seed1 | LRN:E32_f9095_seed1 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.90 | 20 | 22.49 | **24.43** | 19.37 | 16.46 | n/a |
| 32 | 0.95 | 36 | 14.91 | **17.63** | 14.32 | 12.23 | n/a |

### Summary across cells at 10 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| M0 | - | 2.28 | 2.36 | 0 |
| BANDIT:8 | 2/0/0 | 1.09 | 0.04 | 2 |
| LRN:S_dord_seed1 | 0/0/2 | 2.77 | 4.22 | 0 |
| LRN:E32_f9095_seed1 | 0/0/2 | 3.85 | 6.72 | 0 |

### Method minus M0 at 10 s (pp, 95% interval)

| n | fill | BANDIT:8 | LRN:S_dord_seed1 | LRN:E32_f9095_seed1 |
|---:|---:|---:|---:|---:|
| 32 | 0.90 | +1.94 [+1.36,+2.49] + | -3.12 [-4.51,-1.83] - | -6.04 [-7.74,-4.47] - |
| 32 | 0.95 | +2.71 [+2.15,+3.27] + | -0.59 [-1.19,-0.02] - | -2.68 [-3.36,-2.01] - |

## Improvement over the common initial layout at 60 s (%)

| n | fill | inst | M0 | BANDIT:8 | LRN:S_dord_seed1 | LRN:E32_f9095_seed1 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.90 | 20 | 24.85 | **27.15** | 21.59 | 19.13 | n/a |
| 32 | 0.95 | 36 | 18.32 | **20.97** | 16.58 | 14.44 | n/a |

### Summary across cells at 60 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| M0 | - | 2.08 | 2.56 | 0 |
| BANDIT:8 | 2/0/0 | 1.12 | 0.09 | 2 |
| LRN:S_dord_seed1 | 0/0/2 | 2.98 | 5.07 | 0 |
| LRN:E32_f9095_seed1 | 0/0/2 | 3.81 | 7.36 | 0 |

### Method minus M0 at 60 s (pp, 95% interval)

| n | fill | BANDIT:8 | LRN:S_dord_seed1 | LRN:E32_f9095_seed1 |
|---:|---:|---:|---:|---:|
| 32 | 0.90 | +2.30 [+1.86,+2.76] + | -3.27 [-4.37,-2.15] - | -5.72 [-7.05,-4.45] - |
| 32 | 0.95 | +2.65 [+1.84,+3.47] + | -1.74 [-2.60,-0.84] - | -3.88 [-4.76,-3.03] - |

## Switching diagnostics

| n | fill | method | runs | first M0 start (s) | M0 time fraction | reconstructions |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 0.90 | BANDIT:8 | 60 | 0.22 | 0.29 | 761.9 |
| 32 | 0.95 | BANDIT:8 | 108 | 0.22 | 0.38 | 659.2 |

CPU/wall ratio: mean 0.993, min 0.937.
