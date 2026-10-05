# EAAI phase report: test

Runs present: 540 of 540 expected on 60 initialized instances (60 attempted so far); instances with all methods and seeds: 60.

Methods: M0, BANDITP3:8, LRN:E128_f9095_seed1

Seeds averaged within instance; geometries equally weighted; intervals are descriptive
geometry-stratified paired bootstrap intervals.

## Improvement over the common initial layout at 10 s (%)

| n | fill | inst | M0 | BANDITP3:8 | LRN:E128_f9095_seed1 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.90 | 20 | 17.13 | **17.23** | 8.96 | n/a |
| 128 | 0.95 | 40 | **11.76** | 11.74 | 6.55 | n/a |

### Summary across cells at 10 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| M0 | - | 1.69 | 1.04 | 1 |
| BANDITP3:8 | 0/2/0 | 1.50 | 1.00 | 1 |
| LRN:E128_f9095_seed1 | 0/0/2 | 2.81 | 7.74 | 0 |

### Method minus M0 at 10 s (pp, 95% interval)

| n | fill | BANDITP3:8 | LRN:E128_f9095_seed1 |
|---:|---:|---:|---:|
| 128 | 0.90 | +0.10 [-0.41,+0.59] | -8.18 [-10.33,-6.21] - |
| 128 | 0.95 | -0.03 [-0.95,+0.85] | -5.21 [-7.26,-3.20] - |

## Improvement over the common initial layout at 60 s (%)

| n | fill | inst | M0 | BANDITP3:8 | LRN:E128_f9095_seed1 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 0.90 | 20 | 19.04 | **21.38** | 12.92 | n/a |
| 128 | 0.95 | 40 | 14.70 | **15.75** | 12.11 | n/a |

### Summary across cells at 60 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| M0 | - | 2.01 | 2.02 | 0 |
| BANDITP3:8 | 2/0/0 | 1.18 | 0.32 | 2 |
| LRN:E128_f9095_seed1 | 0/0/2 | 2.81 | 6.37 | 0 |

### Method minus M0 at 60 s (pp, 95% interval)

| n | fill | BANDITP3:8 | LRN:E128_f9095_seed1 |
|---:|---:|---:|---:|
| 128 | 0.90 | +2.34 [+1.87,+2.79] + | -6.12 [-7.76,-4.65] - |
| 128 | 0.95 | +1.05 [+0.35,+1.66] + | -2.59 [-3.58,-1.57] - |

## Switching diagnostics

| n | fill | method | runs | first M0 start (s) | M0 time fraction | reconstructions |
|---:|---:|---|---:|---:|---:|---:|
| 128 | 0.90 | BANDITP3:8 | 60 | 1.10 | 0.39 | 33.9 |
| 128 | 0.95 | BANDITP3:8 | 120 | 1.29 | 0.52 | 19.2 |

CPU/wall ratio: mean 0.992, min 0.922.
