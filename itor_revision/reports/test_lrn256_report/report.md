# EAAI phase report: test

Runs present: 540 of 540 expected on 60 initialized instances (60 attempted so far); instances with all methods and seeds: 60.

Methods: M0, BANDITP3:8, LRN:E256_f9095_seed1

Seeds averaged within instance; geometries equally weighted; intervals are descriptive
geometry-stratified paired bootstrap intervals.

## Improvement over the common initial layout at 10 s (%)

| n | fill | inst | M0 | BANDITP3:8 | LRN:E256_f9095_seed1 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.90 | 20 | 11.66 | **14.09** | 0.00 | n/a |
| 256 | 0.95 | 40 | 5.80 | **7.84** | 0.00 | n/a |

### Summary across cells at 10 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| M0 | - | 1.71 | 2.64 | 0 |
| BANDITP3:8 | 2/0/0 | 1.30 | 0.41 | 2 |
| LRN:E256_f9095_seed1 | 0/0/2 | 2.99 | 11.37 | 0 |

### Method minus M0 at 10 s (pp, 95% interval)

| n | fill | BANDITP3:8 | LRN:E256_f9095_seed1 |
|---:|---:|---:|---:|
| 256 | 0.90 | +2.43 [+1.37,+3.47] + | -11.66 [-13.27,-10.24] - |
| 256 | 0.95 | +2.04 [+1.03,+3.11] + | -5.80 [-6.73,-4.87] - |

## Improvement over the common initial layout at 60 s (%)

| n | fill | inst | M0 | BANDITP3:8 | LRN:E256_f9095_seed1 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|
| 256 | 0.90 | 20 | 19.71 | **19.81** | 5.41 | n/a |
| 256 | 0.95 | 40 | **10.76** | 10.10 | 0.00 | n/a |

### Summary across cells at 60 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| M0 | - | 1.42 | 0.69 | 1 |
| BANDITP3:8 | 0/2/0 | 1.58 | 0.97 | 1 |
| LRN:E256_f9095_seed1 | 0/0/2 | 3.00 | 13.22 | 0 |

### Method minus M0 at 60 s (pp, 95% interval)

| n | fill | BANDITP3:8 | LRN:E256_f9095_seed1 |
|---:|---:|---:|---:|
| 256 | 0.90 | +0.10 [-0.29,+0.50] | -14.30 [-16.12,-12.61] - |
| 256 | 0.95 | -0.66 [-1.57,+0.27] | -10.76 [-11.90,-9.68] - |

## Switching diagnostics

| n | fill | method | runs | first M0 start (s) | M0 time fraction | reconstructions |
|---:|---:|---|---:|---:|---:|---:|
| 256 | 0.90 | BANDITP3:8 | 60 | 6.50 | 0.44 | 6.2 |
| 256 | 0.95 | BANDITP3:8 | 120 | 7.17 | 0.60 | 3.5 |

CPU/wall ratio: mean 0.993, min 0.938.
