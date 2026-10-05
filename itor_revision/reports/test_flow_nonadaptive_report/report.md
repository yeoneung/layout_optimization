# EAAI phase report: test

Runs present: 948 of 948 expected on 158 initialized instances (160 attempted so far); instances with all methods and seeds: 158.

Methods: BANDITP3:8, RR3, RND3

Seeds averaged within instance; geometries equally weighted; intervals are descriptive
geometry-stratified paired bootstrap intervals.

## Improvement over the common initial layout at 10 s (%)

| n | fill | inst | BANDITP3:8 | RR3 | RND3 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.90 | 20 | **29.26** | 28.21 | 28.99 | n/a |
| 32 | 0.95 | 18 | 20.51 | **21.29** | 21.09 | n/a |
| 64 | 0.90 | 20 | 15.91 | 16.60 | **16.79** | n/a |
| 64 | 0.95 | 20 | 11.78 | 12.21 | **12.59** | n/a |
| 128 | 0.90 | 20 | 9.78 | **10.31** | 9.41 | n/a |
| 128 | 0.95 | 20 | **5.56** | 5.02 | 4.12 | n/a |
| 256 | 0.90 | 20 | 2.84 | 3.01 | **4.25** | n/a |
| 256 | 0.95 | 20 | **1.06** | **1.06** | 1.02 | n/a |

### Summary across cells at 10 s

| method |  | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|
| BANDITP3:8 |  | 2.02 | 1.37 | 3 |
| RR3 |  | 1.97 | 1.25 | 3 |
| RND3 |  | 2.01 | 1.18 | 3 |

## Improvement over the common initial layout at 60 s (%)

| n | fill | inst | BANDITP3:8 | RR3 | RND3 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|---:|
| 32 | 0.90 | 20 | **30.80** | 30.76 | 30.69 | n/a |
| 32 | 0.95 | 18 | 22.66 | 23.00 | **23.07** | n/a |
| 64 | 0.90 | 20 | 20.25 | 20.39 | **21.10** | n/a |
| 64 | 0.95 | 20 | 16.01 | 16.08 | **16.81** | n/a |
| 128 | 0.90 | 20 | 13.87 | 14.43 | **14.49** | n/a |
| 128 | 0.95 | 20 | 8.83 | **9.33** | 8.56 | n/a |
| 256 | 0.90 | 20 | 8.82 | 9.24 | **9.26** | n/a |
| 256 | 0.95 | 20 | **4.34** | 4.25 | 3.08 | n/a |

### Summary across cells at 60 s

| method |  | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|
| BANDITP3:8 |  | 2.14 | 1.18 | 2 |
| RR3 |  | 1.88 | 0.94 | 1 |
| RND3 |  | 1.97 | 0.99 | 5 |

## Switching diagnostics

| n | fill | method | runs | first M0 start (s) | M0 time fraction | reconstructions |
|---:|---:|---|---:|---:|---:|---:|
| 32 | 0.90 | BANDITP3:8 | 40 | 0.03 | 0.43 | 884.3 |
| 32 | 0.90 | RR3 | 40 | 0.03 | 0.41 | 853.3 |
| 32 | 0.90 | RND3 | 40 | 0.10 | 0.44 | 898.9 |
| 32 | 0.95 | BANDITP3:8 | 36 | 0.03 | 0.49 | 627.4 |
| 32 | 0.95 | RR3 | 36 | 0.03 | 0.45 | 581.7 |
| 32 | 0.95 | RND3 | 36 | 0.12 | 0.50 | 651.6 |
| 64 | 0.90 | BANDITP3:8 | 40 | 0.10 | 0.44 | 236.1 |
| 64 | 0.90 | RR3 | 40 | 0.10 | 0.41 | 223.1 |
| 64 | 0.90 | RND3 | 40 | 0.24 | 0.45 | 240.1 |
| 64 | 0.95 | BANDITP3:8 | 40 | 0.13 | 0.51 | 120.5 |
| 64 | 0.95 | RR3 | 40 | 0.13 | 0.47 | 110.0 |
| 64 | 0.95 | RND3 | 40 | 0.24 | 0.54 | 126.2 |
| 128 | 0.90 | BANDITP3:8 | 40 | 0.37 | 0.46 | 54.3 |
| 128 | 0.90 | RR3 | 40 | 0.37 | 0.44 | 49.6 |
| 128 | 0.90 | RND3 | 40 | 0.83 | 0.48 | 53.8 |
| 128 | 0.95 | BANDITP3:8 | 40 | 0.54 | 0.59 | 23.5 |
| 128 | 0.95 | RR3 | 40 | 0.54 | 0.53 | 18.2 |
| 128 | 0.95 | RND3 | 40 | 1.01 | 0.64 | 21.9 |
| 256 | 0.90 | BANDITP3:8 | 40 | 1.49 | 0.55 | 11.8 |
| 256 | 0.90 | RR3 | 40 | 1.50 | 0.52 | 9.8 |
| 256 | 0.90 | RND3 | 40 | 3.92 | 0.58 | 11.5 |
| 256 | 0.95 | BANDITP3:8 | 40 | 2.39 | 0.66 | 4.2 |
| 256 | 0.95 | RR3 | 40 | 2.39 | 0.70 | 3.8 |
| 256 | 0.95 | RND3 | 40 | 3.63 | 0.77 | 5.0 |

CPU/wall ratio: mean 0.993, min 0.937.
