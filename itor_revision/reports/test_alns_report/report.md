# EAAI phase report: test

Runs present: 1416 of 1416 expected on 236 initialized instances (240 attempted so far); instances with all methods and seeds: 236.

Methods: ALNS:alns_c8_b4_t25, M0

Seeds averaged within instance; geometries equally weighted; intervals are descriptive
geometry-stratified paired bootstrap intervals.

## Improvement over the common initial layout at 10 s (%)

| n | fill | inst | ALNS:alns_c8_b4_t25 | M0 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.90 | 20 | **23.21** | 22.51 | n/a |
| 32 | 0.95 | 36 | **15.02** | 14.83 | n/a |
| 64 | 0.90 | 20 | 15.65 | **17.24** | n/a |
| 64 | 0.95 | 40 | 9.27 | **13.18** | n/a |
| 128 | 0.90 | 20 | 9.50 | **17.21** | n/a |
| 128 | 0.95 | 40 | 4.36 | **11.85** | n/a |
| 256 | 0.90 | 20 | 3.16 | **11.76** | n/a |
| 256 | 0.95 | 40 | 1.49 | **5.83** | n/a |

### Summary across cells at 10 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| ALNS:alns_c8_b4_t25 | 0/2/6 | 1.78 | 4.59 | 2 |
| M0 | - | 1.22 | 0.50 | 6 |

### Method minus M0 at 10 s (pp, 95% interval)

| n | fill | ALNS:alns_c8_b4_t25 |
|---:|---:|---:|
| 32 | 0.90 | +0.70 [-0.89,+2.16] |
| 32 | 0.95 | +0.19 [-0.84,+1.23] |
| 64 | 0.90 | -1.59 [-2.83,-0.41] - |
| 64 | 0.95 | -3.91 [-5.13,-2.73] - |
| 128 | 0.90 | -7.71 [-10.26,-5.53] - |
| 128 | 0.95 | -7.48 [-8.76,-6.27] - |
| 256 | 0.90 | -8.60 [-9.91,-7.46] - |
| 256 | 0.95 | -4.35 [-5.25,-3.43] - |

## Improvement over the common initial layout at 60 s (%)

| n | fill | inst | ALNS:alns_c8_b4_t25 | M0 | oracle(M0,M1) |
|---:|---:|---:|---:|---:|---:|
| 32 | 0.90 | 20 | **27.69** | 24.84 | n/a |
| 32 | 0.95 | 36 | **19.50** | 18.31 | n/a |
| 64 | 0.90 | 20 | **21.51** | 18.59 | n/a |
| 64 | 0.95 | 40 | **16.46** | 15.12 | n/a |
| 128 | 0.90 | 20 | 17.48 | **19.08** | n/a |
| 128 | 0.95 | 40 | 9.58 | **14.73** | n/a |
| 256 | 0.90 | 20 | 10.20 | **19.73** | n/a |
| 256 | 0.95 | 40 | 4.04 | **10.76** | n/a |

### Summary across cells at 60 s

| method | vs M0 W/T/L | mean rank | mean gap to all-method oracle (pp) | cells where best |
|---|---:|---:|---:|---:|
| ALNS:alns_c8_b4_t25 | 3/1/4 | 1.57 | 3.26 | 4 |
| M0 | - | 1.43 | 1.43 | 4 |

### Method minus M0 at 60 s (pp, 95% interval)

| n | fill | ALNS:alns_c8_b4_t25 |
|---:|---:|---:|
| 32 | 0.90 | +2.86 [+1.60,+4.08] + |
| 32 | 0.95 | +1.20 [-0.19,+2.52] |
| 64 | 0.90 | +2.93 [+2.12,+3.71] + |
| 64 | 0.95 | +1.34 [+0.44,+2.23] + |
| 128 | 0.90 | -1.59 [-3.21,-0.10] - |
| 128 | 0.95 | -5.15 [-6.33,-3.96] - |
| 256 | 0.90 | -9.53 [-11.61,-7.73] - |
| 256 | 0.95 | -6.72 [-7.77,-5.70] - |

CPU/wall ratio: mean 0.993, min 0.920.
