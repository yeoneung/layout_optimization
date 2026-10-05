# Phase 0: oracle headroom and predictability from the published 60-second study

Source: `code_submission/results/completion_extension_full_cpu/test` (2,124 runs, audited).
Seeds averaged within instance; geometries equally weighted.

## 1. Oracle headroom (percentage points of initial-objective improvement)

oracle_M0_M1 = per-instance better of M0 and M1. headroom = oracle minus the better *cell-level* single method.
regret_always_X = oracle minus X. A switching policy can at most recover the headroom.

### 10 seconds

| n | fill | M0 | M1 | ALNS | oracle(M0,M1) | oracle(all 3) | headroom | regret M0 | regret M1 | M1 win frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32.0 | 0.90 | 22.01 | 20.41 | 19.40 | 22.46 | 22.86 | **0.45** | 0.45 | 2.05 | 0.45 |
| 32.0 | 0.95 | 14.17 | 15.52 | 12.06 | 16.27 | 16.36 | **0.75** | 2.09 | 0.75 | 0.64 |
| 64.0 | 0.90 | 16.89 | 14.91 | 13.54 | 17.24 | 17.26 | **0.36** | 0.36 | 2.34 | 0.20 |
| 64.0 | 0.95 | 12.90 | 11.89 | 7.72 | 13.98 | 13.98 | **1.08** | 1.08 | 2.09 | 0.50 |
| 128.0 | 0.90 | 16.82 | 14.60 | 9.56 | 16.87 | 16.88 | **0.05** | 0.05 | 2.27 | 0.15 |
| 128.0 | 0.95 | 10.54 | 9.56 | 4.46 | 11.84 | 11.88 | **1.30** | 1.30 | 2.28 | 0.45 |
| 256.0 | 0.90 | 9.70 | 13.69 | 3.69 | 13.76 | 13.76 | **0.07** | 4.05 | 0.07 | 0.95 |
| 256.0 | 0.95 | 4.76 | 7.66 | 1.75 | 7.82 | 7.82 | **0.16** | 3.05 | 0.16 | 0.82 |

### 60 seconds

| n | fill | M0 | M1 | ALNS | oracle(M0,M1) | oracle(all 3) | headroom | regret M0 | regret M1 | M1 win frac |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 32.0 | 0.90 | 24.43 | 24.32 | 23.67 | 25.58 | 26.20 | **1.15** | 1.15 | 1.26 | 0.70 |
| 32.0 | 0.95 | 17.58 | 18.81 | 15.84 | 19.73 | 19.78 | **0.93** | 2.16 | 0.93 | 0.69 |
| 64.0 | 0.90 | 18.27 | 17.49 | 19.22 | 18.85 | 19.88 | **0.57** | 0.57 | 1.36 | 0.40 |
| 64.0 | 0.95 | 14.86 | 15.38 | 13.70 | 16.49 | 16.64 | **1.11** | 1.63 | 1.11 | 0.65 |
| 128.0 | 0.90 | 18.83 | 16.23 | 16.65 | 18.83 | 19.38 | **0.00** | 0.00 | 2.59 | 0.00 |
| 128.0 | 0.95 | 14.33 | 11.97 | 8.90 | 15.05 | 15.08 | **0.72** | 0.72 | 3.07 | 0.33 |
| 256.0 | 0.90 | 19.39 | 14.70 | 10.77 | 19.39 | 19.39 | **0.00** | 0.00 | 4.70 | 0.00 |
| 256.0 | 0.95 | 10.38 | 8.58 | 4.45 | 11.11 | 11.11 | **0.73** | 0.73 | 2.53 | 0.33 |

## 2. Can the first seconds of M1 predict the 60-second winner?

Label: M1 beats M0 at 60 s (seed-averaged). Policy value = mean 60 s improvement when the
predicted winner is run alone; compare with M0-only, M1-only and the oracle.

### Early window t0=1 (positive rate 0.42, 236 instances)

| features | model | CV scheme | accuracy | AUC | policy | M0 | M1 | oracle |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instance_only | logistic | stratified5 | 0.746 | 0.813 | 16.63 | 16.25 | 15.12 | 17.24 |
| instance_only | logistic | leave_one_cell_out | 0.691 | 0.747 | 16.52 | 16.25 | 15.12 | 17.24 |
| instance_only | logistic | leave_one_size_out | 0.678 | 0.749 | 16.45 | 16.25 | 15.12 | 17.24 |
| instance_only | hist_gb | stratified5 | 0.733 | 0.804 | 16.55 | 16.25 | 15.12 | 17.24 |
| instance_only | hist_gb | leave_one_cell_out | 0.631 | 0.705 | 16.17 | 16.25 | 15.12 | 17.24 |
| instance_only | hist_gb | leave_one_size_out | 0.487 | 0.522 | 15.77 | 16.25 | 15.12 | 17.24 |
| early_M1_only | logistic | stratified5 | 0.699 | 0.731 | 16.39 | 16.25 | 15.12 | 17.24 |
| early_M1_only | logistic | leave_one_cell_out | 0.610 | 0.621 | 16.09 | 16.25 | 15.12 | 17.24 |
| early_M1_only | logistic | leave_one_size_out | 0.551 | 0.578 | 15.97 | 16.25 | 15.12 | 17.24 |
| early_M1_only | hist_gb | stratified5 | 0.716 | 0.795 | 16.39 | 16.25 | 15.12 | 17.24 |
| early_M1_only | hist_gb | leave_one_cell_out | 0.627 | 0.691 | 16.06 | 16.25 | 15.12 | 17.24 |
| early_M1_only | hist_gb | leave_one_size_out | 0.636 | 0.716 | 16.18 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | logistic | stratified5 | 0.733 | 0.823 | 16.57 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | logistic | leave_one_cell_out | 0.720 | 0.782 | 16.53 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | logistic | leave_one_size_out | 0.627 | 0.723 | 16.18 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | hist_gb | stratified5 | 0.758 | 0.848 | 16.60 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | hist_gb | leave_one_cell_out | 0.703 | 0.776 | 16.36 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | hist_gb | leave_one_size_out | 0.644 | 0.720 | 16.26 | 16.25 | 15.12 | 17.24 |

In-sample (n, fill) lookup table: accuracy 0.725, policy 16.51.

Univariate AUC (>0.5 favors M1 when large): plate 0.23, events 0.74, log_n 0.27, initial_J_per_n 0.30, improve_rate 0.32, frac_repair 0.37, last_gain_s 0.37, mean_facility_area 0.37, imp_t0 0.62, frac_fallback 0.61, fill 0.60, is_nonslicing 0.59, frac_reuse 0.58, slope 0.45, improvements 0.45, achieved_fill 0.52

corr(M1-M0 at 10 s, M1-M0 at 60 s) = 0.47

Per-cell values for the best scheme by policy value:

| cell | policy | M0 | M1 | oracle | accuracy |
|---|---:|---:|---:|---:|---:|
| 128_0.9 | 18.83 | 18.83 | 16.23 | 18.83 | 1.00 |
| 128_0.95 | 14.16 | 14.33 | 11.97 | 15.05 | 0.62 |
| 256_0.9 | 19.39 | 19.39 | 14.70 | 19.39 | 1.00 |
| 256_0.95 | 10.59 | 10.38 | 8.58 | 11.11 | 0.75 |
| 32_0.9 | 25.19 | 24.43 | 24.32 | 25.58 | 0.80 |
| 32_0.95 | 18.81 | 17.58 | 18.81 | 19.73 | 0.69 |
| 64_0.9 | 18.28 | 18.27 | 17.49 | 18.85 | 0.65 |
| 64_0.95 | 15.62 | 14.86 | 15.38 | 16.49 | 0.68 |

### Early window t0=3 (positive rate 0.42, 236 instances)

| features | model | CV scheme | accuracy | AUC | policy | M0 | M1 | oracle |
|---|---|---|---:|---:|---:|---:|---:|---:|
| instance_only | logistic | stratified5 | 0.746 | 0.813 | 16.63 | 16.25 | 15.12 | 17.24 |
| instance_only | logistic | leave_one_cell_out | 0.691 | 0.747 | 16.52 | 16.25 | 15.12 | 17.24 |
| instance_only | logistic | leave_one_size_out | 0.678 | 0.749 | 16.45 | 16.25 | 15.12 | 17.24 |
| instance_only | hist_gb | stratified5 | 0.733 | 0.804 | 16.55 | 16.25 | 15.12 | 17.24 |
| instance_only | hist_gb | leave_one_cell_out | 0.631 | 0.705 | 16.17 | 16.25 | 15.12 | 17.24 |
| instance_only | hist_gb | leave_one_size_out | 0.487 | 0.522 | 15.77 | 16.25 | 15.12 | 17.24 |
| early_M1_only | logistic | stratified5 | 0.703 | 0.760 | 16.41 | 16.25 | 15.12 | 17.24 |
| early_M1_only | logistic | leave_one_cell_out | 0.597 | 0.641 | 16.09 | 16.25 | 15.12 | 17.24 |
| early_M1_only | logistic | leave_one_size_out | 0.521 | 0.581 | 15.67 | 16.25 | 15.12 | 17.24 |
| early_M1_only | hist_gb | stratified5 | 0.758 | 0.838 | 16.60 | 16.25 | 15.12 | 17.24 |
| early_M1_only | hist_gb | leave_one_cell_out | 0.614 | 0.661 | 16.16 | 16.25 | 15.12 | 17.24 |
| early_M1_only | hist_gb | leave_one_size_out | 0.648 | 0.704 | 16.25 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | logistic | stratified5 | 0.742 | 0.831 | 16.58 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | logistic | leave_one_cell_out | 0.716 | 0.770 | 16.51 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | logistic | leave_one_size_out | 0.593 | 0.666 | 15.96 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | hist_gb | stratified5 | 0.792 | 0.856 | 16.72 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | hist_gb | leave_one_cell_out | 0.686 | 0.725 | 16.32 | 16.25 | 15.12 | 17.24 |
| instance_plus_early | hist_gb | leave_one_size_out | 0.585 | 0.701 | 15.92 | 16.25 | 15.12 | 17.24 |

In-sample (n, fill) lookup table: accuracy 0.725, policy 16.51.

Univariate AUC (>0.5 favors M1 when large): plate 0.23, improve_rate 0.24, events 0.75, log_n 0.27, initial_J_per_n 0.30, improvements 0.33, frac_repair 0.36, frac_fallback 0.64, mean_facility_area 0.37, fill 0.60, is_nonslicing 0.59, imp_t0 0.58, slope 0.43, last_gain_s 0.53, achieved_fill 0.52, frac_reuse 0.51

corr(M1-M0 at 10 s, M1-M0 at 60 s) = 0.47

Per-cell values for the best scheme by policy value:

| cell | policy | M0 | M1 | oracle | accuracy |
|---|---:|---:|---:|---:|---:|
| 128_0.9 | 18.83 | 18.83 | 16.23 | 18.83 | 1.00 |
| 128_0.95 | 14.14 | 14.33 | 11.97 | 15.05 | 0.65 |
| 256_0.9 | 19.39 | 19.39 | 14.70 | 19.39 | 1.00 |
| 256_0.95 | 10.79 | 10.38 | 8.58 | 11.11 | 0.85 |
| 32_0.9 | 25.44 | 24.43 | 24.32 | 25.58 | 0.90 |
| 32_0.95 | 18.69 | 17.58 | 18.81 | 19.73 | 0.69 |
| 64_0.9 | 18.63 | 18.27 | 17.49 | 18.85 | 0.80 |
| 64_0.95 | 15.76 | 14.86 | 15.38 | 16.49 | 0.70 |
