# Completion-aware layout optimization

This repository contains the code, data, and reproducibility archive for
**Completion-aware search with partial witness repair for dense
fixed-shape facility layout**.

## Contents

- `code_submission/`: source code, benchmark files, trained checkpoints, raw
  result files, analysis scripts, and verification tests.

The manuscript and other submission documents are distributed separately.

The detailed mapping from manuscript tables and figures to scripts and result
files is given in [`code_submission/README.md`](code_submission/README.md).

## Primary result files

The main 60-second comparison of M0, M1, and a validation-selected ALNS control
is stored in `code_submission/results/completion_extension_full_cpu/`.
It covers 240 attempted instances with 32 to 256 facilities, two geometry
families, and fills 0.90 and 0.95. All 2,124 expected runs on 236 initialized
instances completed; 17,228 stored layouts passed the independent audit.

- `test/report.md`: coverage and primary 10/60-second contrasts.
- `test/summary.json`, `quality.csv`, `paired_contrasts.csv`, and
  `time_to_target.csv`: complete analyzed results.
- `test/complete.json` and `test/audit.json`: completion and integrity checks.
- `selection.json` and `validation/`: independent ALNS configuration selection.

At fill 0.95, M1 exceeds the selected ALNS control at 10 and 60 seconds for
all tested sizes. With 256 facilities, M1 exceeds M0 at 10 seconds, while
M0 leads at 60 seconds. The manuscript reports the weaker cells as well.

The separate one-second mechanism comparison is stored in:

- `code_submission/rl_layout/experiments/completion_comparison_dense50.json`
- `code_submission/rl_layout/experiments/completion_comparison_dense50_summary.json`
- `code_submission/rl_layout/experiments/completion_comparison_dense50_audit.json`
- `code_submission/rl_layout/experiments/completion_comparison_boundary80.json`
- `code_submission/rl_layout/experiments/completion_comparison_boundary80_summary.json`
- `code_submission/rl_layout/experiments/completion_comparison_boundary80_audit.json`

These files include the run metadata, anytime traces, stored layouts, summary
statistics, and independent audit results used in the paper.
Files and folders use descriptive names. The original seed tags inside result
metadata are preserved because the instance generator uses them to reconstruct
the published benchmarks.

## Basic verification

Install the environment listed in
`code_submission/requirements_completion.txt`, then run the following commands
from `code_submission/rl_layout/experiments/`:

```text
python test_bench_families.py
python test_completion_search.py
```

Other environments and all verification entry points are documented in the
reproducibility archive README.

To recheck every stored layout and objective in the two one-second studies, run
these commands from `code_submission/rl_layout/experiments/`. The output names
below keep the shipped audit reports intact.

```text
python audit_completion_results.py --input completion_comparison_dense50.json --out dense_recheck.json
python audit_completion_results.py --input completion_comparison_boundary80.json --out boundary_recheck.json
```
