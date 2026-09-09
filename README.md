# Completion-aware layout optimization

This repository contains the submission manuscript and the complete reproducibility
archive for **Completion-aware search with partial witness repair for dense
fixed-shape facility layout**.

## Contents

- `paper/`: `main.tex`/`main.pdf`, `supplement.tex`/`supplement.pdf`,
  bibliography, figures, and the separate `highlights.docx` file.
- `code_submission/`: source code, benchmark files, trained checkpoints, raw
  result files, analysis scripts, and verification tests.

The detailed mapping from manuscript tables and figures to scripts and result
files is given in [`code_submission/README.md`](code_submission/README.md).

## Primary result files

The matched-time comparison in the main manuscript is stored in:

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

To recheck every stored layout and objective in the two primary studies, run
these commands from `code_submission/rl_layout/experiments/`. The output names
below keep the shipped audit reports intact.

```text
python audit_completion_results.py --input completion_comparison_dense50.json --out dense_recheck.json
python audit_completion_results.py --input completion_comparison_boundary80.json --out boundary_recheck.json
```
