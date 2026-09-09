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

- `code_submission/rl_layout/v2/completion_comparison_dense50_v5.json`
- `code_submission/rl_layout/v2/completion_comparison_dense50_summary_v5.json`
- `code_submission/rl_layout/v2/completion_comparison_dense50_audit_v5.json`
- `code_submission/rl_layout/v2/completion_comparison_boundary80_v5.json`
- `code_submission/rl_layout/v2/completion_comparison_boundary80_summary_v5.json`
- `code_submission/rl_layout/v2/completion_comparison_boundary80_audit_v5.json`

These files include the run metadata, anytime traces, stored layouts, summary
statistics, and independent audit results used in the paper.
The suffixes in these filenames are retained as immutable experiment identifiers;
the submission files in `paper/` do not carry version suffixes.

## Basic verification

Install the environment listed in
`code_submission/requirements_completion.txt`, then run the following commands
from `code_submission/rl_layout/v2/`:

```text
python test_bench_families.py
python test_completion_search.py
```

Other environments and all verification entry points are documented in the
reproducibility archive README.
