# Reproducibility archive - "Completion-aware search with partial witness repair for dense fixed-shape facility layout"

Self-contained code, data, trained policies, result files and figure scripts
for every experiment in the main text and the Online Supplement. Nothing in
this folder reads from outside it; all paths are relative.

## Layout

```
code_submission/
  benchmarks/            MCNC (ami33/49) and GSRC (n100/200/300) floorplanning
                         files; qaplib/ holds 32 QAPLIB instances, first line
                         of each .dat is "n optimum"
  comb_opti_layout/      the hand-authored scenarios (comb_high = OFFICE,
                         hospital = CLINIC) and the reference SA implementation
                         (scenario1_combinatorial_sa_executable.py), which is
                         the reference configuration described in the paper
  rl_layout/
    layout_env.py        scenario loader (imports the modules above; the
                         flip_adj flag documents the affinity sign convention)
    experiments/         all experiment code and result JSONs (see map below)
      runs/              trained policy checkpoints (*.pt) with training
                         histories (*_history.json)
      runs_smoke/        tiny throwaway checkpoints used only by
                         _smoke_eval.py (the end-to-end evaluation-path check)
  figs/               the figure PDFs/PNGs included in the manuscript
```

## Environments

Three environments were used; exact versions are recorded in the requirements
files and every new comparison JSON also carries its runtime metadata.

* main (`requirements.txt`): Python 3.10.18 with numpy 2.2.5, scipy 1.15.3,
  torch 2.7.1+cu118, matplotlib 3.10.9 and Pillow 11.0.0. Everything except
  CP-SAT runs here.
  A GPU is only needed to retrain policies; evaluation of the shipped
  checkpoints runs on CPU, and a CPU-only torch wheel of the same version
  works for that purpose.
* exact (`requirements_exact.txt`): Python 3.11.15 with ortools 9.15, for the
  CP-SAT optimality study and the CP-SAT decision packer only.
* completion (`requirements_completion.txt`): Python 3.7.16 with numpy 1.21.5,
  for the one-second B0--B3/B2S/M0/R0/M1 comparison. These runs used one AMD
  Ryzen 7 9700X CPU process on Windows 10; no learned model or GPU is used.

For the recorded CUDA 11.8 environment, run `python -m pip install -r
requirements.txt`; the file includes the PyTorch CUDA wheel index. For CPU-only
evaluation, install the non-torch packages from that file and then run
`python -m pip install torch==2.7.1 --index-url
https://download.pytorch.org/whl/cpu`.

All scripts are run from `rl_layout/experiments/` as working directory, e.g.
`python run_certified_study.py`.

## Map: paper item -> script -> result file

Main text:

| Item | Script(s) | Result file |
|---|---|---|
| Main method table (B0--B3/B2S/M0/R0/M1/ALNS) | `completion_search.py`, `repair_baseline.py` | method definitions for the two matched-time studies |
| Supplement S11 (datasets and policy coverage) | `bench.py`, training and evaluation scripts listed below | generated suite seeds, checkpoints and result files throughout the archive |
| Main Table 2 (10/60-second contrasts) | `run_extended_study.py`, `analyze_extended_study.py`, `render_extended_paper.py` | `../../results/completion_extension_full_cpu/test/summary.json` and adjacent audit/CSV files |
| Figure 1 (128/256-facility quality curves) | `render_extended_paper.py` | `paper/figs/fig_extended_anytime.pdf` from the same audited summary |
| One-second dense mechanism study, Supplement S9 | `run_completion_comparison.py`, `merge_completion_results.py`, `analyze_completion_comparison.py`, `audit_completion_results.py` | `completion_comparison_dense50.json`, `completion_comparison_dense50_summary.json`, `completion_comparison_dense50_audit.json` |
| Main small-instance CP-SAT table | `export_exact.py`, `exact_cpsat.py`, `audit_exact_results.py` (exact env), `run_exact_cells.py`, `exact_report.py` | `exact_specs_full.json`, `exact_results_full.json`, `exact_results_audited.json`, `exact_cells_heur.json` |
| Fill 0.90 boundary, Supplement S9 | same comparison and audit scripts as the dense table, disjoint tag | `completion_comparison_boundary80.json`, `completion_comparison_boundary80_summary.json`, `completion_comparison_boundary80_audit.json` |
| Supplement S12 (training runs) | `train_construct.py`, `train_improve.py`, `viability_net.py` | checkpoints and histories in `runs/` |

Online Supplement:

| Item | Script(s) | Result file |
|---|---|---|
| S1 zero-shot transfer | `compare_transfer.py`, `run_transfer_greedy.py` | `transfer.json` |
| S2 quality-diversity | `run_qd.py`, `qd_policies.py`, `run_qd_greedy.py`, `run_qd_greedy_sweep.py` | `qd_comb_high.json`, `qd_policies.json`, `qd_greedy_sweep.json` |
| S3 amortized cost | `sa_scaling.py` + training histories in `runs/` | `sa_scaling.json` |
| S4 action granularity | `run_action_axis.py`, `run_track_ac.py` | `compare_action_axis.json` |
| S5 full improvement ablation | `run_track_ac.py`, `compare_main.py` | `compare_improve.json` |
| S6 certificate gallery | `fig_certificate.py` | `figs/fig_gallery.pdf` |
| S7 MCNC/GSRC conversion | `run_fp.py` | `fp_results.json` |
| S8 larger-instance feasibility | `run_bigbench.py` | `bigbench.json`, `bigbench30.json` |
| S9 one-second results | `run_completion_comparison.py`, `analyze_completion_comparison.py`, `audit_completion_results.py` | the dense and boundary comparison, summary and audit files above |
| S13 60-second comparison and Figure S4 | `run_extended_study.py`, `analyze_extended_study.py`, `render_extended_paper.py` | `../../results/completion_extension_full_cpu/` |
| S10 feasible volume | `feasible_volume.py` | `feasible_volume.json` |
| S10 improvement-MDP ablation | `train_improve.py`, `compare_main.py` | `compare_improve.json`, `compare_comb_high.json` |
| S10 certificate comparisons and coverage | `run_constructive_search.py`, `run_certified_study.py`, `analyze_certified_study.py`, `witness_gap.py`, `witness_frontier.py`, `exact_pack.py` (exact env) | `constr_search.json`, `certified_study.json`, `certified_study_summary.json`, `witness_gap.json`, `witness_frontier.json`, `exact_pack.json` |
| S10 certified selectors and best-contact witness | `certified_greedy.py`, `certified_contact.py` | `certified_greedy.json`, `certified_contact.json` |
| S10 learned-filter diagnostics | `viability_data.py`, `viability_net.py`, `run_viability.py`, `run_learned_cert.py` | `viability_train.npz`, `runs/viability_net.pt`, `viability_area.json`, `learned_cert.json` |

Other archived diagnostics summarized in the main text:

| Item | Script(s) | Result file |
|---|---|---|
| Secondary main/density suite comparisons | `run_bench.py`, `run_bench_policy.py`, `run_bench_meta.py`, `bench_stats.py`, `bench_report.py` | `bench_test_merged.json`, `bench_val.json`, `bench_density_merged.json`, `bench_density64.json` |
| QAPLIB control | `qap_bench.py`, `test_qap.py` | `qap_results.json` |

`figures.py` reads the archived JSON files and writes the composite figures to
`figs/`. The paper repository stores its submission copies under
`paper/figs/`.

Training (only needed to regenerate checkpoints): `train_construct.py`
produced `runs/S_dord_seed{1,2}.pt` (n=32) and `runs/S64_seed{1,2}.pt` (n=64);
`train_improve.py` produced the `A*/B*/C*/D*` improvement-MDP checkpoints used
by the ablation. `viability_data.py` and `viability_net.py` regenerate the
predictive-filter diagnostic with an instance-disjoint validation split; the
seven validation-derived thresholds are stored in the checkpoint and consumed
unchanged by `run_learned_cert.py` on the test split.

## Policy labels and checkpoint files

The training table in Supplement S12 uses descriptive policy labels. The table below maps those labels to
the files under `rl_layout/experiments/runs/`. For each policy checkpoint,
the corresponding `_history.json` file records its training run, where present.
The completion predictor stores its training metadata in the checkpoint.

| Paper label | Checkpoint file(s) |
|---|---|
| Dense-32, seeds 1/2 | `S_dord_seed1.pt`, `S_dord_seed2.pt` |
| Dense-64, seeds 1/2 | `S64_seed1.pt`, `S64_seed2.pt` |
| Suite, seeds 1/2/3 | `S_bench_seed1.pt`, `S_bench_seed2.pt`, `S_bench_seed3.pt` |
| Transfer | `D1_constr_random.pt` |
| Office | `B1_constr_fixed.pt` |
| Clinic | `B2_constr_hospital.pt` |
| Completion predictor | `viability_net.pt` |

The improvement-MDP labels A0, A0', A1, A2, C1 and C1' are configuration IDs
shared with the ablation in Supplement S5, rather than descriptive policy names.
Their definitions and individual results are retained in that table and in
`compare_improve.json` and `compare_action_axis.json`.

## Verification entry points

* `test_qap.py` - brute-force check of the QAP delta/marginal algebra.
* `test_fp_core.py` - floorplanning parser and objective checks.
* `test_certified_witness.py`, `test_greedy_construct.py`,
  `test_constructive_search.py`, `verify.py`, `_smoke_eval.py` - decoder and
  environment invariants.
* `test_layout_verifier.py` - adversarial tests for duplicate/missing facilities,
  bounds, overlap, edge contact, stale prefixes and interrupted repairs.
* `test_completion_search.py` - common-initialization, deadline, R0 and B2S checks.
* `audit_completion_results.py` - regenerates every instance in the main comparison,
  verifies all stored layouts, recomputes $J_\beta$, and checks that each budget
  row is the best eligible event in its anytime trace.
* `test_bench_families.py` - source-packing and nonslicing-family checks.
* `audit_exact_results.py` - independent geometry checks and the explicit
  integer-to-real CP-SAT bound transfer.
* `witness_gap.py` regenerates every initial witness from its seed and asserts
  bit-exact agreement with the coverage flags recorded in
  `certified_study.json` before computing any statistic.
* `certified_greedy.py` asserts the same flag agreement for its own decode
  (coverage cannot depend on the selector), and both it and
  `certified_contact.py` re-verify feasibility of every certified return;
  `witness_frontier.py` and `exact_pack.py` verify every witness/packing they
  count with the deterministic checker.

## Extended M0/M1/ALNS study

The extension evaluates 32, 64, 128 and 256 facilities, two geometry families,
and fills 0.90 and 0.95, with three search seeds per method. The fixed design is
documented in [EXTENDED_STUDY_PROTOCOL.md](EXTENDED_STUDY_PROTOCOL.md).
The study has its own CPU environment in `requirements_extended.txt` and does
not pool its timing results with the archived one-second comparison.
The completed execution uses all eight physical CPU cores and stores results in
`results/completion_extension_full_cpu`. The interrupted four-worker records
are preserved separately and are not merged into this execution.

From `rl_layout/experiments`:

```text
python test_repair_baseline.py
python test_extended_analysis.py
python finish_extended_study.py --out ../../results/completion_extension_full_cpu --workers 8 --run-validation
```

The final command runs and verifies the validation phase, uses its selected
ALNS configuration, runs/resumes the 60-second test and automatically audits and
analyzes the completed test. It does not automatically edit manuscript claims.
Use the same output root and unchanged experiment sources to resume. A live
runner lock prevents duplicate study processes. The output root contains
`status.json`, `selection.json`, worker logs and per-run JSON files; completed
phases also contain `audit.json`, `summary.json`, CSV tables and `report.md`.
The test additionally produces `primary_contrasts.tex` after a successful audit.
Initialization failures remain in the coverage denominator.

The independent analysis can be rerun with:

```text
python analyze_extended_study.py ../../results/completion_extension_full_cpu/test
```

`repair_baseline.py` is a layout-specific ALNS control with spatial, random and
low-contribution removal, beam/regret insertion, adaptive weights and
nonimproving acceptance. It is not a reproduction of a published routing solver.
The eight-worker validation selected `small_beam` (destroy cap 4, beam width 4,
four ranked candidates plus a legal original position). Its mean 10-second
improvement was 7.8013%, compared with 5.8335% for B2 and 6.9885% for B2S, on
16 validation instances with two seeds. Test results did not enter selection.
The separate four-worker records are excluded from this analysis.
The Git attributes preserve the original line endings of the frozen decoder
source so that the recorded byte-level hashes remain valid in a fresh checkout.
Test completion is established by `test/complete.json` and a passing
`test/audit.json`, not by the presence of a partially populated result directory.

The test completed all 2,124 expected runs on 236 of 240 attempted instances.
All 17,228 saved layouts passed independent verification, with zero objective
recomputation error. The four initialization failures remain in the coverage
denominator. Main-text Table 2 and Figure 1 use this study; Supplement S13
contains its configuration selection, coverage, geometry-specific quality,
full quality curves, target-time summaries, and timing checks.

`render_extended_paper.py` checks the audited summary, reproduces both quality
figures, and prints a JSON object whose `tex` field is the source for
`paper/extended_results.tex`. It does not modify experimental records:

```text
python render_extended_paper.py --paper ../../../paper
```

The primary contrasts use 10,000 geometry-stratified paired bootstrap samples
after averaging seeds within instances. Their 95% intervals are descriptive
and pointwise, not multiplicity-adjusted tests. M1 leads the selected ALNS
control at fill 0.95 at both primary cutoffs across all four sizes, but it does
not uniformly beat M0. At n=256, M1 leads M0 at 10 seconds at both fills and
M0 leads at 60 seconds. These outcomes are all retained in the paper.

## Experiment resource policy

The user requests efficient use of the available machine for experiments and
does not require reserving cores for concurrent interactive work.

- Inspect CPU topology, available memory, GPU capability and the actual workload
  before choosing the execution strategy. Prefer measured throughput over a high
  utilization percentage alone.
- For independent single-threaded search trials, use one pinned worker per
  available physical core with NumPy/BLAS/OpenMP threads set to one. The wrapper
  defaults to the physical core count when `--workers` is omitted. Avoid nested
  thread oversubscription and do not treat SMT siblings as independent physical
  cores in wall-clock solver comparisons.
- Use the GPU for suitable training, inference and large batched numerical
  operations. M0, M1 and ALNS currently execute NumPy geometry/search code on the
  CPU; importing Torch does not make those search kernels GPU accelerated. A GPU
  port would need separate correctness and performance validation.
- Match hardware allocation across compared methods and validation/test. If
  allocation changes during a wall-clock experiment, preserve the interrupted
  records and restart under a separate frozen protocol instead of mixing timings.
- Do not terminate unrelated processes, change global power settings, or modify
  frozen search sources simply to increase a utilization meter.

## Archive notes

* The held-out suite is fully reproducible from coordinates:
  `bench.suite_cell(split, n, fill, n_inst, tag)` derives its seed from those
  arguments (base seed 20260812), so no instance files need to be shipped.
  Original seed tags in the code and archived metadata must be kept unchanged;
  they identify the published instances, independently of filenames.
* The one-second dense mechanism study is `completion_comparison_dense50.json`. The
  `short_budget` files preserve an earlier run with a shorter cutoff and fewer
  controls; they are not the source for the main dense table. Files named `pilot`, `confirm20`
  and `tune` retain development and validation records.
* `kra30a/kra30b` are absent from `benchmarks/qaplib/` because the upstream
  mirror does not carry them; the paper uses the 32 instances present.
* The numerical records used to generate the manuscript tables are preserved.
  Filename references have been updated to the current archive paths. Output
  behavior differs across scripts, so use a new output filename when rechecking
  a shipped result to avoid overwriting it.
