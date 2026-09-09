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
  comb_opti_layout/      the legacy hand-authored scenarios (comb_high = OFFICE,
                         hospital = CLINIC) and the original SA implementation
                         (scenario1_combinatorial_sa_executable.py), which is
                         the legacy reference configuration described in the paper
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
  for the primary B0--B3/B2S/M0/R0/M1 comparison. These runs used one AMD
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
| Table 1 (methods B0--B3/B2S/M0/R0/M1) | `completion_search.py` | method definitions used in the two comparison files below |
| Table 2 (datasets and policy coverage) | `bench.py`, training and evaluation scripts listed below | generated suite seeds, checkpoints and result files throughout the archive |
| Table 3 (dense matched-time results) | `run_completion_comparison.py`, `merge_completion_results.py`, `analyze_completion_comparison.py`, `audit_completion_results.py` | `completion_comparison_dense50.json`, `completion_comparison_dense50_summary.json`, `completion_comparison_dense50_audit.json` |
| Figure 1 (anytime quality and initialization) | `fig_completion_comparison.py` | `figs/fig_completion_comparison.pdf` |
| Table 4 (small-instance CP-SAT calibration) | `export_exact.py`, `exact_cpsat.py`, `audit_exact_results.py` (exact env), `run_exact_cells.py`, `exact_report.py` | `exact_specs_full.json`, `exact_results_full.json`, `exact_results_audited.json`, `exact_cells_heur.json` |
| Section 7.3 (fill 0.90 boundary), Supplement S9 | same comparison and audit scripts as Table 3, disjoint tag | `completion_comparison_boundary80.json`, `completion_comparison_boundary80_summary.json`, `completion_comparison_boundary80_audit.json` |
| Appendix Table B.5 (training runs) | `train_construct.py`, `train_improve.py`, `viability_net.py` | checkpoints and histories in `runs/` |

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
| S9 complete matched-time results | `run_completion_comparison.py`, `analyze_completion_comparison.py`, `audit_completion_results.py` | the two primary comparison, summary and audit files above |
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

## Notes

* The held-out suite is fully reproducible from coordinates:
  `bench.suite_cell(split, n, fill, n_inst, tag)` derives its seed from those
  arguments (base seed 20260812), so no instance files need to be shipped.
  Original seed tags in the code and archived metadata must be kept unchanged;
  they identify the published instances, independently of filenames.
* The primary dense study is `completion_comparison_dense50.json`. The
  `short_budget` files preserve an earlier run with a shorter cutoff and fewer
  controls; they are not the source for Table 3. Files named `pilot`, `confirm20`
  and `tune` retain development and validation records.
* `kra30a/kra30b` are absent from `benchmarks/qaplib/` because the upstream
  mirror does not carry them; the paper uses the 32 instances present.
* The numerical records used to generate the manuscript tables are preserved.
  Filename references have been updated to the current archive paths. Output
  behavior differs across scripts, so use a new output filename when rechecking
  a shipped result to avoid overwriting it.
