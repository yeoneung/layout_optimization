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
    v2/                  all experiment code and result JSONs (see map below)
    v2/runs/             trained policy checkpoints (*.pt) with training
                         histories (*_history.json)
    v2/runs_smoke/       tiny throwaway checkpoints used only by
                         _smoke_eval.py (the end-to-end evaluation-path check)
  figs_v2/               the figure PDFs/PNGs included in the manuscript
```

## Environments

Three environments were used; exact versions are recorded in the requirements
files and every new comparison JSON also carries its runtime metadata.

* main (`requirements.txt`): Python 3.10.18 with numpy 2.2.5, scipy 1.15.3,
  torch 2.7.1+cu118, matplotlib 3.10.9. Everything except CP-SAT runs here.
  A GPU is only needed to retrain policies; evaluation of the shipped
  checkpoints runs on CPU, and a CPU-only torch wheel of the same version
  works for that purpose.
* exact (`requirements_exact.txt`): Python 3.11.15 with ortools 9.15, for the
  CP-SAT optimality study and the CP-SAT decision packer only.
* completion (`requirements_completion.txt`): Python 3.7.16 with numpy 1.21.5,
  for the v5 B0--B3/B2S/M0/R0/M1 comparison. These runs used one AMD Ryzen 7 9700X
  CPU process on Windows 10; no learned model or GPU is used.

All scripts are run from `rl_layout/v2/` as working directory, e.g.
`python run_certified_study.py`.

## Map: paper item -> script -> result file

Main text:

| Item | Script(s) | Result file |
|---|---|---|
| Table 1 (feasible volume) | `feasible_volume.py` | `feasible_volume.json` |
| Table 2 / Fig. (improvement-MDP ablation) | `train_improve.py`, `compare_main.py` | `compare_improve.json`, `compare_comb_high.json` |
| Table 3 (certificates compared) + certificate figure | `run_constructive_search.py`, `fig_certificate.py` | `constr_search.json` |
| Table 4 (certified coverage sweep), dense confirmation | `run_certified_study.py`, `analyze_certified_study.py` | `certified_study_v3.json`, `certified_study_summary_v3.json` |
| Table 5 (certified return vs. initial witness) | `witness_gap.py` | `witness_gap_v3.json` |
| Dense confirmation from the same initial layout | `run_completion_comparison.py`, `merge_completion_results.py`, `analyze_completion_comparison.py`, `audit_completion_results.py` | `completion_comparison_dense50_v5.json`, `completion_comparison_dense50_summary_v5.json`, `completion_comparison_dense50_audit_v5.json` |
| Anytime and initialization figure for the main comparison | `fig_completion_comparison.py` | `figs_v2/fig_completion_comparison.pdf` |
| Density boundary from the same initial layout | same scripts, disjoint tag | `completion_comparison_boundary80_v5.json`, `completion_comparison_boundary80_summary_v5.json`, `completion_comparison_boundary80_audit_v5.json` |
| Partial witness repair and controls B0--B3/B2S/M0/R0/M1 | `completion_search.py` | recorded in the two comparison files above |
| Witness-generator coverage frontier | `witness_frontier.py`; `exact_pack.py` (exact env, CP-SAT decision packing) | `witness_frontier.json`, `exact_pack.json` |
| Certified decoder with the exact-marginal selector | `certified_greedy.py` | `certified_greedy_v3.json` |
| Certified decoder on the best-contact witness | `certified_contact.py` | `certified_contact.json` |
| Main suite results, paired tests | `run_bench.py`, `run_bench_policy.py`, `run_bench_meta.py` (tabu/GA), `bench_stats.py`, `bench_report.py` | `bench_test_merged.json`, `bench_val.json` |
| Density sweep table + figure | same as above on the density cells | `bench_density_merged.json`, `bench_density64.json` |
| Small-instance CP-SAT audit | `export_exact.py`, `exact_cpsat.py`, `audit_exact_results.py` (exact env), `run_exact_cells.py`, `exact_report.py` | `exact_specs_full.json`, `exact_results_full.json`, `exact_results_audited.json`, `exact_cells_heur.json` |
| Large-n scaling | `run_bigbench.py` | `bigbench.json`, `bigbench30.json` |
| QAPLIB control | `qap_bench.py` (verified by `test_qap.py`) | `qap_results.json` |
| MCNC/GSRC control | `run_fp.py` (`fp_core.py`, `fp_construct.py`, `test_fp_core.py`) | `fp_results.json` |
| Learned predictive-filter failure | `viability_data.py`, `viability_net.py`, `run_viability.py`, `run_learned_cert.py` | `viability_train.npz` (instance-group IDs included), `runs/viability_net.pt` (split/threshold metadata), `viability_area.json`, `learned_cert.json` |
| All composite figures | `figures.py` | reads the JSONs above, writes `figs_v2/` |

Online Supplement:

| Item | Script(s) | Result file |
|---|---|---|
| S1 zero-shot transfer | `compare_transfer.py`, `run_transfer_greedy.py` | `transfer.json` |
| S2 quality-diversity | `run_qd.py`, `qd_policies.py`, `run_qd_greedy.py`, `run_qd_greedy_sweep.py` | `qd_comb_high.json`, `qd_policies.json`, `qd_greedy_sweep.json` |
| S3 amortized cost | `sa_scaling.py` + training histories in `runs/` | `sa_scaling.json` |
| S4 action granularity | `run_action_axis.py`, `run_track_ac.py` | `compare_action_axis.json` |
| S5 full improvement ablation | `run_track_ac.py`, `compare_main.py` | `compare_improve.json` |
| S6 certificate gallery | `fig_certificate.py` | `figs_v2/fig_gallery.pdf` |
| S7 MCNC/GSRC conversion | `run_fp.py` | `fp_results.json` |
| S8 larger-instance feasibility | `run_bigbench.py` | `bigbench.json`, `bigbench30.json` |
| S9 complete matched-time results | `run_completion_comparison.py`, `analyze_completion_comparison.py`, `audit_completion_results.py` | the two v5 comparison, summary and audit files above |

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
  `certified_study_v3.json` before computing any statistic.
* `certified_greedy.py` asserts the same flag agreement for its own decode
  (coverage cannot depend on the selector), and both it and
  `certified_contact.py` re-verify feasibility of every certified return;
  `witness_frontier.py` and `exact_pack.py` verify every witness/packing they
  count with the deterministic checker.

## Notes

* The held-out suite is fully reproducible from coordinates:
  `bench.suite_cell(split, n, fill, n_inst, tag)` derives its seed from those
  arguments (base seed 20260812), so no instance files need to be shipped.
* `kra30a/kra30b` are absent from `benchmarks/qaplib/` because the upstream
  mirror does not carry them; the paper uses the 32 instances present.
* Result JSONs are the exact files the manuscript tables were generated from;
  re-running a script appends/refreshes records (they resume, not overwrite).
