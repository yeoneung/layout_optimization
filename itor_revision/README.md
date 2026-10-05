# ITOR revision: certificate-preserving operator portfolios

This is the public computational archive for **How much of a feasible layout
to rebuild: certificate-preserving operator portfolios for dense facility
layout under a time budget**. It supplements the precursor archive in
[`../code_submission/`](../code_submission/) and must remain beside it.
Manuscript and submission documents are maintained separately.

## Findings and scope

On the development adjacency suite (236 initialized instances), PORT3-R,
PORT3-B and PORT3-U have the three best mean ranks among sixteen methods at
60 seconds: 3.90, 4.28 and 5.33. Across the development, frozen and flow
suites, the tested sliding-window bandit has no consistent advantage over
matched cyclic rotation. Of 48 cell/time comparisons, 3 favor the bandit,
5 favor rotation and 40 have descriptive paired intervals including zero.
The two time cutoffs reuse the same runs; these are not 48 independent tests,
and unresolved differences do not establish equivalence.

On the frozen suite, PORT3-B improves on partial repair in all eight cells
at 60 seconds; its largest cell-mean deficit to regeneration is 0.46 percentage
points. The recorded one-point condition concerns cell means, not a
confidence-bound noninferiority test. Dedicated ALNS generally remains
stronger under material-flow cost. All instances are synthetic; this archive
contains no measured industrial case study.

## Contents and version

- `code/`: portfolio, flow and obstacle extensions; runner, analysis, audit,
  figure/table generators and correctness checks.
- `models/`: the fitted switching model and warm-start arm priors.
- `records/`: unchanged phase protocols, completion records and original
  audits, plus the recorded frozen-suite criterion.
- `reports/`: CSV summaries and readable reports, including the pooled
  sixteen-method results and all three matched-control suites.
- `analysis/`: precursor-trace diagnostics and the fixed-bin density analysis.
- `DATASETS.json`: SHA-256, byte count and download link for each release asset.
- `VALIDATION.json`: checks performed when preparing this public package.

The raw JSON traces, initial layouts, budget layouts and four constructive
policy checkpoints are assets of
[release itor-v1.0.0](https://github.com/yeoneung/layout_optimization/releases/tag/itor-v1.0.0).
The release's source archive contains this directory and the precursor code.
The historical `EAAI` names in scripts, tokens and protocol hash keys are
preserved because they identify the implementation used during the study;
they do not identify the current target journal.

## Environment and quick checks

The recorded environment is Python 3.12.10 on Windows, with package versions
in `requirements.txt` and `environment.json`. Protocols retain each phase's
original environment. The study pinned single-threaded workers to P-cores
of an i9-14900KF; runtime rankings need not transfer unchanged to other hardware.
CPU PyTorch suffices for audits and evaluation. The original learned-policy
training used a GPU. Install from the repository root:

```text
python -m pip install -r itor_revision/requirements.txt
python itor_revision/code/test_methods_ext.py
python itor_revision/code/test_flow_objective.py
python itor_revision/code/test_constraints.py
python itor_revision/code/test_learned_methods.py
```

The checks exercise parsing, feasible/non-worsening outputs, eligible budget
rows, round-robin order, flow objectives, obstacle avoidance and the learned
constructor interface. They are correctness checks, not reruns of the reported
comparative experiments.

## Obtain raw results

Choose phases instead of downloading every archive if only one result is
needed. All assets together occupy about 3.3 GiB compressed. Each archive
contains its phase prefix, such as `test_frozen/test/`.

```text
python itor_revision/download_data.py --list
python itor_revision/download_data.py test_frozen test_nonadaptive test_flow_nonadaptive
python -m zipfile -e itor_revision/downloads/test_frozen.zip itor_revision/data
```

Repeat the extraction for each downloaded ZIP. The downloader checks both
byte count and SHA-256 before accepting a file. The `policies` archive holds
`E32_f9095_seed1.pt`, `E64_f9095_seed1.pt`, `E128_f9095_seed1.pt` and
`E256_f9095_seed1.pt`, with training histories. Extract that archive into
`~/layout_eaai_results/` for the unchanged learned-method wrapper to find it.
The two earlier small-policy checkpoints are already in `code_submission/`.

## Which results support which claims?

| Result | Raw phase(s) | Report or analysis |
| --- | --- | --- |
| Sixteen-method adjacency comparison | `test`, `test_portfolio_pref`, `test_alns`, `test_nonadaptive` | `reports/test_report_pooled16/` |
| Matched selection rules, adjacency | `test_nonadaptive` | `reports/test_nonadaptive_report/` |
| Frozen suite | `test_frozen` | `reports/test_frozen_report/` |
| Matched selection rules, flow | `test_flow_nonadaptive` | `reports/test_flow_nonadaptive_report/` |
| Initial flow controls and later portfolios | `test_flow`, `test_flow_portfolio` | Corresponding `*_report/` folders |
| Obstacles and dock | `test_obstacles` | `reports/test_obstacles_report/` |
| Retuned ALNS | `alns_val`, `test_alns` | `records/alns_val/selection.json`, `reports/test_alns_report/` |
| Learned constructors | `test_lrn32`, `test_lrn64`, `test_lrn128`, `test_lrn256`, `policies` | Corresponding `*_report/` folders |
| Offline models and pilot | `train`, `pilot` | `models/`, `reports/train_report/`, `reports/pilot_report/` |
| Conditional crossover times | Precursor `completion_extension_full_cpu/test` | `analysis/phase0_headroom.py`, then `phase0b_rate_crossover.py` |
| Observed improvement densities | `test_portfolio_pref` | `analysis/fixed_bin_density.py` |

The pilot is a partial exploratory phase: its archive deliberately has no
`complete.json` or successful phase audit. It is not part of the final test
comparison. No preliminary smoke runs or aborted Dropbox-lock runs are included.

## Method names

| Paper | Stored token |
| --- | --- |
| PORT3-B / PORT3-R / PORT3-U | `BANDITP3:8` / `RR3` / `RND3` |
| PORT5 | `BANDITP:8` |
| M0 / M1 | `M0` / `M1` |
| ALNS / ALNSR | `ALNS:small_beam` / `ALNS:alns_c8_b4_t25` |
| CAP8 / CAP16 | `CAP:8` / `CAP:16` |
| SW3 / INTL | `SW:fixed3` / `INTERLEAVE` |
| CAPB / CAPB2 / CAPBW | `BANDIT:8` / `BANDIT2:8` / `BANDITW:priors_v1` |
| SWL | `SWL:swl_v1` |

## Recompute analyses without rerunning search

After extracting the needed archives, run from the repository root. The
following outputs go to a separate `recomputed/` directory.

```text
python itor_revision/code/analyze_eaai.py itor_revision/data/test_frozen/test --out itor_revision/recomputed/frozen --replicates 10000 --controls M0,M1,ALNS:small_beam,RR3
python itor_revision/code/analyze_eaai.py itor_revision/data/test_nonadaptive/test --out itor_revision/recomputed/nonadaptive --replicates 10000 --controls M0,RR3,RND3
python itor_revision/code/analyze_eaai.py itor_revision/data/test_flow_nonadaptive/test --out itor_revision/recomputed/flow_nonadaptive --metric cost --replicates 10000 --controls RR3,RND3
python itor_revision/code/analyze_eaai.py itor_revision/data/test/test --extra-dirs itor_revision/data/test_portfolio_pref/test,itor_revision/data/test_alns/test,itor_revision/data/test_nonadaptive/test --out itor_revision/recomputed/pooled16 --replicates 10000 --controls M0,M1,ALNS:small_beam,BANDIT:8,RR3
python itor_revision/analysis/fixed_bin_density.py itor_revision/data/test_portfolio_pref/test --out itor_revision/recomputed/fixed_bin_density.json
```

Pooling retains an already-present method's runs from the first phase and
adds only new method tokens from later phases; the ordering above matters.
The in-phase BANDITP3:8 reruns in `test_nonadaptive` are used for the matched
selection comparison, while the pooled sixteen-method analysis retains the
earlier `test_portfolio_pref` runs. Bootstrap intervals are descriptive,
geometry-stratified paired intervals. The paper uses 10,000 replicates; pass
`--replicates 10000` explicitly, because the monitoring script defaults to 2,000.
The pooled CSV records the five control tokens in the command above. The
ALNSR contrasts are in `reports/test_report_pooled/`, the earlier fourteen-method
pool using the same first three phases. To reproduce those intervals, omit
`test_nonadaptive` from `--extra-dirs` and use controls
`M0,M1,ALNS:small_beam,ALNS:alns_c8_b4_t25,BANDIT:8`. Ranks and all-method gaps
differ between the fourteen- and sixteen-method pools; primary paired
contrasts for their shared methods do not.

The density script groups **round start times into [0,20), [20,40), [40,60)**,
with equal weight per round. It reproduces 123,428 rounds from 708 runs and
the manuscript's rates (P4: 2.8933/0.0448/0.0188; G: 0.4999/0.0083/0.0042;
D: 0.9488/0.1297/0.0658). It does not infer causal operator response curves.

`code/make_tables_eaai.py` converts report CSVs to LaTeX tables. Its oracle
and summary routines expect M0 and M1; use those routines on the pooled or
frozen reports. Phases without both controls retain their CSV/Markdown reports.
`code/figures_eaai.py` generates anytime, regime and arm-share figures. For the
main figures, combine `test` with `test_portfolio_pref`, set
`--methods BANDITP3:8,M0,M1,ALNS:small_beam --bandit BANDITP3:8`, and use a new
`--out` directory. The archived figure labels may retain earlier short names.

## Audit and source provenance

The six study extension modules used in source fingerprints are copied without
byte changes. Together with the unchanged precursor modules, they match the
frozen implementation fingerprints of `test_nonadaptive`, `test_frozen` and
`test_flow_nonadaptive`. A fresh full frozen-suite audit was run when preparing
this release: 2,370 runs and 19,197 stored layouts passed, with zero objective
recomputation error. To repeat it while preserving the original audit:

```text
python itor_revision/code/audit_eaai.py itor_revision/data/test_frozen/test --out itor_revision/recomputed/frozen_audit.json
```

Create the output directory first if needed. Source fingerprints of earlier
phases differ because the implementation evolved. Their original protocols,
audits and file hashes are retained; this release does not reconstruct every
historical source snapshot. `--skip-source-check` permits a layout/objective
audit of those phases and explicitly records `sources_checked: false`. It
must not be described as reproducing the historical implementation hashes.

Public helper changes are limited to relative trainer paths, portable test
CPU affinity, coverage of matched selectors, an optional audit output path,
and the explicit fixed-bin analysis. The six fingerprinted study modules and
all reported experimental numbers are unchanged. The internally recorded
frozen criterion is documented in `records/FROZEN_CRITERION.md`; it was not
an independently timestamped public preregistration.

## Rerun search

`code/run_eaai_study.py --help` lists phase, method, geometry, fill, instance,
seed, objective and constraint options. Use each `records/*/protocol.json`
as the configuration record and a **new** output directory. Set `--cpus` and
`--workers` for the local machine; the default CPU list is specific to the
original machine. A changed runtime or implementation produces a new study,
not a byte-identical replacement for the archived wall-clock runs.
