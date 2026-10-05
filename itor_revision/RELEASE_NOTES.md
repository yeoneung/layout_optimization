The ITOR revision adds certificate-preserving three-operator portfolios and
matched cyclic, uniform-random and sliding-window-bandit selection rules to
the precursor completion-aware layout archive.

This release includes code, recorded protocols, analyzed tables, fitted
models and 17 downloadable archives of raw run records and constructive
policy checkpoints (about 3.3 GiB compressed). DATASETS.json gives the byte
count and SHA-256 checksum of every archive. The pilot is a partial
exploratory phase and is identified as such in the documentation.

The archive covers the sixteen-method adjacency comparison, matched
selection controls, the frozen instance suite, flow costs, obstacles,
retuned ALNS and learned constructive comparisons. See
[the reproducibility guide](https://github.com/yeoneung/layout_optimization/blob/itor-v1.0.0/itor_revision/README.md)
for dataset mappings and exact commands.

Validation: four correctness-check programs passed; a fresh audit verified
2,370 frozen-suite runs and 19,197 stored layouts with zero objective
recomputation error. The fixed-bin density analysis reproduces all nine
reported rounded means. Detailed checks are recorded in VALIDATION.json.

The current implementation matches the recorded source hashes of all three
final matched-control phases. Earlier phases retain their original protocol
hashes and audit records, but their historical source snapshots are not all
reconstructed by this release. Their stored-layout audits can be repeated
with an explicitly documented source-check exemption. Correctness checks
and reanalysis are not new comparative wall-clock experiments.
