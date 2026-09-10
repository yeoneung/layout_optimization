# Extended comparison protocol

This protocol is fixed before the test comparison. The extension evaluates
partial witness repair (M1) against full completion regeneration (M0) and a
layout-specific adaptive large neighborhood search (ALNS). Historical results
are retained separately. The new short-budget measurements come from the same
runs as the new 10-second and 60-second measurements.

Resource-allocation amendment, 10 September 2026: at the user's request, the
study now uses all eight physical CPU cores. Validation and test are rerun
under this common eight-worker setting in `results/completion_extension_full_cpu`.
The four-worker validation and interrupted test remain in
`results/completion_extension` and are excluded from the eight-worker test
analysis. Instance identities, method implementations, candidate configurations,
selection rule, search budgets and statistical analysis are unchanged. This
change is driven by available hardware, not by method performance on test data.

## Instances and repetitions

- Test: n = 32, 64, 128, 256, guillotine and nonslicing geometry, nominal fill
  0.90 and 0.95. There are 10 attempted instances per 0.90 cell and 20 per 0.95
  cell, totaling 240 attempted instances. Each initialized instance is evaluated
  with three seeds for each of three methods, at most 2,160 runs.
- Validation: n = 64, 128, both geometries and fills, two attempted instances
  per cell, two search seeds. Four ALNS configurations and the existing spatial
  repair controls B2 and B2S receive 10 seconds per run. The ALNS configuration
  with the largest mean improvement at 10 seconds is selected. Seeds are
  averaged within each instance before averaging instances. Ties are resolved
  by alphabetical configuration name. Test results are not used for selection.
- The existing generator is unchanged. `max_grid` is 44, 44, 72, 100 for the
  four sizes, respectively; sampled mean-area bounds remain 9 to 30. Integer
  dimensions make fill approximate. Achieved fill, plate size and facility
  area are retained for every instance. A deviation exceeding 0.01 aborts the
  study for inspection, rather than replacing an instance. The smoke test
  confirmed that an exact 0.001 fill tolerance would incorrectly reject some
  ordinary outputs of this existing generator.
- Full cell batches are regenerated, including the original once-per-cell
  objective-coefficient draw. Instance identity depends on split, tag, geometry,
  n, fill, batch size and index. Validation, smoke and test have distinct tags.
- The generated source packing is checked but withheld from every search.
  Best-contact initialization is identical for all methods. Initialization
  failures remain in the coverage denominator and are never replaced.

## Search and timing

- M0 and M1 use the archived implementation with kappa = 32, repair cap = 4,
  and one completion attempt. Neither implementation is retuned for the test.
- ALNS is a layout-specific adaptation, not a reproduction of a published
  routing solver. Its six operator pairs combine random, spatial and
  low-contribution removal with beam and regret insertion. It retains the best
  complete layout separately from its current layout, permits temperature-based
  nonimproving moves, and adapts operator weights online. Online adaptation is
  part of the fixed algorithm and does not select configurations from test data.
  The four configurations differ in removal cap and beam width; all are
  specified in `repair_baseline.py` before validation.
- One uninterrupted 60-second search records budgets 0, 0.1, 0.3, 1, 3, 10, 30,
  and 60 seconds. A completed candidate enters the trace only after independent
  geometric verification and objective evaluation both finish by the deadline.
  Incomplete or late proposals are not admitted. Search overrun is recorded.
- Initialization is measured separately and added for total-time reporting.
  The primary quality comparisons use identical post-initialization budgets.
  No source-generation or process-startup cost is presented as search time.
- Eight processes are each pinned to a different verified physical CPU core.
  All method/seed runs for one instance use the same core, in a deterministic
  shuffled order. NumPy/BLAS/OpenMP thread counts are one. Process CPU time,
  wall time and their ratio are stored; no low-ratio runs are silently removed.
  This is an eight-worker throughput setting, not a dedicated-machine latency
  claim. Runtime/package versions and source hashes are stored in each phase.

## Analysis and integrity

- Primary contrasts are M1 minus M0 and M1 minus ALNS at 10 and 60 seconds,
  separately by n and fill, pooling the two geometry families with equal weight.
  All individual geometry cells are also reported. Smaller budgets, 30 seconds,
  mechanism counts and timing distributions are secondary descriptive results.
- The experimental unit is the instance, not the evaluation seed. Each method's
  three seed results are averaged within instance. Paired bootstrap intervals
  resample instances within geometry; report descriptive 95% intervals without
  treating the family of intervals as multiplicity-adjusted hypothesis tests.
  Raw objective gains and percent improvements are both retained. The constant
  no-overlap bonus remains in the percent-improvement denominator.
- Report time to 1%, 5% and 10% improvement, hit rate by cutoff, and restricted
  mean time with failures right-censored at 60 seconds. Do not average times
  over successful runs alone. Initialization time is added for the companion
  total-time summaries.
- Every run is written atomically. A protocol/source hash check prevents mixed
  implementations on resume, and an exclusive runner lock prevents duplicates.
  A `STOP` file in the output root requests stopping between individual runs.
- A separate audit regenerates instances, checks coverage and all expected run
  identities, independently verifies every saved budget layout, recomputes its
  objective, and checks trace eligibility, monotonicity and common initialization.
  Missing results or failed audits block final tables. Partial progress reports
  are explicitly marked incomplete and cannot substitute for the final analysis.
- No result is promised to favor M1. A disappearing short-budget advantage or a
  stronger ALNS result changes the paper's claim, not the fixed test settings.

## Execution

From `rl_layout/experiments`, use the Python environment in
`requirements_extended.txt`:

```text
python finish_extended_study.py --out ../../results/completion_extension_full_cpu --workers 8 --run-validation
```

This command runs validation, audits it, selects its configuration, runs the test,
then audits and analyzes the complete test. Rerunning the same command resumes
completed shards under the unchanged protocol. Smoke outputs are stored in
separate directories and are not research observations.
