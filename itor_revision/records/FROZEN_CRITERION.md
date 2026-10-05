# Frozen-suite criterion and provenance

The passage below is an unchanged excerpt from the local study log, in the
entry dated 2026-10-03 titled "two controls added to answer the paper's own
stated gaps". That entry precedes the frozen-phase launch and result entries
in the log. It is an internal record, not an independently timestamped public
preregistration. This public deposit was prepared on 2026-10-05.

## Recorded criterion

*Criterion, fixed now.* The frozen test confirms the main finding if, at 60 s, PORT3
beats M1 in at least six of the eight pooled size-fill cells with intervals excluding
zero and is not worse than M0 by more than 1 pp in any cell. Anything else is reported
as a failure to confirm, with the cells named. The ALNS control and RR3 are carried
along so that the comparator ordering and the feedback question are also re-measured
out of sample. No method is tuned on this split and nothing in it is used to choose a
configuration.


## Interpretation in the final manuscript

PORT3 in this entry denotes PORT3-B (implementation token BANDITP3:8).
The first condition uses paired intervals; the second compares cell means.
The second condition is not a confidence-bound test of noninferiority.
The observed largest cell-mean deficit to M0 is about 0.46 percentage points.
Failure to exclude zero in PORT3-B versus rotation comparisons is not evidence
of equivalence. The primary paper reports no consistent advantage across the
three tested suites, with 3 positive, 5 negative and 40 unresolved cell/time
comparisons; the two time cutoffs are correlated prefixes of the same runs.

Source-log SHA-256 at packaging: 988136fe737b65a38780ad36dedf24f9be540ead2711bf6ce1e38a9d5cd214da
