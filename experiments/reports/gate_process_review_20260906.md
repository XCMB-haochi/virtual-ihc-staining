# Gate And Validation Process Review

Date: 2026-09-06

## Decision

The current gates should remain unchanged for the running Round114 protocol and
should not be relaxed retroactively for Round113. They are appropriate as a
conservative authorization for spending the remaining three folds or creating a
promotion package. They are not calibrated statistical tests and must not be
interpreted as proof that a failed candidate has no useful signal.

## Evidence

| Evidence | Result | What it shows |
| --- | --- | --- |
| Round44 two-fold screen | q95 `+0.155086`; five-fold `+0.140386`, 24/25 ROIs | The strict two-fold screen can retain a real transferable mechanism. |
| Round94 -> Round95 | two-fold `+0.074746`; five-fold `+0.019837`, 3/5 folds and 12/25 ROIs | An actual screen pass did not retain stability on five folds; this does not calibrate alternative thresholds. |
| Round85 complete five-fold | q95 `+0.026427`, 5/5 folds, 18/25 ROIs, worst `-0.033069` | The `20/25` ROI rule can reject a broad but small stable gain. |
| Round46 direct vs fixed ensemble | direct floor failed at ROI009 `-0.305986`; fixed ensemble gained `+0.108112` and all 25 ROIs vs Round35 | A mechanism gate and a final endpoint gate can answer different questions. |
| Round113 | q95 `+0.038374`, fold changes `-0.010961/+0.084600`, 5/10 ROIs, floor `-0.123136` | This is positive but unstable evidence, not a principled promotion candidate. |

Round113 is not rejected merely because one fold is slightly negative. It misses
several correlated conditions: the overall margin is below `+0.05`, only one
of two folds is positive, half of the ROIs improve, and the worst ROI misses the
floor by `0.003136`. Relaxing the floor alone would not turn it into a stable
result. The bottleneck hypothesis and the changed fusion location justify the
Round114 screen more than a gate exception does.

## What Is Strict And What Is Useful

The two-fold requirements of both folds positive, at least 7/10 ROIs positive,
and a worst-ROI floor are deliberately conservative sign checks. With only two
folds, they are underpowered estimates of generalization and can create false
negatives; ROI signs are also correlated and are not independent statistical
trials. The `+0.05` q95 margin is a resource policy, not a measured confidence
interval.

The strict screen remains useful because rejecting a failed structural candidate
saves roughly three-fifths of the five-fold training cost. Round94/95 shows why
automatic expansion from a modest two-fold gain would be unsafe. Checkpoint,
OOF coverage, source, and SHA checks are evidence contracts and should remain;
they are not the redundant part of the process.

## Process Change For Future Rounds

Do not change a gate after seeing a result. Report two-fold outcomes in three
categories:

- `clear negative`: a material aggregate decline supported by broad fold/ROI
  regressions; a tiny negative mean alone is inconclusive;
- `borderline positive`: aggregate is positive but stability gates fail;
- `screen pass`: the preregistered gate passes and unchanged five-fold expansion
  is authorized.

`borderline positive` does not authorize a five-fold run by itself. It preserves
  the mechanism as evidence and permits a separately justified confirmatory
  screen when no stronger route is available. Round113 belongs here; it does not
  belong in the submission path.

For a complete five-fold endpoint, keep the current strict promotion gate, but
add a separately named `stable_small_gain` review status when all folds are
positive, float is positive, the worst ROI floor passes, and only the positive
ROI-count or minimum-margin condition misses. This status may authorize one
fixed preliminary diagnostic package after provenance review; it must not
authorize weight searches, ROI routing, calibration, or leaderboard tuning.
Round85 is the motivating example. Round46 already used this separation for a
strong fixed ensemble whose direct mechanism dependency failed.

Before changing numeric thresholds, accumulate several more complete screens
and compare two-fold decisions with five-fold outcomes. Thresholds should be
changed prospectively in a new protocol, never to rescue an observed result.

## Redundancy Review

The expensive parts are justified when they answer different questions:

- real-data smoke catches environment and runtime failures;
- two-fold training screens structural candidates;
- complete OOF checks fold and ROI distribution;
- test inference confirms a fixed endpoint only after provenance.

The likely avoidable cost is recomputing an immutable baseline OOF for every
candidate with identical folds, inference context, TTA, offset, and checkpoint
hashes. A future cache can reuse that baseline only after validating those
identifiers and the evaluator source digest. Raw GPU polling, worker logs and
verbose recovery traces should remain operational files, as already required by
the project instructions.
