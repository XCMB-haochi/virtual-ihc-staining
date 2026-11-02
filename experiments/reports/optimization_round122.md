# Round122: Median-Normalized Relational Guidance

The registered config changed Round121's pair-distance scale from mean to
`torch.median` (lower median). Everything else remained fixed. This report
records the completed experiment; it is not a retrospective preregistration.
The earlier smoke-only launch did not chain into formal training as claimed
in chat. The subsequent Screen launch did complete the formal experiment.

Exit 0; four complete 14-epoch trajectories; selected checkpoints are finite;
source/protocol/checkpoint hashes match the launch and OOF receipts. OOF covers
2,706 images / 10 ROIs; local gate replay exactly matches. No producer remains.

| Measure vs Round55 auxiliary15 | Round121 | Round122 |
| --- | ---: | ---: |
| q95 gain | +0.093161 | +0.200209 |
| float gain | +0.093126 | +0.200030 |
| fold0 q95 gain | +0.025023 | +0.227592 |
| fold2 q95 gain | +0.157007 | +0.174551 |
| Positive ROIs | 7/10 | 8/10 |
| Worst ROI007 | -0.335121 | -0.178934 |

Round122 q95 is 75.213481 vs 75.013272. ROI-macro gain is +0.202691, so the
aggregate is not merely a patch-count weighting effect. ROI020 turns positive
(+0.178236); ROI014 remains negative (-0.088752). The fixed -0.12 floor fails;
all other checks pass. Preserve `promotion.allowed=false`; classify this as
promising but unconfirmed, not a clear negative or a safe submission.

The same two folds guided the follow-up choice. This is not an independent
confirmation of the gain, and the effect cannot be attributed uniquely to
outlier suppression: median normalization also changes gradient distribution
and effective loss scale. Round121 already applied this loss directly to the
encoder; the earlier suggestion to move it there was mistaken.

Decision: freeze Round122 and confirm on predefined fold1 before another
normalization/weight change or a larger-model run. Fold1 was used elsewhere in
project development; it is additional evidence for this route, not an untouched
final test set. New fold-local teacher training is required because the existing
fold0/fold2 teachers saw fold1 IHC. Never reuse those teachers for confirmation.

Measured student training cost is 3.24 hours for two folds; OOF 25.64 minutes.
The Round96 teachers cost 12.54/12.28 hours per fold. One new teacher plus one
student fold should cost about 14-16 hours without contention, saving two such
additional jobs if confirmation fails. Artifacts: `round122_launch.json`, exit,
OOF and gate JSON. Raw logs remain remote.

Implementation review found two unrelated regressions introduced with Round122:
the old perceptual path received an unsupported `robust_scale` argument, and
robust mode bypassed relational CLI/protocol restrictions. Both are repaired
before confirmation. Round122 used a supported relational constructor, stage4,
batch16 and no accumulation; neither defect invalidates its result. The actual
median loss and gradients are unchanged and regression-checked.
