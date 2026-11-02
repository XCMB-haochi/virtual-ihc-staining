# Round229: distinguish fitting error from held-out generalization error

Round228's full-fold gain shrank to0.06914 and was mixed. Stop the epoch
diagnostic chain. Before another local module, measure whether the retained
generator misses strong signals even in images it trained on.

Evaluate the exact Round208 and Round216 full-data epoch12 checkpoints on
all2380 permitted TRAIN images with the same D4/halo128/offset0/q95 output
encoding. Preserve complete DAPI ROI context and conditions, freeze models,
perform no optimizer update and use no test data. Report Gaussian11/old
uniform7 metrics, per-ROI performance and the established fixed intensity
strata from Round213, especially bias and error in pixels>=128.

These are explicitly SEEN-TRAINING scores, never OOF or submission estimates.
The Round217 fold-teacher OOF is a descriptive reference only: differences
also include additional ROI diversity, initialization and training budget,
so they cannot uniquely measure memorization. Poor training fit does not
establish an architecture ceiling; a train/OOF gap does not prove data shift.
It helps prioritize fitting/objective diagnostics versus generalization work.
No threshold fitting, output calibration or image editing is permitted.

One fixed model per GPU: Round208 on GPU0 and Round216 on GPU1, both idle
and sufficient memory at launch. No training or submission package; expect
roughly20-35min depending on D4/CPU scoring. Retain the best79.0198 package.

User mentioned another team's85.0182 total with no component scores or marker
available. This is an unverified contextual report, not evidence of cheating,
an evaluator bug, a specific model mechanism or a changed scoring rule. Do not
design experiments to imitate guessed behavior from this single total.

## Completed

Exit0; each model evaluated all2380 seen training images and17 ROIs.
Round208 seen-training score80.93051, versus fold-teacher OOF78.55263.
Round216 seen-training score80.13071. These differences are descriptive,
not a clean student generalization estimate or a comparison with another
team's hidden-set85.0182.

For the0.7002% of target pixels>=128, Round208 mean bias is-134.9833 and
RMSE145.794; these pixels contribute70.93% of SSE. Distilled Round216 bias
is-157.0252, RMSE164.458 and SSE share71.68%. Distillation weakens training
fit but slightly improves the platform score, consistent with a possible
regularizing effect, not proof of that mechanism. Strong-signal error is
present even on seen images, motivating a bounded fitting/objective probe.
