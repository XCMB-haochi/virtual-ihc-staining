# Round207: restore fold-local DAPI FCMAE800

Question: does new-stage domain reconstruction improve the CD45RO-primary
baseline and its CD68 auxiliary predictions? The experiment follows the
user's concern that domain FCMAE was omitted, and replaces no existing
reference. Round205 platform score77.9434 remains the confirmed baseline.

First finish the fixed Round206 two-fold D4 audit. Then run fold0 and fold3
independently: DAPI-only FCMAE800 on each fold's training ROIs, followed by
Round204's uniform7 supervised loss, CD45RO-heavy four-output weights,
stage1 epoch14 initialization, 14+14 cosine periods, and fixed stage2
epoch12 endpoint. Keep seed2026 and the same architecture. No validation
ROI image, prior-stage data or test image participates in pretraining.
No relational guidance or pixel distillation is included: this isolates
domain pretraining relative to the current supervised baseline.

Pretraining: tiny model, mask0.5, LR2e-4 to1e-6, 10 warmup epochs,
weight-decay0.05, effective batch32 via microbatch8/accumulation4,
workers0. Start from the identical epoch0 public-only encoder artifact
used by Round204, including its saved input stem. A new strict loader
checks hash, epoch0, public-only scope, encoder specification and all state
keys so that changing the random input stem does not confound the test.
Save resumable model/optimizer/scaler/RNG state every25 epochs. The existing
trainer supports `--resume` with matching configuration. Do not overwrite
existing runs; supervised checkpoints are saved every epoch.

800 epochs is a registered transfer-budget test, not a claim of optimality.
Fold0 sees1,960 images/epoch (1,568,000 views over800 epochs, 62 update
opportunities/epoch); fold3 sees1,820 (1,456,000 views, 57 updates/epoch).
This is not update-matched to Colon800. Same initialization plus additional
domain learning is the comparison against Round204's no-domain baseline.

A two-epoch, 256-image CUDA smoke verified 16 actual optimizer updates,
excluded validation ROIs, and a loadable final encoder. Steady smoke epoch
time7.64 seconds and peak allocation5.70 GiB imply roughly25 hours for
both full pretraining folds, plus approximately2-3 hours supervision and
inference under similar throughput. This is an extrapolation, not a
deadline guarantee; shared compute and I/O can increase it. Two folds
avoid paying the remaining three pretraining runs before evidence exists.
GPU1 concurrent execution only, preserving space for the other task's
20 GiB peak; GPU0 reserved.

After completion compare both CD45RO and CD68 at the same fixed endpoint
under D4. CD68 remains an auxiliary output here: improvement would motivate
a later matched CD68-primary experiment, not prove its best achievable
score. Evaluate paired ROI/fold changes; do not veto a method simply because
one ROI declines, and do not search epochs to rescue a negative outcome.
No automatic full-data training or submission is authorized by this screen.

## Completed

Pipeline exit 0. The preceding Round206 D4 audit was negative for CD45RO
(fold0 -0.14377, fold3 -0.06885) and is closed. Domain FCMAE800 plus matched
supervision then completed on both folds. At stage2 epoch12, CD45RO scores
were fold0 77.32281 and fold3 76.71147, versus Round204's 75.91182 and
75.33410: gains +1.41099 and +1.37737. CD68 changed to 68.92947 and
67.88437; this screen targets CD45RO and does not establish CD68's optimum.
The gain is replicated across both selected difficult folds, so expanding to
all training data is justified. There is no legal Round207 submission package.
